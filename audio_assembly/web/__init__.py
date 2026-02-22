"""Flask web application for Audio Assembly."""

import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
    Response,
)

from ..models import Config, ProcessingMode, SoftenMode
from ..pipeline import process_audio
from ..report import format_report_summary

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output_web")
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "flac", "ogg"}

# In-memory job store
jobs: dict = {}


def create_app() -> Flask:
    """Create and configure Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max
    app.secret_key = os.urandom(24)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Routes ──────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload():
        if "audio_file" not in request.files:
            return jsonify({"error": "Brak pliku audio"}), 400

        file = request.files["audio_file"]
        if file.filename == "":
            return jsonify({"error": "Nie wybrano pliku"}), 400

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Nieobsługiwany format: .{ext}"}), 400

        # Save uploaded file
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        safe_name = f"input.{ext}"
        filepath = os.path.join(job_dir, safe_name)
        file.save(filepath)

        return jsonify({
            "job_id": job_id,
            "filename": file.filename,
            "filepath": filepath,
        })

    @app.route("/process", methods=["POST"])
    def start_processing():
        data = request.get_json()
        job_id = data.get("job_id")
        if not job_id or job_id not in _get_upload_ids():
            return jsonify({"error": "Nieprawidłowy job_id"}), 400

        # Parse config from request
        mode_str = data.get("mode", "A")
        mode = (
            ProcessingMode.CLEAN_FILLERS_AND_PAUSES
            if mode_str == "A"
            else ProcessingMode.REMOVE_REPETITIONS_CONTEXTUAL
        )

        config = Config(
            mode=mode,
            silence_threshold_sec=float(data.get("silence_threshold_sec", 3.0)),
            min_gap_to_edit_ms=int(data.get("min_gap_to_edit_ms", 120)),
            crossfade_ms=int(data.get("crossfade_ms", 60)),
            soften_mode=SoftenMode[data.get("soften_mode", "ROOMTONE").upper()],
            soften_ms=int(data.get("soften_ms", 120)),
            max_pause_after_edit_sec=float(data.get("max_pause_after_edit_sec", 0.4)),
            start_offset_sec=_parse_optional_float(data.get("start_offset_sec")),
            end_offset_sec=_parse_optional_float(data.get("end_offset_sec")),
            whisper_model=data.get("whisper_model", "base"),
            strict_mode=bool(data.get("strict_mode", False)),
            confidence_threshold=float(data.get("confidence_threshold", 0.75)),
        )

        # Find uploaded file
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        input_files = [f for f in os.listdir(job_upload_dir) if f.startswith("input.")]
        if not input_files:
            return jsonify({"error": "Nie znaleziono pliku wejściowego"}), 400

        input_file = os.path.join(job_upload_dir, input_files[0])
        output_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize job state
        job_state = {
            "id": job_id,
            "status": "processing",
            "progress": 0,
            "step": "Inicjalizacja...",
            "error": None,
            "report": None,
            "output_dir": output_dir,
            "events": queue.Queue(),
        }
        jobs[job_id] = job_state

        # Run processing in background thread
        thread = threading.Thread(
            target=_run_processing,
            args=(job_id, input_file, output_dir, config),
            daemon=True,
        )
        thread.start()

        return jsonify({"job_id": job_id, "status": "processing"})

    @app.route("/progress/<job_id>")
    def progress_stream(job_id):
        """Server-Sent Events stream for real-time progress."""
        if job_id not in jobs:
            return jsonify({"error": "Nieznane zadanie"}), 404

        def generate():
            job = jobs[job_id]
            while True:
                try:
                    event = job["events"].get(timeout=30)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("status") in ("done", "error"):
                        break
                except queue.Empty:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/results/<job_id>")
    def results(job_id):
        if job_id not in jobs:
            return render_template("index.html")
        job = jobs[job_id]
        return render_template("results.html", job=job, job_id=job_id)

    @app.route("/api/results/<job_id>")
    def api_results(job_id):
        if job_id not in jobs:
            return jsonify({"error": "Nieznane zadanie"}), 404
        job = jobs[job_id]
        result = {
            "id": job["id"],
            "status": job["status"],
            "error": job["error"],
        }
        if job["report"]:
            result["report"] = job["report"]
        return jsonify(result)

    @app.route("/download/<job_id>/<filename>")
    def download_file(job_id, filename):
        output_dir = os.path.join(OUTPUT_DIR, job_id)
        if not os.path.isfile(os.path.join(output_dir, filename)):
            return jsonify({"error": "Plik nie istnieje"}), 404
        return send_from_directory(
            output_dir, filename, as_attachment=True
        )

    @app.route("/stream/<job_id>/<filename>")
    def stream_audio(job_id, filename):
        output_dir = os.path.join(OUTPUT_DIR, job_id)
        if not os.path.isfile(os.path.join(output_dir, filename)):
            return jsonify({"error": "Plik nie istnieje"}), 404
        return send_from_directory(output_dir, filename)

    @app.route("/stream_input/<job_id>/<filename>")
    def stream_input_audio(job_id, filename):
        upload_dir = os.path.join(UPLOAD_DIR, job_id)
        if not os.path.isfile(os.path.join(upload_dir, filename)):
            return jsonify({"error": "Plik nie istnieje"}), 404
        return send_from_directory(upload_dir, filename)

    return app


# ── Background processing ──────────────────────────────────────

def _run_processing(job_id: str, input_file: str, output_dir: str, config: Config):
    """Run audio processing in a background thread with progress events."""
    job = jobs[job_id]
    eq = job["events"]

    class ProgressHandler(logging.Handler):
        """Capture log messages as progress events."""
        step_map = {
            "Step 1": ("Wczytywanie audio...", 10),
            "Step 2": ("Wykrywanie mowy (VAD)...", 25),
            "Step 3": ("Transkrypcja PL...", 40),
            "Step 4": ("Analiza i budowa edycji...", 60),
            "Step 5": ("Renderowanie audio...", 75),
            "Step 7": ("Zapisywanie plików...", 90),
        }

        def emit(self, record):
            msg = record.getMessage()
            for key, (label, pct) in self.step_map.items():
                if key in msg:
                    job["progress"] = pct
                    job["step"] = label
                    eq.put({
                        "type": "progress",
                        "progress": pct,
                        "step": label,
                    })
                    break

    handler = ProgressHandler()
    handler.setLevel(logging.INFO)
    logging.getLogger("audio_assembly").addHandler(handler)

    try:
        eq.put({"type": "progress", "progress": 5, "step": "Start przetwarzania..."})

        report = process_audio(input_file, output_dir, config)

        # Build result data
        report_dict = report.to_dict()
        summary = format_report_summary(report)

        # List output files
        output_files = []
        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            output_files.append({
                "name": fname,
                "size": os.path.getsize(fpath),
                "is_audio": fname.endswith((".wav", ".mp3", ".flac")),
                "is_json": fname.endswith(".json"),
                "is_text": fname.endswith(".txt"),
            })

        job["status"] = "done"
        job["progress"] = 100
        job["step"] = "Gotowe!"
        job["report"] = {
            "summary": summary,
            "data": report_dict,
            "files": output_files,
        }

        eq.put({
            "type": "done",
            "status": "done",
            "progress": 100,
            "report": job["report"],
        })

    except Exception as e:
        logger.exception("Processing error for job %s", job_id)
        job["status"] = "error"
        job["error"] = str(e)
        eq.put({
            "type": "error",
            "status": "error",
            "error": str(e),
        })
    finally:
        logging.getLogger("audio_assembly").removeHandler(handler)


def _get_upload_ids():
    """Get list of upload job IDs."""
    if not os.path.exists(UPLOAD_DIR):
        return set()
    return set(os.listdir(UPLOAD_DIR))


def _parse_optional_float(val):
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def run_web(host="0.0.0.0", port=5000, debug=False):
    """Run the web application."""
    app = create_app()
    app.run(host=host, port=port, debug=debug, threaded=True)
