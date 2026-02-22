/* ═══════════════════════════════════════════════════
   Audio Assembly — Web UI (JavaScript)
   ═══════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── State ──────────────────────────────────────
  let state = {
    jobId: null,
    filename: null,
    mode: "A",
    uploadReady: false,
  };

  // ── DOM refs ───────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const dropZone     = $("#drop-zone");
  const fileInput    = $("#file-input");
  const fileSelected = $("#file-selected");
  const fileName     = $("#file-name");
  const fileSize     = $("#file-size");
  const fileRemove   = $("#file-remove");

  const btnStart   = $("#btn-start");
  const btnNew     = $("#btn-new");
  const btnRetry   = $("#btn-retry");

  const stepUpload     = $("#step-upload");
  const stepSettings   = $("#step-settings");
  const stepProcessing = $("#step-processing");
  const stepResults    = $("#step-results");
  const stepError      = $("#step-error");

  const progressBar  = $("#progress-bar");
  const progressPct  = $("#progress-pct");
  const progressStep = $("#progress-step");

  const confidenceSlider = $("#confidence_threshold");
  const confidenceVal    = $("#confidence_val");

  // ── Upload ─────────────────────────────────────
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });
  fileRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    const allowed = ["wav", "mp3", "m4a", "flac", "ogg"];
    if (!allowed.includes(ext)) {
      alert("Nieobsługiwany format pliku. Dozwolone: " + allowed.join(", "));
      return;
    }

    state.filename = file.name;
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    dropZone.querySelector(".upload-zone__content").hidden = true;
    fileSelected.hidden = false;
    state.uploadReady = false;

    // Upload to server
    const form = new FormData();
    form.append("audio_file", file);

    fetch("/upload", { method: "POST", body: form })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          alert(data.error);
          resetUpload();
          return;
        }
        state.jobId = data.job_id;
        state.uploadReady = true;
        btnStart.disabled = false;
        stepSettings.classList.add("active");
      })
      .catch((err) => {
        alert("Błąd przesyłania: " + err.message);
        resetUpload();
      });
  }

  function resetUpload() {
    state.jobId = null;
    state.filename = null;
    state.uploadReady = false;
    fileInput.value = "";
    dropZone.querySelector(".upload-zone__content").hidden = false;
    fileSelected.hidden = true;
    btnStart.disabled = true;
  }

  // ── Mode selector ──────────────────────────────
  $$(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".mode-btn").forEach((b) => b.classList.remove("mode-btn--active"));
      btn.classList.add("mode-btn--active");
      state.mode = btn.dataset.mode;
    });
  });

  // ── Soften radio cards ─────────────────────────
  $$('input[name="soften_mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      $$(".radio-card").forEach((c) => c.classList.remove("radio-card--active"));
      radio.closest(".radio-card").classList.add("radio-card--active");
    });
  });

  // ── Confidence slider ──────────────────────────
  confidenceSlider.addEventListener("input", () => {
    confidenceVal.textContent = parseFloat(confidenceSlider.value).toFixed(2);
  });

  // ── Start processing ──────────────────────────
  btnStart.addEventListener("click", startProcessing);

  function startProcessing() {
    if (!state.uploadReady || !state.jobId) return;

    const config = {
      job_id: state.jobId,
      mode: state.mode,
      silence_threshold_sec: parseFloat($("#silence_threshold").value),
      max_pause_after_edit_sec: parseFloat($("#max_pause").value),
      crossfade_ms: parseInt($("#crossfade_ms").value),
      soften_mode: document.querySelector('input[name="soften_mode"]:checked').value,
      soften_ms: parseInt($("#soften_ms").value),
      confidence_threshold: parseFloat(confidenceSlider.value),
      whisper_model: $("#whisper_model").value,
      strict_mode: $("#strict_mode").checked,
      start_offset_sec: $("#start_offset").value || null,
      end_offset_sec: $("#end_offset").value || null,
    };

    // Show processing panel
    showStep("processing");

    fetch("/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          showError(data.error);
          return;
        }
        listenProgress(data.job_id);
      })
      .catch((err) => showError(err.message));
  }

  // ── SSE progress ──────────────────────────────
  function listenProgress(jobId) {
    const evtSource = new EventSource("/progress/" + jobId);

    evtSource.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.type === "heartbeat") return;

      if (data.type === "progress") {
        updateProgress(data.progress, data.step);
      }

      if (data.type === "done") {
        evtSource.close();
        updateProgress(100, "Gotowe!");
        setTimeout(() => showResults(jobId, data.report), 500);
      }

      if (data.type === "error") {
        evtSource.close();
        showError(data.error);
      }
    };

    evtSource.onerror = () => {
      evtSource.close();
      // Try to get the final status
      fetch("/api/results/" + jobId)
        .then((r) => r.json())
        .then((data) => {
          if (data.status === "done" && data.report) {
            showResults(jobId, data.report);
          } else if (data.error) {
            showError(data.error);
          }
        })
        .catch(() => showError("Utracono połączenie z serwerem."));
    };
  }

  function updateProgress(pct, step) {
    progressBar.style.width = pct + "%";
    progressPct.textContent = pct;
    progressStep.textContent = step;
  }

  // ── Results ───────────────────────────────────
  function showResults(jobId, report) {
    showStep("results");

    const data = report.data;
    const files = report.files || [];

    // Stats cards
    const statsHtml = `
      <div class="stat-card">
        <div class="stat-card__value">${fmtDuration(data.original_duration_sec)}</div>
        <div class="stat-card__label">Czas oryginału</div>
      </div>
      <div class="stat-card stat-card--accent">
        <div class="stat-card__value">${fmtDuration(data.processed_duration_sec)}</div>
        <div class="stat-card__label">Czas po montażu</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__value">${data.total_removed_sec.toFixed(1)}s</div>
        <div class="stat-card__label">Usunięto</div>
      </div>
      ${data.fillers_removed > 0 ? `
        <div class="stat-card">
          <div class="stat-card__value">${data.fillers_removed}</div>
          <div class="stat-card__label">Wypełniacze usunięte</div>
        </div>` : ""}
      ${data.pauses_shortened > 0 ? `
        <div class="stat-card">
          <div class="stat-card__value">${data.pauses_shortened}</div>
          <div class="stat-card__label">Pauzy skrócone</div>
        </div>` : ""}
      ${data.repetitions_removed > 0 ? `
        <div class="stat-card">
          <div class="stat-card__value">${data.repetitions_removed}</div>
          <div class="stat-card__label">Powtórzenia usunięte</div>
        </div>` : ""}
    `;
    $("#results-stats").innerHTML = statsHtml;

    // Audio players
    const audioFile = files.find((f) => f.is_audio);
    const inputFiles = [];
    // Find the input file for comparison
    try {
      const ext = state.filename.split(".").pop().toLowerCase();
      $("#player-original").src = `/stream_input/${jobId}/input.${ext}`;
    } catch (e) { /* ignore */ }
    if (audioFile) {
      $("#player-result").src = `/stream/${jobId}/${audioFile.name}`;
    }

    // Edits table
    const edits = data.edits || [];
    if (edits.length > 0) {
      let tableHtml = `<table class="edits-table">
        <thead><tr>
          <th>Czas</th><th>Typ</th><th>Tekst</th><th>Czas trw.</th><th>Pewność</th><th>Status</th>
        </tr></thead><tbody>`;

      edits.forEach((e) => {
        const typeBadge = getTypeBadge(e.type);
        const statusBadge = getStatusBadge(e.status);
        tableHtml += `<tr>
          <td style="font-family:var(--mono);white-space:nowrap">${fmtTime(e.start)}</td>
          <td>${typeBadge}</td>
          <td>${escHtml(e.original_text || e.description).substring(0, 60)}</td>
          <td style="font-family:var(--mono)">${e.duration.toFixed(2)}s</td>
          <td style="font-family:var(--mono)">${(e.confidence * 100).toFixed(0)}%</td>
          <td>${statusBadge}</td>
        </tr>`;
      });

      tableHtml += "</tbody></table>";
      $("#edits-table-wrap").innerHTML = tableHtml;
    } else {
      $("#edits-table-wrap").innerHTML = '<p class="text-muted">Brak edycji</p>';
    }

    // Download cards
    let dlHtml = "";
    files.forEach((f) => {
      const icon = f.is_audio
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>'
        : f.is_json
          ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>'
          : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';

      dlHtml += `
        <a class="download-card" href="/download/${jobId}/${f.name}" download>
          ${icon}
          <div class="download-card__info">
            <div class="download-card__name">${f.name}</div>
            <div class="download-card__size">${formatSize(f.size)}</div>
          </div>
        </a>`;
    });
    $("#download-grid").innerHTML = dlHtml;
  }

  // ── Error handling ────────────────────────────
  function showError(msg) {
    showStep("error");
    $("#error-msg").textContent = msg;
  }

  // ── Step navigation ───────────────────────────
  function showStep(name) {
    [stepUpload, stepSettings, stepProcessing, stepResults, stepError].forEach(
      (el) => (el.hidden = true)
    );

    if (name === "upload") {
      stepUpload.hidden = false;
      stepSettings.hidden = false;
    } else if (name === "processing") {
      stepUpload.hidden = false;
      stepSettings.hidden = false;
      stepProcessing.hidden = false;
      // Disable controls during processing
      btnStart.disabled = true;
    } else if (name === "results") {
      stepResults.hidden = false;
    } else if (name === "error") {
      stepError.hidden = false;
    }
  }

  // ── New / Retry ───────────────────────────────
  btnNew && btnNew.addEventListener("click", () => {
    resetUpload();
    updateProgress(0, "Oczekiwanie...");
    [stepProcessing, stepResults, stepError].forEach((el) => (el.hidden = true));
    stepUpload.hidden = false;
    stepSettings.hidden = false;
    stepSettings.classList.remove("active");
  });

  btnRetry && btnRetry.addEventListener("click", () => {
    updateProgress(0, "Oczekiwanie...");
    stepError.hidden = true;
    stepUpload.hidden = false;
    stepSettings.hidden = false;
    if (state.uploadReady) btnStart.disabled = false;
  });

  // ── Helpers ───────────────────────────────────
  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function fmtDuration(sec) {
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(2);
    return `${String(m).padStart(2, "0")}:${s.padStart(5, "0")}`;
  }

  function escHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function getTypeBadge(type) {
    const map = {
      filler_removal: ["Wypełniacz", "badge--filler"],
      pause_shortening: ["Pauza", "badge--pause"],
      repetition_removal: ["Powtórzenie", "badge--repeat"],
      sentence_restart_removal: ["Restart", "badge--restart"],
    };
    const [label, cls] = map[type] || [type, ""];
    return `<span class="badge ${cls}">${label}</span>`;
  }

  function getStatusBadge(status) {
    const map = {
      applied: ["Zastosowano", "badge--applied"],
      review: ["Do przeglądu", "badge--review"],
      skipped: ["Pominięto", "badge--skipped"],
    };
    const [label, cls] = map[status] || [status, ""];
    return `<span class="badge ${cls}">${label}</span>`;
  }
})();
