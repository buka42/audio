#!/usr/bin/env python3
"""Run the Audio Assembly web UI."""

import logging
import sys

from audio_assembly.web import run_web

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"\n  Audio Assembly — Web UI")
    print(f"  Otwórz: http://localhost:{port}\n")
    run_web(host="0.0.0.0", port=port, debug=False)
