#!/bin/bash
# One-command submission and monitoring; invoke from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${VENV_DIR:-$PWD/.venv}/bin/python" scripts/stage3_pipeline.py "$@"
