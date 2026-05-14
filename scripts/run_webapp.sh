#!/usr/bin/env bash
set -euo pipefail
exec python -m streamlit run "$(dirname "$0")/../src/embedcluster/webapp/app.py" "$@"
