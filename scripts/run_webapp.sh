#!/usr/bin/env bash
set -euo pipefail
exec streamlit run "$(dirname "$0")/../src/embedcluster/webapp/app.py" "$@"
