#!/usr/bin/env bash
# Start the OCR API (port 8000) and the web app (port 3000) together. Ctrl+C stops both.
#   ./start.sh             production build of the web app
#   ./start.sh --dev       Next.js dev server with hot reload
set -euo pipefail
cd "$(dirname "$0")"
ROOT=..
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"

"$PY" -c "import fastapi, uvicorn, multipart, pillow_heif, ultralytics" 2>/dev/null || "$PY" -m pip install -r backend/requirements.txt
[ -d frontend/node_modules ] || (cd frontend && npm install)

trap 'kill 0' EXIT
(cd backend && "../$PY" -m uvicorn app:app --host 127.0.0.1 --port 8000) &
if [ "${1:-}" = "--dev" ]; then
  (cd frontend && npx next dev -p 3000) &
else
  (cd frontend && npx next build && npx next start -p 3000) &
fi
echo "Open http://localhost:3000 once both servers are up."
wait
