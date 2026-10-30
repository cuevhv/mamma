#!/usr/bin/env bash
# Start the MAMMA webtool from inside mamma_release: Flask backend on
# :8000 and Vite dev server on :3000 (which proxies /api -> :8000).
# Ctrl-C cleans up both.
#
# Expects the `mamma` conda env to be active (or at least on PATH) —
# the env ships Flask, python-dotenv, pyyaml, and Node 20
# so both halves of the GUI run out of a single env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # gui/
REPO_ROOT="$(cd "$ROOT/.." && pwd)"                              # mamma_release/
cd "$ROOT"

# Default data dir lives inside gui/var/ — keeps GUI state next to the
# GUI checkout and a `git clean -fdx` resets the slate. Override by
# exporting MAMMA_DATA_DIR before running this script. gui/var/ is
# gitignored.
export MAMMA_DATA_DIR="${MAMMA_DATA_DIR:-$ROOT/var}"
mkdir -p "$MAMMA_DATA_DIR"

# Where the webtool reads task templates / capture jsons / writes the
# per-task generated configs. Override by exporting MAMMA_INTERFACE_DIR
# before running this script.
export MAMMA_INTERFACE_DIR="${MAMMA_INTERFACE_DIR:-$MAMMA_DATA_DIR/interface}"
mkdir -p "$MAMMA_INTERFACE_DIR/capture_jsons" "$MAMMA_INTERFACE_DIR/task_jsons" "$MAMMA_INTERFACE_DIR/samples/tasks"

# Seed the preset picker on first launch by copying the parent repo's
# example task files into samples/tasks/. Skipped if the dir already
# has presets — user-added presets are never overwritten.
if [ -z "$(ls -A "$MAMMA_INTERFACE_DIR/samples/tasks" 2>/dev/null)" ]; then
  for f in "$REPO_ROOT"/configs/examples/quick_tasks/140725_Breakdance.yaml \
           "$REPO_ROOT"/configs/examples/tasks/140725_Breakdance.yaml; do
    if [ -f "$f" ]; then
      cp "$f" "$MAMMA_INTERFACE_DIR/samples/tasks/"
      echo "==> seeded preset: $(basename "$f")"
    fi
  done
fi

# Default task template the New Task form falls back to when nothing is
# selected. Points at the example task that ships in configs/examples/.
export MAMMA_DEFAULT_TASK_JSON="${MAMMA_DEFAULT_TASK_JSON:-$REPO_ROOT/configs/examples/quick_tasks/140725_Breakdance.yaml}"

echo "==> MAMMA_DATA_DIR=$MAMMA_DATA_DIR"
echo "==> MAMMA_INTERFACE_DIR=$MAMMA_INTERFACE_DIR"
echo "==> MAMMA_DEFAULT_TASK_JSON=$MAMMA_DEFAULT_TASK_JSON"

_cleaned=0
cleanup() {
  [ "$_cleaned" = "1" ] && return
  _cleaned=1
  echo
  echo "==> stopping dev servers"
  kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(
  cd backend
  echo "==> backend: python app.py (http://localhost:8000)"
  python app.py
) &
BACKEND_PID=$!

(
  cd frontend
  if [ ! -d node_modules ]; then
    echo "==> frontend: npm install"
    npm install
  fi
  echo "==> frontend: npm run dev (http://localhost:3000)"
  npm run dev
) &
FRONTEND_PID=$!

wait $BACKEND_PID $FRONTEND_PID
