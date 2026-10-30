#!/usr/bin/env bash
# Start the MAMMA webtool in production mode: a single waitress WSGI
# worker on :8000 serving both the JSON API and the pre-built React
# bundle.
#
# Difference vs dev.sh:
#   * No Vite dev server — runs `npm run build` once, then app.py's
#     static-serve fallback serves gui/frontend/build/ at "/".
#   * waitress instead of Flask's built-in Werkzeug dev server.
#     Multi-threaded, no interactive debugger, no auto-reloader, no
#     "this is a development server" warning. UI changes require
#     re-running this script (or `npm run build` manually) to show up.
#   * One port, one process. Lower memory, snappier loads, no source
#     maps in DevTools.
#
# Expects the `mamma` conda env to be active (or at least on PATH) —
# the env ships Flask, python-dotenv, pyyaml, waitress, and
# Node 20 so the build and the server both run from a single env.
#
# Flags:
#   --skip-build   reuse gui/frontend/build/ if present; only run
#                  `npm run build` when the dir is missing.
#
# Override the bind address via MAMMA_BIND_HOST / MAMMA_BIND_PORT.
# Binding to 0.0.0.0 exposes the file-read endpoints to the network;
# only do that on a trusted host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # gui/
REPO_ROOT="$(cd "$ROOT/.." && pwd)"                              # mamma_release/
cd "$ROOT"

SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help)
      sed -n '1,30p' "$0"  # print the banner
      exit 0 ;;
    *)
      echo "Unknown flag: $arg" >&2
      echo "Usage: $0 [--skip-build]" >&2
      exit 2 ;;
  esac
done

# Same data + interface dirs as dev.sh so a switch between the two
# modes doesn't lose history.
export MAMMA_DATA_DIR="${MAMMA_DATA_DIR:-$ROOT/var}"
mkdir -p "$MAMMA_DATA_DIR"
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

export MAMMA_DEFAULT_TASK_JSON="${MAMMA_DEFAULT_TASK_JSON:-$REPO_ROOT/configs/examples/quick_tasks/140725_Breakdance.yaml}"
# Production default: no Flask reloader / debugger. Override with
# MAMMA_DEBUG=1 if you really need it.
export MAMMA_DEBUG="${MAMMA_DEBUG:-0}"

# Build the React bundle. `npm run build` lands in gui/frontend/build/
# and app.py's static-serve fallback picks it up at /.
BUILD_DIR="$ROOT/frontend/build"
if [ "$SKIP_BUILD" = "1" ] && [ -d "$BUILD_DIR" ]; then
  echo "==> reusing existing build: $BUILD_DIR"
else
  if [ "$SKIP_BUILD" = "1" ]; then
    echo "==> --skip-build set but $BUILD_DIR is missing; building anyway"
  fi
  if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "==> frontend: npm install"
    (cd "$ROOT/frontend" && npm install)
  fi
  echo "==> frontend: npm run build"
  (cd "$ROOT/frontend" && npm run build)
fi

BIND_HOST="${MAMMA_BIND_HOST:-127.0.0.1}"
BIND_PORT="${MAMMA_BIND_PORT:-8000}"

echo "==> MAMMA_DATA_DIR=$MAMMA_DATA_DIR"
echo "==> MAMMA_INTERFACE_DIR=$MAMMA_INTERFACE_DIR"
echo "==> MAMMA_DEFAULT_TASK_JSON=$MAMMA_DEFAULT_TASK_JSON"
echo "==> MAMMA_BIND_HOST=$BIND_HOST"
echo "==> MAMMA_BIND_PORT=$BIND_PORT"
echo "==> MAMMA_DEBUG=$MAMMA_DEBUG"
echo "==> bundle: $BUILD_DIR"
echo "==> open: http://$BIND_HOST:$BIND_PORT/"

# Replace the shell with waitress so Ctrl-C reaches the server directly
# and the script's PID == waitress' PID (handy for systemd / nohup
# wrappers). Run from gui/backend/ so the `app` module (app.py) and the
# sibling `db.py`, `sync.py`, `sinks.py`, `objects/` packages all
# resolve via the cwd-on-path convention the GUI uses. `app:app` =
# module `app`, attribute `app` (the Flask instance at module scope).
cd "$ROOT/backend"
exec waitress-serve --listen="$BIND_HOST:$BIND_PORT" app:app
