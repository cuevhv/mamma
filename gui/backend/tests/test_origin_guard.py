"""Browser-origin guard and output-document sandbox on the GUI backend.

Another site (or an opaque-origin page such as a sandboxed output HTML) must
not be able to drive the API through the user's browser, while the UI's own
origin and the Vite dev proxy keep working.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]  # gui/backend


@pytest.fixture(scope="module")
def gui(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gui")
    env = {
        "MAMMA_DATA_DIR": str(tmp),
        "MAMMA_DB_PATH": str(tmp / "db.sqlite"),
        "MAMMA_INTERFACE_DIR": str(tmp / "interface"),
        "MAMMA_DEBUG": "1",  # reloader-parent branch: no queue coordinator thread
        "MAMMA_ALLOWED_ORIGINS": "https://mamma.lab.example/",
    }
    saved = {k: os.environ.get(k) for k in (*env, "WERKZEUG_RUN_MAIN")}
    os.environ.update(env)
    os.environ.pop("WERKZEUG_RUN_MAIN", None)
    sys.path.insert(0, str(_BACKEND))
    try:
        mod = importlib.import_module("app")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    secret = tmp / "secret.txt"
    secret.write_text("secret")
    html = tmp / "report.html"
    html.write_text("<script>fetch('/api/files/content?path=/etc/hostname')</script>")
    return mod.app.test_client(), secret, html


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "file://"])
def test_foreign_origin_refused(gui, origin):
    client, secret, _ = gui
    r = client.get(f"/api/files/content?path={secret}", headers={"Origin": origin})
    assert r.status_code == 403
    assert "Access-Control-Allow-Origin" not in r.headers
    r = client.post("/api/processes/1/stop", headers={"Origin": origin, "Content-Type": "text/plain"})
    assert r.status_code == 403


@pytest.mark.parametrize("headers", [
    {},                                              # same-origin GET, no Origin sent
    {"Origin": "http://localhost"},                  # same origin as the test client's Host
    {"Origin": "http://localhost:3000"},             # Vite dev proxy
    {"Origin": "http://127.0.0.1:5173"},
    {"Origin": "https://mamma.lab.example"},         # MAMMA_ALLOWED_ORIGINS
    {"Origin": "http://gpu-box.lan:8000", "Host": "gpu-box.lan:8000"},  # LAN / tunnel, same origin
])
def test_trusted_origins_allowed(gui, headers):
    client, secret, _ = gui
    r = client.get(f"/api/files/content?path={secret}", headers=headers)
    assert r.status_code == 200
    assert r.get_json()["content"] == "secret"


def test_output_html_is_sandboxed(gui):
    client, _, html = gui
    r = client.get(f"/api/files/html?path={html}")
    assert r.status_code == 200
    assert r.headers["Content-Security-Policy"].startswith("sandbox")
    assert "allow-same-origin" not in r.headers["Content-Security-Policy"]


def test_single_process_stop_is_refused(gui):
    """Stopping one step would only relabel its row while the runner keeps
    going, so the endpoint refuses and points at the whole-task stop."""
    client, _, _ = gui
    r = client.post("/api/processes/1/stop")
    assert r.status_code == 409
    assert "/api/tasks/<task_id>/stop" in r.get_json()["error"]
