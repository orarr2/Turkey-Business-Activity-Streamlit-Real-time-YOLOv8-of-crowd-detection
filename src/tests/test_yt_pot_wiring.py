"""PO-token wiring (2026-07-29): the bgutil script-mode extractor arg is
attached ONLY when YT_POT_SCRIPT is set - unset env keeps resolve_youtube's
options byte-identical to the pre-POT behavior (every VM without the
provider installed must keep working unchanged)."""
import importlib

from app import detect_core


def _reload_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("YT_POT_SCRIPT", raising=False)
    else:
        monkeypatch.setenv("YT_POT_SCRIPT", value)
    importlib.reload(detect_core)
    return detect_core


def test_unset_env_keeps_legacy_extractor_args(monkeypatch):
    dc = _reload_with(monkeypatch, None)
    args = dc._yt_extractor_args("android")
    assert args == {"youtube": {"player_client": ["android"]}}


def test_set_env_attaches_bgutil_script_arg(monkeypatch):
    dc = _reload_with(monkeypatch, "/opt/pot/server/build/generate_once.js")
    args = dc._yt_extractor_args("web")
    assert args["youtube"] == {"player_client": ["web"]}
    assert args["youtubepot-bgutilscript"] == {
        "script_path": ["/opt/pot/server/build/generate_once.js"]}


def test_client_name_is_stripped(monkeypatch):
    dc = _reload_with(monkeypatch, None)
    assert dc._yt_extractor_args(" tv ")["youtube"]["player_client"] == ["tv"]


def teardown_module(_m):
    # Leave the module in the env-driven state other tests expect.
    import os
    os.environ.pop("YT_POT_SCRIPT", None)
    importlib.reload(detect_core)
