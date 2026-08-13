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


def test_cookies_attach_only_when_file_exists(monkeypatch, tmp_path):
    """YT_COOKIES_FILE (2026-07-30): a real file attaches cookiefile to the
    yt-dlp options; a dangling path or unset env leaves the options
    byte-identical to the cookie-less shape."""
    ck = tmp_path / "yt_cookies.txt"
    ck.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.delenv("YT_POT_SCRIPT", raising=False)
    monkeypatch.setenv("YT_COOKIES_FILE", str(ck))
    importlib.reload(detect_core)
    opts = detect_core._yt_opts("web")
    assert opts["cookiefile"] == str(ck)
    assert opts["extractor_args"] == {"youtube": {"player_client": ["web"]}}

    monkeypatch.setenv("YT_COOKIES_FILE", str(tmp_path / "missing.txt"))
    importlib.reload(detect_core)
    assert "cookiefile" not in detect_core._yt_opts("web")

    monkeypatch.delenv("YT_COOKIES_FILE", raising=False)
    importlib.reload(detect_core)
    assert "cookiefile" not in detect_core._yt_opts("android")
