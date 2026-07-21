"""IBB proxy rewriting: only IBB hosts get routed, only when both env
vars are set, and only when the URL is intact - nothing else moves.

The direct network chain (worker -> IBB origin) is verified end-to-end
by tools/probe_country from the VM after the operator sets up the
Cloudflare Worker; here we only pin the local rewriting rules, which
are pure and cheap to check."""
import importlib
import os

import pytest

from app import detect_core


@pytest.fixture()
def proxy_env(monkeypatch):
    monkeypatch.setenv("IBB_PROXY_URL", "https://ibb-proxy.example.workers.dev")
    monkeypatch.setenv("IBB_PROXY_SECRET", "s3cret")
    importlib.reload(detect_core)
    yield detect_core
    monkeypatch.delenv("IBB_PROXY_URL", raising=False)
    monkeypatch.delenv("IBB_PROXY_SECRET", raising=False)
    importlib.reload(detect_core)


def test_ibb_url_rewritten_with_secret_header(proxy_env):
    url, hdrs = proxy_env._apply_ibb_proxy(
        "https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8",
        None)
    assert url == ("https://ibb-proxy.example.workers.dev/"
                   "https://kamerayayin.ibb.istanbul/turistikcam/"
                   "taksim.stream/playlist.m3u8")
    assert hdrs["X-Proxy-Secret"] == "s3cret"


def test_ibb_rewrite_preserves_caller_headers(proxy_env):
    url, hdrs = proxy_env._apply_ibb_proxy(
        "https://kamerayayin.ibb.istanbul/x.m3u8",
        {"Referer": "https://player.tvkur.com/"})
    assert "Referer" in hdrs                       # untouched
    assert hdrs["X-Proxy-Secret"] == "s3cret"      # added


def test_non_ibb_hosts_pass_through(proxy_env):
    for u in ("https://www.youtube.com/watch?v=abc",
              "https://webcamera24.com/camera/turkey/foo/",
              "https://content.tvkur.com/l/id/master.m3u8",
              "https://firestore.googleapis.com/..."):
        got_u, got_h = proxy_env._apply_ibb_proxy(u, None)
        assert got_u == u and got_h is None


def test_missing_env_var_is_a_no_op():
    # No env vars set at all: rewriting must not happen (backward compat
    # for every VM that has not yet deployed the worker).
    for k in ("IBB_PROXY_URL", "IBB_PROXY_SECRET"):
        os.environ.pop(k, None)
    importlib.reload(detect_core)
    url, hdrs = detect_core._apply_ibb_proxy(
        "https://kamerayayin.ibb.istanbul/x.m3u8", None)
    assert url == "https://kamerayayin.ibb.istanbul/x.m3u8" and hdrs is None


def test_partial_env_is_still_a_no_op(monkeypatch):
    # URL without the secret would let anyone burn the operator's quota;
    # secret without the URL is meaningless. Either partial config = off.
    monkeypatch.setenv("IBB_PROXY_URL", "https://ibb-proxy.example.workers.dev")
    monkeypatch.delenv("IBB_PROXY_SECRET", raising=False)
    importlib.reload(detect_core)
    url, hdrs = detect_core._apply_ibb_proxy(
        "https://kamerayayin.ibb.istanbul/x.m3u8", None)
    assert url == "https://kamerayayin.ibb.istanbul/x.m3u8" and hdrs is None
