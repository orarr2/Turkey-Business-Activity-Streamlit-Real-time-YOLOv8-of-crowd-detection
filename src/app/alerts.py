"""Outbound alert push for operational events (anomalies, loitering, returns).

An alert that waits in a dashboard nobody is watching is not an alert. This
module pushes events, with the annotated snapshot when there is one, to:

  * Telegram - set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (a bot you created
    with @BotFather and the chat/channel id it should post into);
  * a generic webhook - set ALERT_WEBHOOK_URL; each event is POSTed as JSON
    {kind, cam_id, slot_id, ts, title, body, image_b64?} so it can feed
    Slack/Discord/n8n/your own service.

Design constraints, in order:
  1. NEVER break sampling - every network call is wrapped; failures log once
     per backend per cooldown and are otherwise silent.
  2. Don't storm - a per-(kind, cam) cooldown plus a global hourly cap bound
     the worst case; when the cap trips, events still land in Firestore, only
     the push is skipped.
  3. Zero new dependencies - urllib only.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
import uuid

DEFAULT_PER_KEY_COOLDOWN_S = 600     # same kind+cam at most once per 10 min
DEFAULT_GLOBAL_HOURLY_CAP  = 20      # pushes/hour across everything

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class AlertSink:
    """Fan-out pusher with rate limiting. Construct once per process."""

    def __init__(self,
                 telegram_token: str | None = None,
                 telegram_chat_id: str | None = None,
                 webhook_url: str | None = None,
                 per_key_cooldown_s: float = DEFAULT_PER_KEY_COOLDOWN_S,
                 global_hourly_cap: int = DEFAULT_GLOBAL_HOURLY_CAP,
                 timeout_s: float = 10.0):
        self.telegram_token   = telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.webhook_url      = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
        self.per_key_cooldown_s = per_key_cooldown_s
        self.global_hourly_cap  = global_hourly_cap
        self.timeout_s = timeout_s
        self._last_sent: dict[tuple[str, str], float] = {}
        self._hour_window: list[float] = []
        self._last_backend_error: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool((self.telegram_token and self.telegram_chat_id)
                    or self.webhook_url)

    def _allowed(self, kind: str, cam_id: str) -> bool:
        now = time.time()
        if now - self._last_sent.get((kind, cam_id), 0.0) < self.per_key_cooldown_s:
            return False
        self._hour_window = [t for t in self._hour_window if now - t < 3600]
        if len(self._hour_window) >= self.global_hourly_cap:
            return False
        return True

    def _mark(self, kind: str, cam_id: str) -> None:
        now = time.time()
        self._last_sent[(kind, cam_id)] = now
        self._hour_window.append(now)

    def _log_backend_error(self, backend: str, err: Exception) -> None:
        now = time.time()
        if now - self._last_backend_error.get(backend, 0.0) > 1800:
            self._last_backend_error[backend] = now
            print(f"  ! alert push via {backend} failed ({err}) - "
                  f"suppressing this log for 30 min")

    # ---- backends ----------------------------------------------------------

    def _post(self, url: str, data: bytes, content_type: str) -> None:
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            r.read()

    def _send_telegram(self, title: str, body: str,
                       image_jpeg: bytes | None) -> None:
        text = f"{title}\n{body}" if body else title
        if image_jpeg is not None:
            boundary = uuid.uuid4().hex
            parts = []
            for name, value in (("chat_id", str(self.telegram_chat_id)),
                                ("caption", text[:1024])):
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                             f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f"name=\"photo\"; filename=\"event.jpg\"\r\n"
                         f"Content-Type: image/jpeg\r\n\r\n".encode())
            payload = b"".join(parts) + image_jpeg + f"\r\n--{boundary}--\r\n".encode()
            self._post(_TELEGRAM_API.format(token=self.telegram_token,
                                            method="sendPhoto"),
                       payload, f"multipart/form-data; boundary={boundary}")
        else:
            data = json.dumps({"chat_id": self.telegram_chat_id,
                               "text": text[:4096]}).encode()
            self._post(_TELEGRAM_API.format(token=self.telegram_token,
                                            method="sendMessage"),
                       data, "application/json")

    def _send_webhook(self, event: dict, image_jpeg: bytes | None) -> None:
        payload = dict(event)
        if image_jpeg is not None:
            payload["image_b64"] = base64.b64encode(image_jpeg).decode()
        self._post(self.webhook_url, json.dumps(payload).encode(),
                   "application/json")

    # ---- public API --------------------------------------------------------

    def send(self, kind: str, cam_id: str, slot_id: str, ts: str,
             title: str, body: str = "",
             image_jpeg: bytes | None = None) -> bool:
        """Push one event. Returns True iff at least one backend accepted it.
        Rate-limited; never raises."""
        if not self.enabled or not self._allowed(kind, cam_id):
            return False
        event = {"kind": kind, "cam_id": cam_id, "slot_id": slot_id,
                 "ts": ts, "title": title, "body": body}
        sent = False
        if self.telegram_token and self.telegram_chat_id:
            try:
                self._send_telegram(title, body, image_jpeg)
                sent = True
            except Exception as e:
                self._log_backend_error("telegram", e)
        if self.webhook_url:
            try:
                self._send_webhook(event, image_jpeg)
                sent = True
            except Exception as e:
                self._log_backend_error("webhook", e)
        if sent:
            self._mark(kind, cam_id)
        return sent
