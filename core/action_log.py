"""Structured user-action trace for reproducing intermittent input bugs."""

from __future__ import annotations

import atexit
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

_handle: TextIO | None = None
_lock = threading.Lock()
_sequence = 0
_MAX_LOG_BYTES = 10 * 1024 * 1024
_FLUSH_EVENTS = {
    "startup", "shutdown", "key", "mouse_press", "mouse_release",
    "pinch", "decal_placement_end", "decal_transform_end", "exception",
}


def start_action_log(path: str | Path, **details: Any) -> Path:
    """Start a fresh JSON-lines trace, replacing the previous session."""
    global _handle, _sequence
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if _handle is not None:
            _handle.close()
        _handle = destination.open("w", encoding="utf-8", buffering=65536)
        _sequence = 0
    log_action("startup", log_path=str(destination), **details)
    return destination


def log_action(event: str, **details: Any) -> None:
    """Append one compact, machine-readable event when tracing is active."""
    global _sequence
    with _lock:
        if _handle is None:
            return
        # Let the record that crosses the limit finish intact, then clear the
        # file immediately before the following write.  This keeps every line
        # valid JSON and prevents an active session from growing without bound.
        if _handle.tell() > _MAX_LOG_BYTES:
            _handle.seek(0)
            _handle.truncate(0)
            _sequence = 0
        _sequence += 1
        record = {
            "seq": _sequence,
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": str(event),
            **details,
        }
        _handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        if event in _FLUSH_EVENTS:
            _handle.flush()


def close_action_log() -> None:
    global _handle
    with _lock:
        if _handle is not None:
            _handle.flush()
            _handle.close()
            _handle = None


atexit.register(close_action_log)
