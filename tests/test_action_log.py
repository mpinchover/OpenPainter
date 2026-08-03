from __future__ import annotations

import json

from core import action_log
from core.action_log import close_action_log, log_action, start_action_log


def test_action_log_is_fresh_structured_and_sequenced(tmp_path):
    path = tmp_path / "debug-log.txt"
    path.write_text("old session\n")

    start_action_log(path, title="OpenPainter")
    log_action("pinch", magnification=0.25, camera_radius=2.0)
    close_action_log()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["startup", "pinch"]
    assert [record["seq"] for record in records] == [1, 2]
    assert records[0]["title"] == "OpenPainter"
    assert records[1]["magnification"] == 0.25
    assert "old session" not in path.read_text()


def test_action_log_clears_before_writing_after_size_limit(tmp_path, monkeypatch):
    path = tmp_path / "debug-log.txt"
    monkeypatch.setattr(action_log, "_MAX_LOG_BYTES", 256)

    start_action_log(path, title="OpenPainter")
    log_action("large_action", payload="x" * 512)
    log_action("next_action", value=42)
    close_action_log()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {
            "seq": 1,
            "time": records[0]["time"],
            "event": "next_action",
            "value": 42,
        }
    ]
