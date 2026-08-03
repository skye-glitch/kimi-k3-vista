from __future__ import annotations

import contextlib
import csv
import datetime as dt
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import chat  # noqa: E402


class FakeResponse:
    status = 200

    def __init__(self, events: list[dict[str, object] | str]) -> None:
        self.lines: list[bytes] = []
        for event in events:
            payload = event if isinstance(event, str) else json.dumps(event)
            self.lines.extend((f"data: {payload}\n".encode(), b"\n"))

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class ChatStreamTests(unittest.TestCase):
    def test_stream_content_timing_usage_and_message_reconstruction(self) -> None:
        events: list[dict[str, object] | str] = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
            "[DONE]",
        ]
        response = FakeResponse(events)
        output = io.StringIO()
        with mock.patch.object(chat.urllib.request, "urlopen", return_value=response):
            with contextlib.redirect_stdout(output):
                result = chat.stream_chat_completion(
                    "http://example.invalid/v1/chat/completions",
                    {"stream": True},
                    10,
                )

        self.assertEqual(result.status, 200)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.assistant["reasoning_content"], "think")
        self.assertEqual(result.assistant["content"], "answer")
        self.assertEqual(result.usage["total_tokens"], 12)
        self.assertIsNotNone(result.first_token_perf)
        self.assertIsNotNone(result.first_reasoning_perf)
        self.assertIsNotNone(result.first_content_perf)
        self.assertIn("thinking> think\nkimi> answer", output.getvalue())

    def test_sse_parser_ignores_comments_and_joins_data_lines(self) -> None:
        response = iter(
            [
                b": keepalive\n",
                b"\n",
                b"data: first\n",
                b"data: second\n",
                b"\n",
            ]
        )
        self.assertEqual(list(chat.sse_payloads(response)), ["first\nsecond"])

    def test_main_writes_metrics_under_the_serving_job_id(self) -> None:
        perf = time.perf_counter() + 10
        timestamp = dt.datetime.now().astimezone()
        result = chat.StreamResult(
            status=200,
            assistant={"role": "assistant", "content": "answer"},
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "reasoning_tokens": 1,
                "total_tokens": 12,
            },
            finish_reason="stop",
            headers_perf=perf,
            headers_at=timestamp,
            first_token_perf=perf + 1,
            first_token_at=timestamp,
            first_reasoning_perf=perf + 1,
            first_reasoning_at=timestamp,
            first_content_perf=perf + 2,
            first_content_at=timestamp,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = ["chat.py", "--job-id", "123456", "--base-url", "http://test/v1"]
            with mock.patch.dict(os.environ, {"KIMI_K3_ROOT": temp_dir}):
                with mock.patch.object(sys, "argv", argv):
                    with mock.patch("builtins.input", side_effect=["hello", "/exit"]):
                        with mock.patch.object(
                            chat, "stream_chat_completion", return_value=result
                        ):
                            with contextlib.redirect_stdout(io.StringIO()):
                                self.assertEqual(chat.main(), 0)

            output = Path(temp_dir) / "logs" / "123456" / "k3-chat.csv"
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["job_id"], "123456")
            self.assertEqual(rows[0]["turn_index"], "1")
            self.assertEqual(rows[0]["prompt_tokens"], "10")


if __name__ == "__main__":
    unittest.main()
