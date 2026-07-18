"""Tests for the shim — stream re-emission against a MOCK upstream.

No live gateway anywhere: the upstream is an httpx.MockTransport handler playing
the provider, and the shim app is driven in-process through ASGITransport. What's
pinned: the stream:true request is forwarded stream:false (the whole point — the
provider's cache only fires on non-stream), the response comes back as the exact
SSE event sequence Claude Code expects with every block type and the REAL usage
(cache_read included) preserved, content:null can't hang the agent, and the retry
policy treats 529 as transient and 4xx as fatal.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

import httpx

os.environ.setdefault("GATEWAY_BASE", "http://upstream.invalid")
from cc_cache_shim import proxy

UPSTREAM_BODY = {
    "id": "msg_mock1",
    "type": "message",
    "role": "assistant",
    "model": "mock-coder",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "content": [
        {"type": "thinking", "thinking": "plan the edit", "signature": "sig123"},
        {"type": "text", "text": "Editing now."},
        {"type": "tool_use", "id": "toolu_1", "name": "Edit",
         "input": {"file_path": "x.py", "old_string": "a", "new_string": "b"}},
    ],
    "usage": {"input_tokens": 21185, "output_tokens": 411,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 21106},
}


def _events(body: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    ev = None
    for line in body.splitlines():
        if line.startswith("event: "):
            ev = line[len("event: "):]
        elif line.startswith("data: "):
            out.append((ev or "", json.loads(line[len("data: "):])))
    return out


def _drive(handler, payload: dict) -> tuple[httpx.Response, list[dict]]:
    """POST one /v1/messages through the shim with the upstream mocked; return
    (response, [payloads the upstream actually received])."""
    seen: list[dict] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request)

    async def run() -> httpx.Response:
        proxy.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://shim") as c:
                return await c.post("/v1/messages", json=payload)
        finally:
            await proxy.app.state.client.aclose()

    return asyncio.run(run()), seen


def test_stream_request_goes_nonstream_upstream_and_restreams() -> None:
    res, seen = _drive(lambda _r: httpx.Response(200, json=UPSTREAM_BODY),
                       {"model": "mock-coder", "stream": True, "max_tokens": 1000,
                        "messages": [{"role": "user", "content": "hi"}]})
    # the cache-recovery move: the upstream saw stream:false
    assert len(seen) == 1 and seen[0]["stream"] is False
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = _events(res.text)
    names = [e for e, _ in events]
    assert names == [
        "message_start",
        "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "message_delta", "message_stop",
    ]
    by_name = dict(events)  # last event of each name is fine for the singletons
    # REAL usage preserved — cache_read is the number the whole shim exists for
    start_usage = by_name["message_start"]["message"]["usage"]
    assert start_usage["cache_read_input_tokens"] == 21106
    assert start_usage["input_tokens"] == 21185
    assert by_name["message_delta"]["usage"]["output_tokens"] == 411
    assert by_name["message_delta"]["delta"]["stop_reason"] == "tool_use"
    # every block type round-trips
    deltas = [d for e, d in events if e == "content_block_delta"]
    assert deltas[0]["delta"] == {"type": "thinking_delta", "thinking": "plan the edit"}
    assert deltas[1]["delta"] == {"type": "signature_delta", "signature": "sig123"}
    assert deltas[2]["delta"] == {"type": "text_delta", "text": "Editing now."}
    assert json.loads(deltas[3]["delta"]["partial_json"]) == UPSTREAM_BODY["content"][2]["input"]


def test_nonstream_request_returns_plain_json() -> None:
    res, seen = _drive(lambda _r: httpx.Response(200, json=UPSTREAM_BODY),
                       {"model": "mock-coder", "messages": [{"role": "user", "content": "hi"}]})
    assert seen[0]["stream"] is False
    assert res.status_code == 200
    assert res.json()["usage"]["cache_read_input_tokens"] == 21106


def test_content_null_normalized_no_hang() -> None:
    """A reasoning-only/empty turn can come back content:null. The shim must
    normalize it and complete the SSE — an aborted stream hangs the agent."""
    body = dict(UPSTREAM_BODY, content=None, stop_reason="end_turn")
    res, _ = _drive(lambda _r: httpx.Response(200, json=body),
                    {"model": "mock-coder", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 200
    names = [e for e, _ in _events(res.text)]
    assert names == ["message_start", "message_delta", "message_stop"]


def test_heavy_turn_marker_streams_through_upstream() -> None:
    """The wedge-avoidance path: when the LAST message carries the heavy-turn
    sentinel, the shim relays the upstream SSE as-is (stream stays true — this
    turn trades the cache for not buffering a long reasoning response)."""
    raw = b'event: ping\ndata: {"type": "ping"}\n\n'
    res, seen = _drive(lambda _r: httpx.Response(200, content=raw,
                                                 headers={"content-type": "text/event-stream"}),
                       {"model": "mock-coder", "stream": True,
                        "messages": [{"role": "user", "content": f"go {proxy.MARKER}"}]})
    assert seen[0]["stream"] is True
    assert res.content == raw


def _retry_probe(statuses: list[int], monkeypatch: pytest.MonkeyPatch) -> tuple[httpx.Response, int]:
    """Run _post_retry against an upstream that answers statuses in order
    (repeating the last). Sleep is stubbed so backoff costs nothing."""
    calls = {"n": 0}

    def upstream(_r: httpx.Request) -> httpx.Response:
        s = statuses[min(calls["n"], len(statuses) - 1)]
        calls["n"] += 1
        return httpx.Response(s, json={"status": s})

    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(proxy.asyncio, "sleep", fake_sleep)

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            return await proxy._post_retry(client, "http://upstream.invalid/v1/messages", {}, {})

    return asyncio.run(run()), calls["n"]


def test_retry_529_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    res, n = _retry_probe([529, 200], monkeypatch)
    assert (res.status_code, n) == (200, 2)


def test_fatal_4xx_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    res, n = _retry_probe([400], monkeypatch)
    assert (res.status_code, n) == (400, 1)


def test_retries_exhaust_and_return_last(monkeypatch: pytest.MonkeyPatch) -> None:
    res, n = _retry_probe([529], monkeypatch)
    assert (res.status_code, n) == (529, proxy._MAX_TRIES)
