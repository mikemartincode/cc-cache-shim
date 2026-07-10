"""CC cache shim - recovers MiniMax prompt-cache for Claude Code.

CC always streams, and MiniMax doesn't read prompt-cache on stream:true. This shim sits
between CC and the LiteLLM gateway: it forwards CC's /v1/messages with stream:false (the
gateway strips the billing nonce, injects cache_control, and MiniMax CACHES it), then
re-emits the full Anthropic response as the SSE event sequence CC expects - preserving
text + tool_use + thinking blocks and the REAL usage (incl cache_read). Any other path is
forwarded transparently. Config via env: GATEWAY_BASE (default http://localhost:4000),
SHIM_HOST (default 127.0.0.1), SHIM_PORT (default 4100).
"""
import json
import os
import pathlib
import re
import time

import asyncio

import httpx
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

GATEWAY = os.environ.get("GATEWAY_BASE", "http://localhost:4000").rstrip("/")
HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHIM_PORT", "4100"))
_HOPBYHOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
# MiniMax transient statuses to retry with backoff (its real failure mode under load is
# 529 overloaded_error, NOT 429 - measured: 0 429 up to ~7k RPM). 4xx are fatal client
# errors (bad request/auth/too-large) - pass straight through, a retry just re-rejects.
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}
_MAX_TRIES = 4


def _parse_retry_after(value: str | None) -> float | None:
    """Read a plain-seconds Retry-After like '2' or '1.5'; anything else -> None.
    (stripping at most one dot then .isdigit() = digits with one optional decimal
    point; this deliberately rejects 'inf', '1e3', and negatives.)"""
    if not value:
        return None
    if value.replace(".", "", 1).isdigit():
        return float(value)
    return None


async def _post_retry(client: httpx.AsyncClient, url: str, headers: dict, payload: dict):
    """POST with backoff on transient status / network errors; fatal 4xx returned as-is."""
    delay = 1.0
    for attempt in range(_MAX_TRIES):
        is_last_attempt = attempt == _MAX_TRIES - 1
        try:
            response = await client.post(url, headers=headers, json=payload)
        except Exception:  # network/timeout - transient
            if is_last_attempt:
                raise
            await asyncio.sleep(delay)
            delay *= 2
            continue
        if response.status_code in _RETRYABLE and not is_last_attempt:
            server_wait = _parse_retry_after(response.headers.get("retry-after"))
            await asyncio.sleep(server_wait if server_wait is not None else delay)
            delay *= 2
            continue
        return response
    raise AssertionError("unreachable: the final attempt returned or raised")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "gateway": GATEWAY}


def _fwd_headers(request: Request) -> dict:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOPBYHOP}


# Heavy-turn marker: `ct brief` emits this sentinel, so the turn whose LAST message carries it
# (M3 acting on the just-returned brief) is the reasoning-heavy one. We stream that turn instead
# of forcing non-stream - incremental delivery can't wedge past the timeout the way a single
# buffered non-stream response does. Scoped to the LAST message so only that one turn streams;
# every later turn keeps the non-stream cache path.
MARKER = "<<CT_SHIM_STREAM>>"


THINK_CAP_FILE = pathlib.Path(__file__).resolve().parent / "think_cap"
THINK_OFF_MARKER = "<<CT_THINK_OFF>>"
_THINK_CAP_RE = re.compile(r"<<CT_THINK_CAP=(\d+)>>")
# Speed lever (reasoning_tail_cap): a per-request max_tokens ceiling carried in the
# persistent system prompt, parallel-safe. Only LOWERS an existing max_tokens (never raises).
_MAXTOK_RE = re.compile(r"<<CT_MAX_TOKENS=(\d+)>>")


def _maxtok_clamp(payload: dict) -> str:
    m = _MAXTOK_RE.search(_system_text(payload))
    if not m:
        return ""
    cap = int(m.group(1))
    cur = payload.get("max_tokens")
    if isinstance(cur, int) and cur > cap:
        payload["max_tokens"] = cap
        return f"maxtok={cap}"
    return ""


def _system_text(payload: dict) -> str:
    """The request's system prompt as one string (str or list-of-blocks form). CC's
    --append-system-prompt lands here and PERSISTS across every turn - so a marker placed
    in it clamps thinking for the whole dispatch, which is what enables PARALLEL arms (each
    dispatch carries its own arm; no shared global file to race on)."""
    s = payload.get("system")
    if isinstance(s, str):
        return s
    if isinstance(s, list):
        return " ".join(b.get("text", "") for b in s if isinstance(b, dict))
    return ""


def _think_spec(payload: dict) -> str | None:
    """Resolve the thinking arm for THIS request. Per-request system marker wins (parallel-safe);
    falls back to the global ./think_cap file (single-stream convenience). Returns 'off', an int
    string, or None (passthrough)."""
    sysmsg = _system_text(payload)
    if THINK_OFF_MARKER in sysmsg:
        return "off"
    m = _THINK_CAP_RE.search(sysmsg)
    if m:
        return m.group(1)
    try:
        spec = THINK_CAP_FILE.read_text().strip()
    except OSError:
        spec = ""
    return spec or None


def _think_clamp(payload: dict) -> str:
    """Optional reasoning-budget clamp (the speed lever). Arm from _think_spec:
      'off'        -> thinking disabled (no reasoning)
      <integer N>  -> thinking enabled with budget_tokens=N (Anthropic min 1024)
      None         -> passthrough (CC's native 'adaptive')
    Only rewrites an 'adaptive'/'enabled' request; never touches CC's own 'disabled' pre-flight
    turn. Returns a short label for the log ('' = no change)."""
    spec = _think_spec(payload)
    if not spec:
        return ""
    th = payload.get("thinking")
    if not isinstance(th, dict) or th.get("type") not in ("adaptive", "enabled"):
        return ""
    if spec.lower() == "off":
        payload["thinking"] = {"type": "disabled"}
        return "off"
    try:
        n = max(1024, int(spec))
    except ValueError:
        return ""
    payload["thinking"] = {"type": "enabled", "budget_tokens": n}
    return f"cap={n}"


def _last_msg_has_marker(payload: dict) -> bool:
    msgs = payload.get("messages") or []
    if not msgs:
        return False
    content = msgs[-1].get("content")
    if isinstance(content, str):
        return MARKER in content
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            if MARKER in (b.get("text") or ""):
                return True
            v = b.get("content")  # tool_result content: str or list of blocks
            if isinstance(v, str) and MARKER in v:
                return True
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict) and MARKER in (x.get("text") or ""):
                        return True
    return False


async def _stream_passthrough(client: httpx.AsyncClient, fwd: dict, payload: dict):
    """Relay the gateway's SSE straight to CC. Incremental delivery means no single buffered
    response to time out, so a long reasoning turn degrades to slow-stream, never a wedge."""
    try:
        async with client.stream("POST", f"{GATEWAY}/v1/messages", headers=fwd, json=payload,
                                 timeout=httpx.Timeout(None, connect=10.0)) as r:
            async for chunk in r.aiter_bytes():
                yield chunk
    except Exception as exc:
        yield _sse("error", {"type": "error",
                             "error": {"type": "api_error", "message": f"shim stream: {exc}"}}).encode()


@app.post("/v1/messages")
async def shim(request: Request):
    client: httpx.AsyncClient = request.app.state.client
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"type": "error",
                            "error": {"type": "invalid_request_error", "message": "bad json"}})
    was_stream = bool(payload.get("stream", False))
    # These two MUTATE payload (clamp thinking / max_tokens); the returned
    # labels are only for the log line below.
    think_label = _think_clamp(payload)    # e.g. "off" / "cap=4096" / ""
    maxtok_label = _maxtok_clamp(payload)  # e.g. "maxtok=8000" / ""
    clamp_note = ""
    if think_label:
        clamp_note += f" <{think_label}"
    if maxtok_label:
        clamp_note += f" <{maxtok_label}"
    print(f"[shim] REQ model={payload.get('model')} thinking={payload.get('thinking')}"
          f"{clamp_note} max_tokens={payload.get('max_tokens')} stream={was_stream} "
          f"msgs={len(payload.get('messages') or [])}", flush=True)
    fwd = {"content-type": "application/json"}
    for h in ("authorization", "x-api-key", "anthropic-version", "anthropic-beta"):
        if h in request.headers:
            fwd[h] = request.headers[h]
    if was_stream and _last_msg_has_marker(payload):
        payload["stream"] = True  # heavy turn: stream through (skip cache) so it can't wedge
        print("[shim] STREAM (heavy-turn marker) - bypassing cache to avoid the non-stream wedge",
              flush=True)
        return StreamingResponse(_stream_passthrough(client, fwd, payload),
                                 media_type="text/event-stream")
    payload["stream"] = False  # force non-stream upstream -> triggers MiniMax cache
    t0 = time.time()
    try:
        r = await _post_retry(client, f"{GATEWAY}/v1/messages", fwd, payload)
    except Exception as exc:  # upstream unreachable after retries -> clean error, don't 500 the agent
        return JSONResponse(status_code=502, content={"type": "error",
                            "error": {"type": "api_error", "message": f"shim upstream: {exc}"}})
    if r.status_code != 200:
        return JSONResponse(status_code=r.status_code, content=_safe_json(r))
    resp = r.json()
    if not isinstance(resp.get("content"), list):
        # MiniMax/gateway can return content:null (a reasoning-only or empty turn). Without
        # this, `resp.get("content", [])` yields None and iterating it throws - at the log
        # line (-> 500) AND inside _restream (-> the SSE aborts mid-stream and CC HANGS).
        print(f"[shim] WARN content={resp.get('content')!r} normalised to [] "
              f"stop={resp.get('stop_reason')}", flush=True)
        resp["content"] = []
    u = resp.get("usage", {}) or {}
    print(f"[shim] {round(time.time()-t0,2)}s respModel={resp.get('model')!r} in={u.get('input_tokens')} "
          f"cache_read={u.get('cache_read_input_tokens')} out={u.get('output_tokens')} "
          f"blocks={[b.get('type') for b in resp.get('content', [])]}", flush=True)
    if not was_stream:
        return JSONResponse(content=resp)
    return StreamingResponse(_restream(resp), media_type="text/event-stream")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(path: str, request: Request):
    """Transparent forward for everything that isn't the streaming /v1/messages shim."""
    client: httpx.AsyncClient = request.app.state.client
    body = await request.body()
    try:
        r = await client.request(request.method, f"{GATEWAY}/{path}",
                                 headers=_fwd_headers(request), content=body,
                                 params=request.query_params)
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


def _safe_json(r):
    try:
        return r.json()
    except Exception:
        return {"type": "error", "error": {"type": "api_error", "message": r.text[:500]}}


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _restream(resp: dict):
    usage = resp.get("usage", {}) or {}
    meta = {"id": resp.get("id", f"msg_{int(time.time())}"), "type": "message", "role": "assistant",
            "content": [], "model": resp.get("model", "minimax-m3"),
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0,
                      "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                      "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0)}}
    yield _sse("message_start", {"type": "message_start", "message": meta})
    for i, b in enumerate(resp.get("content", [])):
        bt = b.get("type")
        if bt == "text":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                       "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                       "delta": {"type": "text_delta", "text": b.get("text", "")}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
        elif bt == "thinking":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                       "content_block": {"type": "thinking", "thinking": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                       "delta": {"type": "thinking_delta", "thinking": b.get("thinking", "")}})
            if b.get("signature"):
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                           "delta": {"type": "signature_delta", "signature": b["signature"]}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
        elif bt == "tool_use":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                       "content_block": {"type": "tool_use", "id": b.get("id"),
                                         "name": b.get("name"), "input": {}}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                       "delta": {"type": "input_json_delta", "partial_json": json.dumps(b.get("input", {}))}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    yield _sse("message_delta", {"type": "message_delta",
               "delta": {"stop_reason": resp.get("stop_reason", "end_turn"),
                         "stop_sequence": resp.get("stop_sequence")},
               "usage": {"output_tokens": usage.get("output_tokens", 0)}})
    yield _sse("message_stop", {"type": "message_stop"})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
