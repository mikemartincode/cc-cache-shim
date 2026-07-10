# cc-cache-shim

A small FastAPI proxy that recovers prompt-caching for Claude Code when it talks to
MiniMax models through a LiteLLM gateway.

## The problem

Claude Code always sends `/v1/messages` requests with `stream: true`. MiniMax's
Anthropic-compatible endpoint does not populate (or read) its prompt cache on streamed
requests - so an agent whose every turn re-sends a large, mostly-identical prefix pays
full input-token price on every turn, and gets full-prefix latency to match.

A second, independent cache-buster: Claude Code injects a per-request
`x-anthropic-billing-header` system block carrying a changing `cch=` nonce. Even with
caching active upstream, that block sits in the prefix and invalidates the cache every
turn. The companion LiteLLM hook in `hooks/` removes it before the request leaves the
gateway.

## How it works

The shim sits between Claude Code and the gateway
(`Claude Code -> cc-cache-shim -> LiteLLM -> MiniMax`) and does one trick on
`POST /v1/messages`:

1. It records whether the client asked for streaming, then forwards the request upstream
   with `stream: false`. The non-streamed request is the one MiniMax caches, so the
   shared prefix is written to (and read from) the prompt cache.
2. It buffers the complete JSON response, then re-emits it to the client as the full
   Anthropic Messages SSE event sequence the streaming client expects:
   - `message_start` - carries the real usage numbers from the buffered response,
     including `input_tokens`, `cache_creation_input_tokens`, and
     `cache_read_input_tokens` (so client-side cost accounting stays truthful).
   - For each content block, in order: `content_block_start`, one or more
     `content_block_delta`, `content_block_stop`. Text blocks are replayed as a single
     `text_delta`; `tool_use` blocks emit the block's full input as one
     `input_json_delta` (`partial_json` containing the serialized input object);
     `thinking` blocks emit a `thinking_delta` followed by a `signature_delta` when the
     response carries a signature.
   - `message_delta` - the real `stop_reason` / `stop_sequence` and `output_tokens`.
   - `message_stop`.

   From the client's point of view this is a normal (if bursty) streamed response.
3. Every other path (`/health` aside) is forwarded transparently, byte for byte.

One response-shape fix is applied before restreaming: MiniMax/the gateway can return
`content: null` on a reasoning-only or empty turn. The shim normalizes that to `[]` -
without it the SSE generator throws mid-stream and the client hangs waiting for events
that never arrive.

## Measured results

On real agent workloads (long Claude Code sessions with large tool-definition and
system-prompt prefixes), the shim plus the billing-block hook sustained a **~82% prompt
cache hit rate** - versus effectively 0% with streaming straight through. Cache-read
tokens are billed at a fraction of fresh input tokens, and time-to-first-token on cached
turns drops with the prefix re-read.

A related load observation baked into the retry table: MiniMax's real overload signal is
**HTTP 529** (`overloaded_error`), not 429 - measured zero 429s up to ~7k RPM.

## Quickstart

```bash
pip install -e .

# point the shim at your LiteLLM gateway (or any Anthropic-compatible endpoint)
export GATEWAY_BASE=http://localhost:4000

# run it
python -m cc_cache_shim.proxy
# or explicitly with uvicorn:
uvicorn cc_cache_shim.proxy:app --host 127.0.0.1 --port 4100
```

Then aim Claude Code at the shim instead of the gateway:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4100
```

To also stop the billing nonce from busting the cache, register the hook on the LiteLLM
gateway - copy `hooks/strip_billing_hook.py` next to your gateway config and add:

```yaml
litellm_settings:
  callbacks: ["strip_billing_hook.proxy_handler_instance"]
```

An example systemd user unit is in `examples/cc-cache-shim.service`.

## Configuration

All configuration is via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GATEWAY_BASE` | `http://localhost:4000` | Upstream base URL (LiteLLM gateway or any Anthropic-compatible endpoint). |
| `SHIM_HOST` | `127.0.0.1` | Listen address (used by `python -m cc_cache_shim.proxy`). |
| `SHIM_PORT` | `4100` | Listen port (used by `python -m cc_cache_shim.proxy`). |

Auth headers (`authorization`, `x-api-key`, `anthropic-version`, `anthropic-beta`) are
forwarded from the client as-is; the shim holds no credentials of its own.

## Operational notes

**Retry semantics.** Upstream POSTs retry on `{408, 409, 429, 500, 502, 503, 504, 529}`
and on network errors, up to 4 attempts with exponential backoff starting at 1s. A
numeric `Retry-After` header is honored when present. Other 4xx are treated as fatal
client errors and passed straight through - retrying a bad request just re-rejects.
529 is in the set because it is MiniMax's actual under-load signal (see above). If the
upstream is unreachable after all retries, the shim returns a clean Anthropic-shaped
502 error object instead of surfacing a raw exception to the agent.

**Heavy-turn passthrough.** Buffering a non-streamed response means one very long
reasoning turn can wedge against the request timeout. A request whose *last* message
contains the sentinel `<<CT_SHIM_STREAM>>` is streamed through to the upstream unchanged
(skipping the cache path for that turn only) - incremental delivery degrades to
slow-stream rather than a timeout. Scoping to the last message means only the marked
turn streams; every later turn returns to the non-stream cache path.

**Reasoning-budget markers.** The shim can clamp thinking per request via sentinels in
the system prompt (Claude Code's `--append-system-prompt` persists across every turn of
a dispatch, so a marker there configures the whole session - and because it travels
with the request, parallel sessions can carry different settings without racing on
shared state):

| Marker | Effect |
|---|---|
| `<<CT_THINK_OFF>>` | Disables thinking (`{"type": "disabled"}`). |
| `<<CT_THINK_CAP=N>>` | Enables thinking with `budget_tokens=N` (floored at the API minimum of 1024). |
| `<<CT_MAX_TOKENS=N>>` | Caps `max_tokens` at N - only ever lowers an existing value, never raises it. |

The thinking clamp only rewrites requests that arrive as `adaptive`/`enabled`; a request
the client itself sent as `disabled` is never touched. With no per-request marker, the
shim falls back to a `think_cap` file next to `proxy.py` (`off` or an integer;
convenient for a single interactive session), and with neither it passes the request
through unmodified.

**Logging.** One line per request (model, thinking mode, any clamp applied,
`max_tokens`, message count) and one per response (latency, input/cache-read/output
token counts, block types) to stdout. The response line is the fastest way to confirm
the cache is working: watch `cache_read` climb to cover most of the prefix.

## Status

This is personal infrastructure, shared as-is. It has run continuously under real agent
workloads, but it is deliberately small and specific to the failure modes described
above - expect to read the ~300 lines of `proxy.py` before trusting it with yours.
