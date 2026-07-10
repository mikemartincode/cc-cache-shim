"""LiteLLM custom callback: strip Claude Code's per-request `x-anthropic-billing-header`
system block so the prompt-cache prefix is stable (it carries a changing `cch=` nonce
that otherwise invalidates the cache every turn). Logs the first few hits to stderr
(docker logs litellm) for verification, then goes quiet.
"""
from litellm.integrations.custom_logger import CustomLogger

_PREFIX = "x-anthropic-billing-header"
_log_budget = [8]


def _note(msg: str) -> None:
    if _log_budget[0] > 0:
        _log_budget[0] -= 1
        print(f"[strip_billing_hook] {msg}", flush=True)


def _drop(blocks):
    out, removed = [], 0
    for b in blocks:
        t = b.get("text", "") if isinstance(b, dict) else (b if isinstance(b, str) else "")
        if isinstance(t, str) and t.startswith(_PREFIX):
            removed += 1
            continue
        out.append(b)
    return out, removed


def _strip_data(data: dict, where: str) -> None:
    n = 0
    s = data.get("system")
    if isinstance(s, list):
        data["system"], r = _drop(s)
        n += r
    elif isinstance(s, str):
        lines = s.split("\n")
        kept = [ln for ln in lines if not ln.startswith(_PREFIX)]
        if len(kept) != len(lines):
            data["system"] = "\n".join(kept)
            n += len(lines) - len(kept)
    msgs = data.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                c = m.get("content")
                if isinstance(c, list):
                    m["content"], r = _drop(c)
                    n += r
                elif isinstance(c, str):
                    lines = c.split("\n")
                    kept = [ln for ln in lines if not ln.startswith(_PREFIX)]
                    if len(kept) != len(lines):
                        m["content"] = "\n".join(kept)
                        n += len(lines) - len(kept)
    if n:
        _note(f"{where}: stripped {n} billing block(s)")
    else:
        _note(f"{where}: no billing block (system type={type(data.get('system')).__name__})")


class StripBillingHeader(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            _strip_data(data, f"pre_call({call_type})")
        except Exception as e:  # never break a request
            print(f"[strip_billing_hook] err {e}", flush=True)
        return data

    def get_chat_completion_prompt(self, model, messages, non_default_params,
                                   prompt_id=None, prompt_variables=None,
                                   dynamic_callback_params=None, **kwargs):
        try:
            _strip_data({"messages": messages}, "get_prompt")
        except Exception as e:
            print(f"[strip_billing_hook] gp-err {e}", flush=True)
        return model, messages, non_default_params


proxy_handler_instance = StripBillingHeader()
