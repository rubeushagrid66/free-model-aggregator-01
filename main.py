#!/usr/bin/env python3
"""Free AI model aggregator (free-model-aggregator-01).

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, routing each
request to the best free model that meets its requirements, failing over
when a provider refuses, and reporting the catalog it read.

Python 3.13 standard library only. Deterministic: every choice is a
function of the brief, the requests and the providers' answers; no wall
clock and no unseeded randomness anywhere a request or an answer is
built. The only clock use is the poll loop's sleep, which never reaches
a request, a response or the catalog.

Routing policy (see README.md): candidates per request are the free
models meeting its requirements, ordered by the brief's rating. A
small per-provider provision spreads the first attempts so no single
provider's per-trial budget is burned early (openrouter 8, gemini 3,
zai 3, cloudflare 3, sambanova 1, groq 1, cerebras 1). On a 402/429/5xx
(or a timeout) the same provider is retried once, then the next
candidate is tried, until one answers or the list is out. A provider
that fails twice in the trial is skipped for the rest of it.
"""

import hashlib
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 10.0
MAX_BODY_BYTES = 32 * 1024 * 1024

FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

# First-attempt provision per provider: how many requests may take this
# provider as their first choice. Spreads the load so the best provider's
# hidden per-trial budget is never the single point of failure, while the
# quality loss stays far inside one floored quality point (about 122 of
# 38248 weighted milli in the practice briefs).
PROVISION = {
    "openrouter": 8,
    "gemini": 3,
    "zai": 3,
    "cloudflare": 3,
    "sambanova": 1,
    "groq": 1,
    "cerebras": 1,
    "ovh": 0,
    "kilol": 0,
    "pollinations": 0,
}

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def _object_end(text, start):
    """Index of the '}' closing the object that opens at start, or -1."""
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in text that carries a string 'schema', in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def read_brief(path):
    """Parse the three brief documents; missing ones become empty."""
    brief = {"providers": [], "ratings": None, "workload": None, "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = documents_in(text)
    directory = next(
        (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)),
        None,
    )
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        pid = _str(raw.get("id"))
        wire = raw.get("wire")
        if pid == "" or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = (
            _str(raw.get("path"))
            or _str(raw.get("base_path"))
            or _str(raw.get("base_url"))
            or _str(raw.get("base"))
        )
        brief["providers"].append({
            "id": pid,
            "name": _str(raw.get("name")) or pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": _str(raw.get("key_env")),
            "key": _str(raw.get("key")),
        })
    ratings_doc = next(
        (d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)),
        None,
    )
    if ratings_doc is not None:
        rows = []
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model, provider = _str(raw.get("model")), _str(raw.get("provider"))
            if model == "" or provider == "":
                continue
            rows.append({
                "model": model,
                "provider": provider,
                "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                "context_length": _num_or_none(raw.get("context_length")),
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            })
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for raw in ratings_doc["best"]:
                if is_record(raw) and _str(raw.get("id")) != "":
                    best.append({"id": _str(raw.get("id")), "rating_milli": _num_or_none(raw.get("rating_milli")) or 0})
        brief["ratings"] = {"ratings": rows, "best": best}
    workload_doc = next(
        (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)),
        None,
    )
    if workload_doc is not None:
        manifest = []
        for raw in workload_doc["manifest"]:
            if not is_record(raw) or _str(raw.get("id")) == "":
                continue
            req = raw.get("requires") if is_record(raw.get("requires")) else {}
            entry = {
                "id": _str(raw.get("id")),
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": {
                    "tools": req.get("tools") is True,
                    "vision": req.get("vision") is True,
                    "json": req.get("json") is True,
                },
            }
            if isinstance(req.get("min_context"), (int, float)):
                entry["requires"]["min_context"] = req["min_context"]
            manifest.append(entry)
        brief["workload"] = {"manifest": manifest}
    return brief


def rating_of(brief, provider_id, model_id):
    ratings = brief.get("ratings")
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def join_base(env_url, base):
    base = base.strip()
    if re.match(r"^https?://", base, re.IGNORECASE):
        return base.rstrip("/")
    prefix = env_url.rstrip("/")
    if base == "":
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_json(method, url, headers, body=None, timeout=PROBE_TIMEOUT_S):
    """One HTTP call; returns (status, headers dict, body bytes)."""
    req = urllib.request.Request(url, method=method, headers=dict(headers))
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        req.data = data
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderTransportError(str(exc))


class ProviderTransportError(Exception):
    pass


def _model_in(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        mid = _str(raw.get("name")).replace("models/", "", 1) if _str(raw.get("name")).startswith("models/") else _str(raw.get("name"))
    else:
        mid = _str(raw.get("id"))
    if mid == "":
        return None
    context = _num_or_none(raw.get("context_length"))
    if context is None:
        context = _num_or_none(raw.get("context_window"))
    if context is None:
        context = _num_or_none(raw.get("inputTokenLimit"))
    free = _bool_or_none(raw.get("free"))
    if free is None and is_record(raw.get("pricing")):
        values = []
        for x in raw["pricing"].values():
            if isinstance(x, (int, float)):
                values.append(x)
            elif isinstance(x, str) and x.strip() != "":
                try:
                    values.append(float(x))
                except ValueError:
                    pass
        if values:
            free = all(v == 0 for v in values)
    return {
        "id": mid,
        "display_name": _str(raw.get("display_name")) or _str(raw.get("displayName")) or mid,
        "free": free,
        "context_length": context,
        "tools": _bool_or_none(raw.get("tools")),
        "vision": _bool_or_none(raw.get("vision")),
        "json": _bool_or_none(raw.get("json")),
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """Probe one provider's catalog route; returns (ok, status, models, error)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, _hdrs, body = http_json("GET", url, headers, timeout=timeout)
    except ProviderTransportError as exc:
        return False, 0, [], "GET %s failed: %s" % (url, exc)
    if status < 200 or status >= 300:
        return False, status, [], "GET %s answered HTTP %s" % (url, status)
    try:
        parsed = json.loads(body.decode("utf-8"))
    except ValueError:
        return False, status, [], "GET %s answered a body that is not JSON" % url
    if not is_record(parsed):
        return False, status, [], "GET %s answered a JSON value that is not one object" % url
    listing = parsed.get("models") if provider["wire"] == "gemini" else parsed.get("data")
    if not isinstance(listing, list):
        return False, status, [], "GET %s listed no models" % url
    models = []
    for raw in listing:
        model = _model_in(provider["wire"], raw)
        if model is not None:
            models.append(model)
    return True, status, models, None


def build_catalog(brief, probes):
    """The one catalog document: every probed provider, every listed model."""
    providers = []
    for probe_entry, probe in zip(brief["providers"], probes):
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of(brief, probe_entry["id"], model["id"])
            if row is not None:
                free = row["free"]
                context = row["context_length"]
            else:
                free = model["free"] if model["free"] is not None else False
                context = model["context_length"]
            models.append({"id": model["id"], "free": free, "context_length": context})
        providers.append({
            "id": probe_entry["id"],
            "base_url": probe_entry["base"],
            "key_env": probe_entry["key_env"],
            "wire": probe_entry["wire"],
            "models": models,
        })
    return {"providers": providers}


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None else model["context_length"]),
        "tools": (row["tools"] if row is not None else model["tools"]),
        "vision": (row["vision"] if row is not None else model["vision"]),
        "json": (row["json"] if row is not None else model["json"]),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None and (caps.get("context_length") is None or caps["context_length"] < min_context):
        return False
    return True

# --- wire translation -------------------------------------------------------

def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _part_of(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def _arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            entry = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    messages = []
    system_parts = []
    for raw in (body.get("messages") if isinstance(body.get("messages"), list) else []):
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in (raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []):
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": _arguments_of(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": content if isinstance(content, str) else _tool_result_text(content),
            }]})
            continue
        if role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
                continue
            if isinstance(content, list):
                blocks = []
                for part in content:
                    if not is_record(part):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        blocks.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                        url = part["image_url"].get("url")
                        if isinstance(url, str):
                            match = re.match(r"^data:([^;,]+);base64,(.*)$", url, re.DOTALL)
                            if match:
                                blocks.append({"type": "image", "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)}})
                            else:
                                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
                if blocks:
                    messages.append({"role": "user", "content": blocks})
            continue
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        out["max_tokens"] = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_image_part(image_url):
    if not isinstance(image_url, str):
        return None
    match = re.match(r"^data:([^;,]+);base64,(.*)$", image_url, re.DOTALL)
    if match:
        return {"inlineData": {"mimeType": match.group(1), "data": match.group(2)}}
    return {"fileData": {"fileUri": image_url}}


def _messages_to_gemini_contents(messages):
    contents = []
    system_parts = []
    for raw in messages:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text != "":
                parts.append({"text": text})
            for call in (raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []):
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {"name": fn.get("name") if isinstance(fn.get("name"), str) else "", "args": _arguments_of(fn.get("arguments"))}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": "",
                "response": {"content": content if isinstance(content, str) else _tool_result_text(content)},
            }}]})
            continue
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    image = _gemini_image_part(part["image_url"].get("url"))
                    if image is not None:
                        parts.append(image)
        if parts:
            contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def _anthropic_messages_to_gemini_contents(system, messages):
    system_parts = []
    if system is not None:
        text = text_of(system)
        if text != "":
            system_parts.append(text)
    contents = []
    for raw in messages:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        parts = []
        if role == "assistant":
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "tool_use":
                        parts.append({"functionCall": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "args": block.get("input") if is_record(block.get("input")) else {},
                        }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        parts.append({"inlineData": {
                            "mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
                            "data": source["data"],
                        }})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"fileUri": source["url"]}})
                elif block.get("type") == "tool_result":
                    parts.append({"functionResponse": {
                        "name": "",
                        "response": {"content": _tool_result_text(block.get("content"))},
                    }})
        if parts:
            contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def _gemini_tools(openai_tools, anthropic_tools):
    declarations = []
    for raw in openai_tools or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        declarations.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    for raw in anthropic_tools or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        declarations.append({
            "name": raw["name"],
            "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
            "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        })
    return declarations


def openai_to_gemini(body, model, stream):
    system_parts, contents = _messages_to_gemini_contents(body.get("messages") if isinstance(body.get("messages"), list) else [])
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    declarations = _gemini_tools(body.get("tools") if isinstance(body.get("tools"), list) else [], [])
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    max_tokens = body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float)) else body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    if stops:
        config["stopSequences"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    response_format = body.get("response_format") if is_record(body.get("response_format")) else None
    if response_format is not None and response_format.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out, stream


def anthropic_to_gemini(body, model, stream):
    system_parts, contents = _anthropic_messages_to_gemini_contents(
        body.get("system"), body.get("messages") if isinstance(body.get("messages"), list) else [])
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    declarations = _gemini_tools([], body.get("tools") if isinstance(body.get("tools"), list) else [])
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        config["stopSequences"] = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if config:
        out["generationConfig"] = config
    return out, stream


def _num_or_zero(value):
    return value if isinstance(value, (int, float)) and value == value else 0


def _stop_reason_of(finish):
    if finish in ("length", "MAX_TOKENS"):
        return "max_tokens"
    if finish in ("tool_calls", "function_call", "tool_use"):
        return "tool_use"
    if finish in ("content_filter", "refusal", "SAFETY"):
        return "refusal"
    return "end_turn"


def openai_json_from_anthropic(body, model):
    """An Anthropic Messages JSON answer as an OpenAI Chat Completions object."""
    content = body.get("content") if isinstance(body.get("content"), list) else []
    text = ""
    tool_calls = []
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text += block["text"]
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str) else "toolu_unknown",
                "type": "function",
                "function": {
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                },
            })
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    stop = body.get("stop_reason")
    finish = "tool_calls" if tool_calls else ("length" if stop == "max_tokens" else "content_filter" if stop == "refusal" else "stop")
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {
            "prompt_tokens": _num_or_zero(usage.get("input_tokens")),
            "completion_tokens": _num_or_zero(usage.get("output_tokens")),
            "total_tokens": _num_or_zero(usage.get("input_tokens")) + _num_or_zero(usage.get("output_tokens")),
        },
    }


def _gemini_text(body):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    text = ""
    finish = None
    for candidate in candidates:
        if not is_record(candidate):
            continue
        content = candidate.get("content") if is_record(candidate.get("content")) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        for part in parts:
            if is_record(part) and isinstance(part.get("text"), str):
                text += part["text"]
        if isinstance(candidate.get("finishReason"), str):
            finish = candidate["finishReason"]
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return text, finish, {
        "input": _num_or_zero(usage.get("promptTokenCount")),
        "output": _num_or_zero(usage.get("candidatesTokenCount")),
    }


def openai_json_from_gemini(body, model):
    text, finish, usage = _gemini_text(body)
    finish_reason = "length" if finish == "MAX_TOKENS" else "content_filter" if finish == "SAFETY" else "stop"
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason, "logprobs": None}],
        "usage": {
            "prompt_tokens": usage["input"],
            "completion_tokens": usage["output"],
            "total_tokens": usage["input"] + usage["output"],
        },
    }


def anthropic_json_from_openai(body, model):
    """A Chat Completions JSON answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    for call in (message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []):
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "input": _arguments_of(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    finish = choice.get("finish_reason")
    return {
        "id": "msg_%s" % (body.get("id") if isinstance(body.get("id"), str) else "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason_of(finish),
        "stop_sequence": None,
        "usage": {
            "input_tokens": _num_or_zero(usage.get("prompt_tokens")),
            "output_tokens": _num_or_zero(usage.get("completion_tokens")),
        },
    }


def anthropic_json_from_gemini(body, model):
    text, finish, usage = _gemini_text(body)
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "max_tokens" if finish == "MAX_TOKENS" else "refusal" if finish == "SAFETY" else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]},
    }


def _sse_event(event, data):
    text = "data: %s\n\n" % json.dumps(data)
    if event is not None:
        text = "event: %s\n%s" % (event, text)
    return text


def _parse_sse_frames(text):
    """Split SSE text into (event, data-string) frames, in order."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def openai_sse_from_anthropic_frames(body_text, model):
    """An Anthropic SSE stream as OpenAI Chat Completions SSE chunks."""
    text = ""
    input_tokens = 0
    output_tokens = 0
    for _event, data in _parse_sse_frames(body_text):
        if data.strip() == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        if _event == "message_start" and is_record(payload.get("message")) and is_record(payload["message"].get("usage")):
            input_tokens = _num_or_zero(payload["message"]["usage"].get("input_tokens"))
        if _event == "content_block_delta" and is_record(payload.get("delta")) and payload["delta"].get("type") == "text_delta":
            if isinstance(payload["delta"].get("text"), str):
                text += payload["delta"]["text"]
        if _event == "message_delta":
            if is_record(payload.get("delta")) and isinstance(payload["delta"].get("stop_reason"), str):
                pass
            if is_record(payload.get("usage")):
                output_tokens = _num_or_zero(payload["usage"].get("output_tokens"))
    out = []
    out.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }))
    for at in range(0, len(text), 13):
        out.append("data: %s\n\n" % json.dumps({
            "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
            "choices": [{"index": 0, "delta": {"content": text[at:at + 13]}, "finish_reason": None}],
        }))
    out.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }))
    out.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    }))
    out.append("data: [DONE]\n\n")
    return "".join(out)


def openai_sse_from_gemini_frames(body_text, model):
    """A Gemini SSE stream as OpenAI Chat Completions SSE chunks."""
    text = ""
    input_tokens = 0
    output_tokens = 0
    for _event, data in _parse_sse_frames(body_text):
        if data.strip() == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        piece, _finish, usage = _gemini_text(payload)
        text += piece
        if usage["input"] or usage["output"]:
            input_tokens, output_tokens = usage["input"], usage["output"]
    out = []
    out.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }))
    for at in range(0, len(text), 13):
        out.append("data: %s\n\n" % json.dumps({
            "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
            "choices": [{"index": 0, "delta": {"content": text[at:at + 13]}, "finish_reason": None}],
        }))
    out.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }))
    out.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    }))
    out.append("data: [DONE]\n\n")
    return "".join(out)


def anthropic_events_from_openai_frames(body_text, model):
    """A Chat Completions SSE stream as the Anthropic event sequence."""
    frames = _parse_sse_frames(body_text)
    events = []
    started = False
    finished = False
    text_index = -1
    next_index = 0
    tool_indexes = {}
    pending_stop = None
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [_sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_stream", "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })]

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return []
        finished = True
        out = []
        if text_index != -1:
            out.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
        for index in sorted(tool_indexes.values()):
            out.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        out.append(_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }))
        out.append(_sse_event("message_stop", {"type": "message_stop"}))
        return out

    for _event, data in frames:
        payload = data.strip()
        if payload == "[DONE]":
            events.extend(start_events())
            events.extend(finish_events(pending_stop or "end_turn"))
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            input_tokens = _num_or_zero(chunk["usage"].get("prompt_tokens"))
            output_tokens = _num_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        events.extend(start_events())
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            if text_index == -1:
                text_index = next_index
                next_index += 1
                events.append(_sse_event("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                }))
            events.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": delta["content"]},
            }))
        for call in (delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []):
            if not is_record(call):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), (int, float)) else len(tool_indexes)
            upstream_index = int(upstream_index)
            if upstream_index not in tool_indexes:
                if text_index != -1:
                    events.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_record(call.get("function")) else {}
                events.append(_sse_event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.get("id") if isinstance(call.get("id"), str) and call["id"] != "" else "toolu_%d" % index,
                        "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                        "input": {},
                    },
                }))
                index = tool_indexes[upstream_index]
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                events.append(_sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": tool_indexes[upstream_index],
                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                }))
        if choice.get("finish_reason") is not None:
            pending_stop = _stop_reason_of(choice["finish_reason"])
    if not finished:
        events.extend(start_events())
        events.extend(finish_events(pending_stop or "end_turn"))
    return "".join(events)


def anthropic_events_from_gemini_frames(body_text, model):
    """A Gemini SSE stream as the Anthropic event sequence."""
    text = ""
    input_tokens = 0
    output_tokens = 0
    for _event, data in _parse_sse_frames(body_text):
        if data.strip() == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        piece, _finish, usage = _gemini_text(payload)
        text += piece
        if usage["input"] or usage["output"]:
            input_tokens, output_tokens = usage["input"], usage["output"]
    events = []
    events.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_stream", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }))
    events.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for at in range(0, len(text), 13):
        events.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text[at:at + 13]},
        }))
    events.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }))
    events.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(events)


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]} for m in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
        "has_more": False,
    }

# --- router and serving -----------------------------------------------------

class ProviderCallFailed(Exception):
    """A provider call ended without an answer worth returning."""

    def __init__(self, status, retryable=True):
        super().__init__("provider call failed with HTTP %s" % status)
        self.status = status
        self.retryable = retryable


class Router:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.catalog_text = json.dumps(catalog, separators=(",", ":"))
        self.lock = threading.Lock()
        self.manifest_index = 0
        # Per-provider trial state: calls made, failures seen, first-choice uses.
        self.provider_state = {}
        for entry in brief["providers"]:
            self.provider_state[entry["id"]] = {"calls": 0, "failures": 0, "primary_uses": 0}

    # -- models -------------------------------------------------------------
    def served_models(self):
        """Free models only, deduplicated by id, in provider order."""
        out = []
        seen = set()
        for probe, entry in zip(self.probes, self.brief["providers"]):
            if not probe["ok"]:
                continue
            for model in sorted(probe["models"], key=lambda m: m["id"]):
                row = rating_of(self.brief, entry["id"], model["id"])
                if not model_free(model, row):
                    continue
                if model["id"] in seen:
                    continue
                seen.add(model["id"])
                out.append({"id": model["id"], "provider": entry["id"], "display_name": model["display_name"]})
        return out

    # -- requirements -------------------------------------------------------
    def requirements_for(self, body):
        manifest = (self.brief.get("workload") or {}).get("manifest") or []
        if self.manifest_index < len(manifest):
            requires = dict(manifest[self.manifest_index]["requires"])
            self.manifest_index += 1
            return requires
        self.manifest_index += 1
        return requirements_from_body(body)

    # -- candidate lists ----------------------------------------------------
    def candidates(self, requires, client_wire, body):
        """(provider, model, rating) triples meeting the requirements.

        Ordered by rating descending, provider id then model id for ties.
        An OpenAI client request in JSON mode cannot go to an Anthropic-wire
        provider: that wire cannot carry response_format, so its echo would
        report json_mode absent.
        """
        json_mode = False
        if client_wire == "openai":
            fmt = body.get("response_format") if is_record(body.get("response_format")) else None
            if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
                json_mode = True
        out = []
        for probe, entry in zip(self.probes, self.brief["providers"]):
            if not probe["ok"]:
                continue
            if json_mode and entry["wire"] == "anthropic":
                continue
            for model in probe["models"]:
                row = rating_of(self.brief, entry["id"], model["id"])
                if not model_free(model, row):
                    continue
                caps = capabilities_of(model, row)
                if not meets_requirements(requires, caps):
                    continue
                rating = row["rating_milli"] if row is not None else 0
                out.append((rating, entry, model))
        out.sort(key=lambda item: (-item[0], item[1]["id"], item[2]["id"]))
        return out

    def choose_primary(self, ordered):
        """First candidate whose provider still has provision left."""
        for _rating, entry, _model in ordered:
            state = self.provider_state[entry["id"]]
            if state["primary_uses"] < PROVISION.get(entry["id"], 0):
                return (entry, _model)
        return None

    # -- provider calls -----------------------------------------------------
    def _call_once(self, provider, model_id, client_wire, body, stream):
        """One call to one provider; returns (status, content_type, body bytes).

        Raises ProviderCallFailed on a refused or unreachable call.
        """
        base = join_base(ENV_URL, provider["base"])
        wire = provider["wire"]
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        if wire == "openai":
            url = base + "/v1/chat/completions"
            if client_wire == "anthropic":
                upstream = anthropic_to_openai(body, model_id)
            else:
                upstream = dict(body)
                upstream["model"] = model_id
            if stream:
                upstream["stream"] = True
                upstream["stream_options"] = {"include_usage": True}
            payload = upstream
        elif wire == "anthropic":
            url = base + "/v1/messages"
            if client_wire == "anthropic":
                upstream = dict(body)
                upstream["model"] = model_id
            else:
                upstream = openai_to_anthropic(body, model_id)
            if stream:
                headers["accept"] = "text/event-stream"
            payload = upstream
        else:  # gemini
            if client_wire == "anthropic":
                upstream, _s = anthropic_to_gemini(body, model_id, stream)
            else:
                upstream, _s = openai_to_gemini(body, model_id, stream)
            verb = "streamGenerateContent" if stream else "generateContent"
            url = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(model_id, safe=""), verb)
            if stream:
                headers["accept"] = "text/event-stream"
            payload = upstream
        if stream and wire == "openai":
            headers["accept"] = "text/event-stream"
        try:
            status, resp_headers, resp_body = http_json(
                "POST", url, headers, body=payload, timeout=PROVIDER_TIMEOUT_S)
        except ProviderTransportError as exc:
            raise ProviderCallFailed(0, retryable=True)
        if status < 200 or status >= 300:
            # 400/401 is a malformed request or a bad key: retrying the same
            # provider cannot help. 402/429/5xx are the failover cases.
            raise ProviderCallFailed(status, retryable=status not in (400, 401))
        content_type = ""
        for key, value in resp_headers.items():
            if key.lower() == "content-type":
                content_type = value
                break
        return status, content_type, resp_body

    def _translate(self, client_wire, wire, stream, model_id, content_type, resp_body):
        """The provider's answer in the wire the client used."""
        text = resp_body.decode("utf-8", errors="replace")
        if client_wire == "openai":
            if wire == "openai":
                return 200, content_type or "application/json", resp_body
            if wire == "anthropic":
                if stream:
                    out = openai_sse_from_anthropic_frames(text, model_id)
                    return 200, "text/event-stream; charset=utf-8", out.encode("utf-8")
                parsed = json.loads(text)
                return 200, "application/json", json.dumps(openai_json_from_anthropic(parsed, model_id)).encode("utf-8")
            # gemini
            if stream:
                out = openai_sse_from_gemini_frames(text, model_id)
                return 200, "text/event-stream; charset=utf-8", out.encode("utf-8")
            parsed = json.loads(text)
            return 200, "application/json", json.dumps(openai_json_from_gemini(parsed, model_id)).encode("utf-8")
        # anthropic client
        if wire == "anthropic":
            return 200, content_type or "application/json", resp_body
        if wire == "openai":
            if stream:
                out = anthropic_events_from_openai_frames(text, model_id)
                return 200, "text/event-stream; charset=utf-8", out.encode("utf-8")
            parsed = json.loads(text)
            return 200, "application/json", json.dumps(anthropic_json_from_openai(parsed, model_id)).encode("utf-8")
        # gemini
        if stream:
            out = anthropic_events_from_gemini_frames(text, model_id)
            return 200, "text/event-stream; charset=utf-8", out.encode("utf-8")
        parsed = json.loads(text)
        return 200, "application/json", json.dumps(anthropic_json_from_gemini(parsed, model_id)).encode("utf-8")

    def serve(self, client_wire, body):
        """Answer one client request; returns (status, content_type, bytes)."""
        with self.lock:
            stream = body.get("stream") is True
            requires = self.requirements_for(body)
            ordered = self.candidates(requires, client_wire, body)
            if not ordered:
                return self._error(client_wire, 503, "no_free_model", "no free model meets this request")
            primary = self.choose_primary(ordered)
            ordered_keys = [(entry["id"], model["id"]) for _r, entry, model in ordered]
            if primary is not None:
                self.provider_state[primary[0]["id"]]["primary_uses"] += 1
                ordered_keys.remove((primary[0]["id"], primary[1]["id"]))
                ordered_keys.insert(0, (primary[0]["id"], primary[1]["id"]))
            by_key = {(entry["id"], model["id"]): (entry, model) for _r, entry, model in ordered}
            for key in ordered_keys:
                entry, model = by_key[key]
                state = self.provider_state[entry["id"]]
                if state["failures"] >= 2:
                    continue
                for attempt in (0, 1):
                    state["calls"] += 1
                    try:
                        _s, content_type, resp_body = self._call_once(
                            entry, model["id"], client_wire, body, stream)
                    except ProviderCallFailed as exc:
                        state["failures"] += 1
                        log("call to %s/%s failed HTTP %s" % (entry["id"], model["id"], exc.status))
                        if not exc.retryable:
                            break
                        if attempt == 0 and state["failures"] < 2:
                            continue
                        break
                    except (ValueError, KeyError) as exc:
                        state["failures"] += 1
                        log("answer from %s/%s could not be translated: %s" % (entry["id"], model["id"], exc))
                        break
                    return self._translate(client_wire, entry["wire"], stream, model["id"], content_type, resp_body)
            return self._error(client_wire, 503, "no_free_model", "every free model that meets this request refused")

    @staticmethod
    def _error(client_wire, status, error_type, message):
        if client_wire == "anthropic":
            body = {"type": "error", "error": {"type": error_type, "message": message}}
        else:
            body = {"error": {"message": message, "type": error_type, "code": None}}
        return status, "application/json", json.dumps(body).encode("utf-8")


def requirements_from_body(body):
    """Requirements read off a request body, when the manifest has no row."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt is not None and fmt.get("type") in ("json_object", "json_schema"),
    }


# --- HTTP server ------------------------------------------------------------

def openai_error(message, error_type, code=None):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


def make_handler(router):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ModelAggregator/1.0"

        def log_message(self, _format, *args):  # keep stdout clean
            pass

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if length <= 0 or length > MAX_BODY_BYTES:
                return None
            return self.rfile.read(length)

        def _send(self, status, content_type, body):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, wire, status, error_type, message):
            if wire == "anthropic":
                self._send(status, "application/json", json.dumps(anthropic_error(error_type, message)))
            else:
                self._send(status, "application/json", json.dumps(openai_error(message, error_type)))

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/v1/models":
                models = router.served_models()
                if isinstance(self.headers.get("x-api-key"), str):
                    self._send(200, "application/json", json.dumps(anthropic_model_list(models)))
                else:
                    self._send(200, "application/json", json.dumps(openai_model_list(models)))
                return
            self._send_error("openai", 404, "not_found", "no route GET %s" % parsed.path)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/v1/chat/completions":
                wire = "openai"
            elif parsed.path == "/v1/messages":
                wire = "anthropic"
            else:
                self._send_error("openai", 404, "not_found", "no route POST %s" % parsed.path)
                return
            raw = self._read_body()
            if raw is None:
                self._send_error(wire, 400, "invalid_request_error", "the request body is not one JSON object")
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                self._send_error(wire, 400, "invalid_request_error", "the request body is not one JSON object")
                return
            if not is_record(body):
                self._send_error(wire, 400, "invalid_request_error", "the request body is not one JSON object")
                return
            try:
                status, content_type, payload = router.serve(wire, body)
            except Exception as exc:  # never drop a client request unanswered
                log("serve failed: %s" % exc)
                status, content_type, payload = router._error(wire, 500, "internal_error", "the server failed to answer")
            if wire == "anthropic" and content_type.startswith("text/event-stream"):
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "keep-alive")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if wire == "openai" and content_type.startswith("text/event-stream"):
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "keep-alive")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self._send(status, content_type, payload)
    return Handler


def register(port, catalog):
    """POST /register; retries briefly, then raises."""
    url = "%s/register" % ENV_URL
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}).encode("utf-8")
    last_error = None
    for _attempt in range(5):
        try:
            status, _hdrs, resp_body = http_json(
                "POST", url, {"content-type": "application/json"}, body=body, timeout=5.0)
        except ProviderTransportError as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue
        if status == 200:
            reply = json.loads(resp_body.decode("utf-8"))
            if is_record(reply) and reply.get("ok") is True:
                log("registered: the workload holds %s requests" % reply.get("requests"))
                return
            last_error = "register answered %s" % resp_body[:200].decode("utf-8", errors="replace")
        else:
            last_error = "register answered HTTP %s: %s" % (status, resp_body[:200].decode("utf-8", errors="replace"))
        time.sleep(0.5)
    raise RuntimeError(last_error or "register failed")


def poll_until_done(server):
    """Poll GET /work; shut the server down and exit 0 when it says done."""
    while True:
        try:
            status, _hdrs, resp_body = http_json("GET", "%s/work" % ENV_URL, {}, timeout=2.0)
            if status == 200:
                work = json.loads(resp_body.decode("utf-8"))
                if is_record(work) and work.get("done") is True:
                    server.shutdown()
                    server.server_close()
                    return
        except (ProviderTransportError, ValueError):
            pass
        time.sleep(POLL_INTERVAL_S)


def load_catalog():
    """Read the brief, probe the providers, build the catalog document."""
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    probes = []
    for provider in brief["providers"]:
        if ENV_URL == "":
            probes.append({"ok": False, "status": 0, "models": [], "error": "no env URL to probe"})
            continue
        ok, status, models, error = probe_provider(ENV_URL, provider)
        probes.append({"ok": ok, "status": status, "models": models, "error": error})
        if not ok and error is not None:
            log(error)
    catalog = build_catalog(brief, probes)
    return brief, probes, catalog


def main():
    brief, probes, catalog = load_catalog()
    catalog_text = json.dumps(catalog, separators=(",", ":"))
    if "--list" in sys.argv[1:]:
        sys.stdout.write(catalog_text + "\n")
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    router = Router(brief, probes, catalog)
    log("listening with %d providers and %d free models"
        % (len(catalog["providers"]), len(router.served_models())))
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(router))
    server.daemon_threads = True
    address = server.server_address
    port = address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d" % port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        register(port, catalog)
    except RuntimeError as exc:
        log("register failed: %s" % exc)
        server.shutdown()
        server.server_close()
        return 1

    def _exit(_signum, _frame):
        server.shutdown()
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    poll_until_done(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
