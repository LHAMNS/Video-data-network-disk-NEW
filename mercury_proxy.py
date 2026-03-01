#!/usr/bin/env python3
"""
Mercury 2 API Proxy for Claude Code
Translates Anthropic Messages API format <-> OpenAI Chat Completions format
so Claude Code can use Mercury 2 (Inception Labs) as its backend.

Usage:
    export INCEPTION_API_KEY="sk_..."
    python3 mercury_proxy.py [--port 8082]

Then in another terminal:
    export ANTHROPIC_BASE_URL="http://127.0.0.1:8082"
    export ANTHROPIC_API_KEY="dummy"
    claude
"""

import json
import sys
import os
import argparse
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MERCURY_BASE = "https://api.inceptionlabs.ai/v1"
MERCURY_MODEL = "mercury-2"
REASONING_EFFORT = "high"  # instant | low | medium | high
DEFAULT_MAX_TOKENS = 16384

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mercury-proxy")


# ---------------------------------------------------------------------------
# Format translators
# ---------------------------------------------------------------------------

def anthropic_to_openai(body: dict) -> dict:
    """Convert Anthropic Messages API request body to OpenAI Chat Completions."""
    messages = []

    # System prompt
    system_text = body.get("system")
    if system_text:
        if isinstance(system_text, list):
            # Anthropic allows system as list of content blocks
            parts = []
            for block in system_text:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            system_text = "\n".join(parts)
        messages.append({"role": "system", "content": system_text})

    # Conversation messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        # Anthropic content can be string or list of content blocks
        if isinstance(content, list):
            text_parts = []
            tool_results = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        # Assistant requesting tool use -> translate to function call
                        pass  # handled below
                    elif btype == "tool_result":
                        tool_results.append(block)
                    elif btype == "image":
                        text_parts.append("[image content omitted]")

            if tool_results:
                for tr in tool_results:
                    tool_content = tr.get("content", "")
                    if isinstance(tool_content, list):
                        parts = []
                        for tc in tool_content:
                            if isinstance(tc, dict) and tc.get("type") == "text":
                                parts.append(tc.get("text", ""))
                            elif isinstance(tc, str):
                                parts.append(tc)
                        tool_content = "\n".join(parts)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": str(tool_content),
                    })
            elif text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

            # Check for tool_use blocks in assistant messages
            if role == "assistant":
                tool_calls = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            }
                        })
                if tool_calls:
                    assistant_msg = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                        "tool_calls": tool_calls,
                    }
                    # Replace last message if we already added text
                    if text_parts and messages and messages[-1].get("role") == "assistant":
                        messages[-1] = assistant_msg
                    else:
                        messages.append(assistant_msg)
        elif isinstance(content, str):
            messages.append({"role": role, "content": content})

    # Tools translation
    tools = None
    anthropic_tools = body.get("tools")
    if anthropic_tools:
        tools = []
        for tool in anthropic_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }
            })

    openai_body = {
        "model": MERCURY_MODEL,
        "messages": messages,
        "max_tokens": body.get("max_tokens", DEFAULT_MAX_TOKENS),
        "temperature": body.get("temperature", 0.7),
        "reasoning_effort": REASONING_EFFORT,
        "stream": body.get("stream", False),
    }

    if tools:
        openai_body["tools"] = tools

    stop = body.get("stop_sequences")
    if stop:
        openai_body["stop"] = stop

    return openai_body


def openai_to_anthropic(oai_resp: dict, stream: bool = False) -> dict:
    """Convert OpenAI Chat Completion response to Anthropic Messages API format."""
    choice = oai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "end_turn")

    # Map finish reasons
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish, "end_turn")

    content_blocks = []

    # Text content
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw": func.get("arguments", "")}

            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{os.urandom(12).hex()}"),
                "name": func.get("name", ""),
                "input": arguments,
            })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = oai_resp.get("usage", {})

    return {
        "id": oai_resp.get("id", f"msg_{os.urandom(12).hex()}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": oai_resp.get("model", MERCURY_MODEL),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def build_anthropic_stream_events(oai_resp: dict) -> list[str]:
    """Convert a non-streaming OpenAI response into Anthropic SSE events."""
    anthropic_resp = openai_to_anthropic(oai_resp)
    events = []

    # message_start
    events.append(_sse({"type": "message_start", "message": {
        "id": anthropic_resp["id"],
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": anthropic_resp["model"],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": anthropic_resp["usage"]["input_tokens"], "output_tokens": 0},
    }}))

    # content blocks
    for idx, block in enumerate(anthropic_resp["content"]):
        if block["type"] == "text":
            events.append(_sse({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}}))
            # Send text in chunks for a streaming feel
            text = block["text"]
            chunk_size = 50
            for i in range(0, len(text), chunk_size):
                events.append(_sse({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": text[i:i+chunk_size]}}))
            events.append(_sse({"type": "content_block_stop", "index": idx}))

        elif block["type"] == "tool_use":
            events.append(_sse({"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            input_json = json.dumps(block["input"])
            events.append(_sse({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": input_json}}))
            events.append(_sse({"type": "content_block_stop", "index": idx}))

    # message_delta
    events.append(_sse({"type": "message_delta", "delta": {"stop_reason": anthropic_resp["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": anthropic_resp["usage"]["output_tokens"]}}))

    # message_stop
    events.append(_sse({"type": "message_stop"}))

    return events


def _sse(data: dict) -> str:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    api_key = ""

    def log_message(self, format, *args):
        log.info(f"{self.client_address[0]} - {format % args}")

    def do_POST(self):
        path = self.path

        # Handle Anthropic Messages API
        if "/v1/messages" in path:
            self._handle_messages()
        else:
            self.send_error(404, f"Unknown endpoint: {path}")

    def do_GET(self):
        # Health check / model list
        if self.path in ("/", "/health"):
            self._respond_json(200, {"status": "ok", "backend": "mercury-2"})
        elif "/v1/models" in self.path:
            self._respond_json(200, {
                "data": [{"id": "claude-sonnet-4-20250514", "object": "model"}],
                "object": "list",
            })
        else:
            self.send_error(404)

    def _handle_messages(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body)

            is_stream = body.get("stream", False)

            # Translate to OpenAI format (always non-streaming to Mercury)
            openai_body = anthropic_to_openai(body)
            openai_body["stream"] = False  # Mercury handles streaming differently

            log.info(f"→ Mercury 2 | model={openai_body['model']} | msgs={len(openai_body['messages'])} | stream={is_stream}")

            # Call Mercury 2
            mercury_resp = self._call_mercury(openai_body)

            if mercury_resp is None:
                self.send_error(502, "Mercury 2 API call failed")
                return

            if is_stream:
                # Send as SSE
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                for event in build_anthropic_stream_events(mercury_resp):
                    self.wfile.write(event.encode("utf-8"))
                    self.wfile.flush()
            else:
                anthropic_resp = openai_to_anthropic(mercury_resp)
                self._respond_json(200, anthropic_resp)

        except json.JSONDecodeError as e:
            log.error(f"JSON decode error: {e}")
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            log.error(f"Handler error: {e}", exc_info=True)
            self.send_error(500, str(e))

    def _call_mercury(self, openai_body: dict) -> dict | None:
        url = f"{MERCURY_BASE}/chat/completions"
        data = json.dumps(openai_body).encode("utf-8")

        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            log.error(f"Mercury API error {e.code}: {error_body}")
            return None
        except URLError as e:
            log.error(f"Mercury API connection error: {e}")
            return None

    def _respond_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mercury 2 proxy for Claude Code")
    parser.add_argument("--port", type=int, default=8082, help="Proxy port (default: 8082)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    api_key = os.environ.get("INCEPTION_API_KEY", "sk_7a760ef3d2444f756cb947511d21b9d5")
    if not api_key:
        print("ERROR: Set INCEPTION_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    ProxyHandler.api_key = api_key

    server = HTTPServer((args.host, args.port), ProxyHandler)
    log.info(f"Mercury 2 proxy started on http://{args.host}:{args.port}")
    log.info(f"Model: {MERCURY_MODEL} | Reasoning: {REASONING_EFFORT}")
    log.info("")
    log.info("To use with Claude Code:")
    log.info(f"  export ANTHROPIC_BASE_URL=http://{args.host}:{args.port}")
    log.info(f"  export ANTHROPIC_API_KEY=dummy")
    log.info(f"  claude")
    log.info("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
