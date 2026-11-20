"""A stand-in for llama-server, for the tests: it takes llama-server's command line and answers the endpoints the
worker uses (/health, /props, /slots, /tokenize, /v1/chat/completions, /v1/messages), with no model and no GPU.

What a chat request gets back is scripted in its first user message:

    [[say:TEXT]]       the answer (default "ok")
    [[say-file:PATH]]  the answer is that file's text (a real answer kept as a test's fixture)
    [[loop:TEXT]]      streamed: TEXT over and over, {n} in it counting 1, 2, 3..., until max_tokens words
    [[then:TEXT]]      the answer once the request carries more turns (the worker's fix after a failed check)
    [[cut]]            finish_reason "length"
    [[sleep:SECONDS]]  answer after that long
    [[fail:CODE:TEXT]] an HTTP error
    [[empty-think]]    with thinking on: all thinking and no answer; with it off: the answer "recovered"
    [[busy:N]]         HTTP 429 with Retry-After: 0 the first N times this prompt is asked, as a hosted server does
    [[die]]            the server exits without answering

A token is a word. A chat request with "stream": true is answered as llama-server and OpenAI do, as server-sent
events: a word an event, the counts in the last one. --load S answers /health with 503 for the first S seconds, as
llama-server does while it loads the model; --record FILE appends every chat request's body to FILE, one JSON a
line, with its Authorization header under "auth"; --streams FILE appends, for every streamed answer, how many words
it sent and whether the client closed the connection before its end; --key KEY answers 401 to a chat request
without that key, and 400 to one with a field that is not OpenAI's (the tests use it as the server elsewhere of
--remote). Standard library only.
"""
import argparse
import http.server
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

DIRECTIVE = re.compile(r"\[\[([a-z-]+)(?::(.*?))?\]\]", re.S)
LATE_SYSTEM = "{{- raise_exception('System message must be at the beginning.') }}"


def text_of(content):
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", dest="model", default="fake.gguf")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("-c", dest="ctx", type=int, default=65536)
    ap.add_argument("-np", dest="slots", type=int, default=8)
    ap.add_argument("--chat-template-file")
    for name in ("-a", "-ngl", "-fa", "-ctk", "-ctv", "-t"):  # named, or argparse reads -ctk as -c "tk"
        ap.add_argument(name)
    ap.add_argument("--load", type=float, default=float(os.environ.get("FAKE_LLAMA_LOAD", 0)))
    ap.add_argument("--record", default=os.environ.get("FAKE_LLAMA_RECORD"))
    ap.add_argument("--key")
    ap.add_argument("--streams", default=os.environ.get("FAKE_LLAMA_STREAMS"))
    args, _ = ap.parse_known_args()
    t0 = time.time()
    lock = threading.Lock()
    running = {}  # task id: its slot's state
    counter = [0]
    asked = {}  # a [[busy:N]] prompt: how many times it was asked
    openai = {"model", "messages", "max_tokens", "temperature", "top_p", "reasoning_effort", "stream", "stream_options"}
    template = (Path(args.chat_template_file).read_text(encoding="utf-8") if args.chat_template_file
                else "{%- for message in messages %}" + LATE_SYSTEM + "{%- endfor %}")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream(self, words, thoughts, finish, n_prompt):
            """The answer as server-sent events, a word an event, until its end or until the client leaves"""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            sent, closed = 0, False

            def event(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")

            try:
                event({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]})
                for thought in thoughts:
                    event({"choices": [{"index": 0, "delta": {"reasoning_content": thought}}]})
                for word in words:
                    event({"choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]})
                    sent += 1
                event({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                event({"choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": sent + len(thoughts)},
                       "timings": {"prompt_per_second": 1000.0, "predicted_per_second": 30.0}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:  # the client left: ConnectionAbortedError, ConnectionResetError, BrokenPipeError
                closed = True
            if args.streams:
                with lock, open(args.streams, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sent": sent, "closed": closed}) + "\n")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                if time.time() - t0 < args.load:
                    return self.send_json({"error": {"message": "Loading model", "code": 503}}, 503)
                return self.send_json({"status": "ok"})
            if path == "/props":
                return self.send_json({"model_path": args.model, "total_slots": args.slots, "chat_template": template,
                                       "default_generation_settings": {"n_ctx": args.ctx}})
            if path == "/slots":
                with lock:
                    busy = list(running.values())
                idle = [{"id": i, "is_processing": False} for i in range(len(busy), args.slots)]
                return self.send_json(busy + idle)
            if path == "/quit":  # the tests' way to end one that a failed test left behind
                self.send_json({"status": "bye"})
                os._exit(0)
            self.send_json({"error": {"message": "not found"}}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            path = self.path.split("?")[0]
            if path == "/tokenize":
                return self.send_json({"tokens": [0] * len(body.get("content", "").split())})
            if path not in ("/v1/chat/completions", "/v1/messages"):
                return self.send_json({"error": {"message": "not found"}}, 404)
            if args.record:
                with lock, open(args.record, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"path": path, "auth": self.headers.get("Authorization"), **body}) + "\n")
            if args.key:  # a hosted server: its key, and OpenAI's fields only
                if self.headers.get("Authorization") != "Bearer " + args.key:
                    return self.send_json({"error": {"code": "invalid_key", "type": "authentication_error"}}, 401)
                unknown = sorted(set(body) - openai)
                if unknown:
                    return self.send_json({"error": {"code": "unsupported_parameter", "param": unknown[0]}}, 400)
            messages = body.get("messages") or []
            first = next((text_of(m.get("content")) for m in messages if m.get("role") == "user"), "")
            said = dict(DIRECTIVE.findall(first))
            if "busy" in said:
                with lock:
                    asked[first] = asked.get(first, 0) + 1
                    limited = asked[first] <= int(said["busy"])
                if limited:
                    body = json.dumps({"error": {"type": "rate_limit_error", "message": "slow down"}}).encode("utf-8")
                    self.send_response(429)
                    self.send_header("Retry-After", "0")
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
            n_prompt = sum(len(text_of(m.get("content")).split()) for m in messages)
            with lock:
                counter[0] += 1
                task = counter[0]
                running[task] = {"id": len(running), "id_task": task, "is_processing": True, "n_prompt_tokens": n_prompt,
                                 "next_token": [{"n_decoded": 0, "n_remain": body.get("max_tokens", -1)}]}
            try:
                if "die" in said:
                    os._exit(1)
                time.sleep(float(said.get("sleep") or 0))
                if "fail" in said:
                    code, _, message = said["fail"].partition(":")
                    return self.send_json({"error": {"message": message, "code": int(code)}}, int(code))
                answer, reasoning = said.get("say", "ok"), ""
                if "say-file" in said:
                    answer = Path(said["say-file"]).read_text(encoding="utf-8")
                if "then" in said and len(messages) > 2:
                    answer = said["then"]
                if "empty-think" in said:
                    answer, reasoning = ("", "half a thought") if thinking else ("recovered", "")
                finish = "length" if "cut" in said else "stop"
                if body.get("stream") and path == "/v1/chat/completions":
                    if "loop" in said:
                        words, finish = looping(said["loop"], body.get("max_tokens") or 4096), "length"
                    else:
                        words = re.findall(r"\s*\S+\s*", answer)
                    return self.stream(words, re.findall(r"\S+\s*", reasoning), finish, n_prompt)
                n_out = len((answer + " " + reasoning).split())
                if path == "/v1/messages":
                    return self.send_json({
                        "type": "message", "role": "assistant", "content": [{"type": "text", "text": answer}],
                        "stop_reason": "max_tokens" if finish == "length" else "end_turn",
                        "usage": {"input_tokens": n_prompt, "output_tokens": n_out}})
                self.send_json({
                    "choices": [{"finish_reason": finish,
                                 "message": {"role": "assistant", "content": answer, "reasoning_content": reasoning}}],
                    "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out},
                    "timings": {"prompt_per_second": 1000.0, "predicted_per_second": 30.0}})
            finally:
                with lock:
                    running.pop(task, None)

    def looping(text, limit):
        """TEXT over and over, with {n} counting, word by word: `limit` words in all"""
        n = 0
        while limit > 0:
            n += 1
            for word in re.findall(r"\s*\S+\s*", text.replace("{n}", str(n)) + "\n")[:limit]:
                limit -= 1
                yield word

    class Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = False  # on Windows it would let a second server take the same port
        daemon_threads = True

    server = Server((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
