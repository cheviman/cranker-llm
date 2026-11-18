"""A batch with a check, on answers cut by max_tokens: the cut is said first, the retry asks for the same answer with
more room (1.5 times, as far as the context holds it), once, and a task cut even so needs splitting.

The worker runs as it does for the user (scripts/worker.py batch, a process of its own) against a fake llama-server:
the answer the model "needs" is in the prompt (ANSWER n: n words, the last one END; THINK n: the thinking it would
do, up to its budget), a token is a word, and an answer past max_tokens is cut with finish_reason 'length', as the
server does. CRANKER_HOME is a temporary folder, so the user's logs are not touched.

    py -m unittest discover -s tests -v
"""

import http.server
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

WORKER = Path(__file__).resolve().parent.parent / "scripts" / "worker.py"
SYSTEM = "You are a test."  # 4 words: with "ANSWER n please", a prompt of 7 tokens, 71 with needed_tokens' 64
ADMIT_MARGIN = 1024         # worker.py's: the context kept free beyond the requests


class FakeLlama(http.server.BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"/health": {"status": "ok"}, "/props": {"model_path": "fake.gguf"}}.get(self.path, []))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._send({"tokens": [0] * len(body["content"].split())})
        first = body["messages"][1]["content"]
        answer = int(re.search(r"ANSWER (\d+)", first).group(1))
        think = bool(body.get("chat_template_kwargs", {}).get("enable_thinking"))
        budget = body.get("reasoning_budget_tokens", 0) if think else 0
        m = re.search(r"THINK (\d+)", first)
        thought = min(budget, int(m.group(1)) if m else budget)  # Qwen thinks up to its budget
        limit = body["max_tokens"]
        words = [f"w{i}" for i in range(1, answer)] + ["END"]
        room = max(0, limit - thought)
        FakeLlama.requests.append({"max_tokens": limit, "budget": body.get("reasoning_budget_tokens"),
                                   "think": think, "turns": len(body["messages"])})
        self._send({
            "choices": [{"message": {"content": " ".join(words[:room]), "reasoning_content": "t " * thought},
                         "finish_reason": "length" if thought + answer > limit else "stop"}],
            "usage": {"prompt_tokens": sum(len(m["content"].split()) for m in body["messages"]),
                      "completion_tokens": min(limit, thought + answer)},
            "timings": {},
        })


class CutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLlama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeLlama.requests = []
        self.tmp = Path(tempfile.mkdtemp(prefix="cranker-test-"))
        (self.tmp / "config.json").write_text(json.dumps({"thermal_pause": False}), encoding="utf-8")
        for name, code in {"pass": "0", "fail": "(print('always wrong'), 1)[1]",
                           "end": "0 if open(sys.argv[1]).read().split()[-1:] == ['END'] else "
                                  "(print('the file does not end with END'), 1)[1]"}.items():
            (self.tmp / f"check_{name}.py").write_text(f"import sys\nsys.exit({code})\n", encoding="utf-8")

    def batch(self, task, check, retries=2, ctx=None):
        out = self.tmp / "out.txt"
        task = {"id": "t", "system": SYSTEM, "out": str(out), **task}
        (self.tmp / "tasks.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
        env = {**os.environ, "CRANKER_HOME": str(self.tmp), "CRANKER_PORT": str(self.server.server_port),
               "CRANKER_BRIEF": "", "PYTHONUTF8": "1"}
        if ctx:
            env["CRANKER_CTX"] = str(ctx)
        cmd = f'"{sys.executable}" "{self.tmp / f"check_{check}.py"}" "{{out}}"'
        r = subprocess.run([sys.executable, str(WORKER), "batch", str(self.tmp / "tasks.jsonl"), "--check", cmd,
                            "--retries", str(retries)], cwd=self.tmp, env=env, capture_output=True, text=True,
                           encoding="utf-8", timeout=120)
        self.stdout = r.stdout + r.stderr
        line = next((l for l in r.stdout.splitlines() if re.match(r"^(ok  |CUT |BAD |FAIL)t: ", l)), "")
        usage = [json.loads(l) for l in (self.tmp / "logs" / "usage.jsonl").read_text("utf-8").splitlines()]
        quality = [json.loads(l) for l in (self.tmp / "logs" / "quality.jsonl").read_text("utf-8").splitlines()]
        note = out.with_name(out.name + ".check.txt")
        return r.returncode, line, usage, quality[-1], note.read_text("utf-8") if note.exists() else None, out

    def limits(self):
        return [q["max_tokens"] for q in FakeLlama.requests]

    def test_cut_with_a_trivial_check_is_asked_again_with_more_room(self):
        code, line, usage, quality, note, out = self.batch({"prompt": "ANSWER 300 please", "max_tokens": 250},
                                                           "pass")
        self.assertEqual(self.limits(), [250, 375], self.stdout)
        self.assertEqual(FakeLlama.requests[1]["turns"], 2)  # the same request, not a fix with the cut answer in it
        self.assertTrue(line.startswith("ok  t: "), self.stdout)
        self.assertIn("check passed after 1 fix by the worker (max_tokens raised 250 -> 375 after a cut)", line)
        self.assertIn("cuts: 1 answers cut by max_tokens asked for again", self.stdout)
        self.assertEqual(out.read_text("utf-8").split()[-1], "END")
        self.assertIsNone(note)
        self.assertEqual(code, 0)
        # usage.jsonl: one line per request as before, the cut one says at what limit
        self.assertEqual([(u.get("attempt"), u.get("cut"), u["out"]) for u in usage], [(None, 250, 250), (1, None, 300)])
        self.assertEqual((quality["result"], quality["tries"], quality["raised"]), ("fixed", 1, [250, 375]))

    def test_still_cut_after_the_raise_stops_and_needs_splitting(self):
        code, line, usage, quality, note, out = self.batch({"prompt": "ANSWER 1000 please", "max_tokens": 250},
                                                           "end", retries=3)
        self.assertEqual(self.limits(), [250, 375], self.stdout)  # no more tries, though retries were left
        self.assertTrue(line.startswith("CUT t: cut by max_tokens 375, after raising it from 250 for a first cut: "
                                        "the task needs splitting"), self.stdout)
        self.assertIn("1 need splitting", self.stdout)
        self.assertTrue(note.startswith("cut by max_tokens 375, after raising it from 250"), note)
        self.assertIn("the file does not end with END", note)  # the check's message still follows
        self.assertEqual(code, 2)
        self.assertEqual([u.get("cut") for u in usage], [250, 375])
        self.assertEqual((quality["result"], quality["cut"], quality["split"]), ("fail", 375, True))

    def test_the_raise_stops_at_what_the_context_holds(self):
        # 71 tokens of prompt (needed_tokens' accounting) and 320 of room beside it: 375 does not fit
        code, line, *_ = self.batch({"prompt": "ANSWER 300 please", "max_tokens": 250}, "end",
                                    ctx=ADMIT_MARGIN + 71 + 320)
        self.assertEqual(self.limits(), [250, 320], self.stdout)
        self.assertIn("max_tokens raised 250 -> 320 after a cut", line)
        self.assertEqual(code, 0)

    def test_no_room_to_raise_needs_splitting_at_once(self):
        code, line, usage, quality, note, out = self.batch({"prompt": "ANSWER 300 please", "max_tokens": 250},
                                                           "end", ctx=ADMIT_MARGIN + 71 + 250)
        self.assertEqual(self.limits(), [250], self.stdout)
        self.assertIn("cut by max_tokens 250, and the context has no room to raise it", line)
        self.assertIn("needs splitting", note.splitlines()[0])
        self.assertEqual(code, 2)

    def test_a_thinking_answer_keeps_its_budget_and_gains_the_room(self):
        # think_budget(3000) = 2000: the model thinks all of it and has 1000 left for an answer of 1500
        code, line, usage, *_ = self.batch({"prompt": "ANSWER 1500 please", "max_tokens": 3000, "think": True},
                                           "end")
        self.assertEqual([(q["max_tokens"], q["budget"]) for q in FakeLlama.requests], [(3000, 2000), (4500, 2000)])
        self.assertEqual(code, 0, self.stdout)
        self.assertEqual(usage[0]["cut"], 3000)

    def test_a_cut_answer_without_retries_says_so_first(self):
        code, line, usage, quality, note, out = self.batch({"prompt": "ANSWER 300 please", "max_tokens": 250},
                                                           "end", retries=0)
        self.assertEqual(self.limits(), [250])
        self.assertTrue(line.startswith("CUT t: cut by max_tokens 250: the answer is unfinished"), self.stdout)
        self.assertTrue(note.startswith("cut by max_tokens 250"), note)
        self.assertEqual(code, 2)

    def test_the_fix_that_thinks_keeps_the_answer_room(self):
        # two plain fixes that do not take: the second thinks, with the task's 4000 for the answer and the
        # thinking budget on top. With the thinking default instead (10000, 7952 of it thinking) the 3000-token
        # answer was cut, as on 17-11-2026
        code, line, usage, *_ = self.batch({"prompt": "ANSWER 3000 please", "max_tokens": 4000}, "fail")
        self.assertEqual([(q["max_tokens"], q["budget"], q["think"]) for q in FakeLlama.requests],
                         [(4000, None, False), (4000, None, False), (4000 + 7952, 7952, True)], self.stdout)
        self.assertEqual([u.get("cut") for u in usage], [None, None, None])
        self.assertTrue(line.startswith("BAD t: "), self.stdout)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
