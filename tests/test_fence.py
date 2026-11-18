"""An answer that is one fenced code block and nothing else (```c ... ```) is written without its fence, before the
check runs, so the check and its fixes go to the code; any other answer is written as it came.

unfence() is tried on answers alone; then worker.py ask and batch run as they do for the user, against test_cut.py's
fake llama-server answering what each test gives it for its prompt, with CRANKER_HOME a temporary folder.

    py -m unittest discover -s tests -v
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import http.server
import threading
from pathlib import Path

from test_cut import FakeLlama, WORKER

_spec = importlib.util.spec_from_file_location("worker", WORKER)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

C = "#include <stdio.h>\n\nint main(void)\n{\n    puts(\"janus\");\n    return 0;\n}\n"
WRONG = C.replace("return 0;", "return 1;")


class UnfenceTests(unittest.TestCase):
    def test_one_block_with_a_language_tag_is_its_inside(self):
        self.assertEqual(worker.unfence(f"```c\n{C}```"), C)

    def test_one_block_without_a_language_tag(self):
        self.assertEqual(worker.unfence(f"```\n{C}```\n"), C)

    def test_one_block_with_crlf_keeps_its_line_ends(self):
        crlf = C.replace("\n", "\r\n")
        self.assertEqual(worker.unfence(f"```c\r\n{crlf}```\r\n"), crlf)

    def test_whitespace_around_the_block_is_aside(self):
        self.assertEqual(worker.unfence(f"\n\n  ```c\n{C}```  \n\n"), C)

    def test_an_empty_block_is_empty(self):
        self.assertEqual(worker.unfence("```c\n```"), "")

    def test_backticks_inside_a_line_are_not_a_fence(self):
        code = 'FENCE = "```"\nprint(FENCE)\n'
        self.assertEqual(worker.unfence(f"```python\n{code}```"), code)

    def test_any_other_answer_is_unchanged(self):
        for text in (
            f"Here is the file:\n\n```c\n{C}```\n",               # prose before
            f"```c\n{C}```\n\nIt compiles with gcc.\n",           # prose after
            f"```c\n{C}```\n\n```c\n{WRONG}```\n",                # two blocks
            f"```c\r\n{C}```\r\n\r\n```c\r\n{C}```\r\n",          # two blocks, CRLF
            f"# Notes\n\nThe file:\n\n```c\n{C}```\n\nMore.\n",   # a markdown file
            f"```markdown\n# Notes\n\n```c\n{C}```\n```\n",       # a fence inside the block
            C, "", "```c int x; ```", f"```c\n{C}",               # no fence, empty, one line, not closed
        ):
            with self.subTest(text=text[:40]):
                self.assertEqual(worker.unfence(text), text)


class FencedLlama(FakeLlama):
    """Answers each prompt with the next of the answers the test gave for it, and the last one again once they run
    out (as a model that does not fix what the check says)."""
    answers = {}
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._send({"tokens": [0] * len(body["content"].split())})
        FencedLlama.requests.append(body)
        left = FencedLlama.answers[body["messages"][1]["content"]]
        content = left.pop(0) if len(left) > 1 else left[0]
        self._send({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": len(content.split())}, "timings": {}})


# a compiler's view: a fence line is a syntax error on its line, and main must return 0. Every file it is run on is
# added to seen.jsonl, so the test knows what the check saw
CHECK = """import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
with open(sys.argv[2], "a", encoding="utf-8") as f:
    f.write(json.dumps(text) + "\\n")
for n, line in enumerate(text.splitlines(), 1):
    if line.startswith("```"):
        sys.exit(print(f"{sys.argv[1]}:{n}:1: error: expected identifier or '(' before '`' token") or 1)
if "return 0;" not in text:
    sys.exit(print(f"{sys.argv[1]}: error: main must return 0") or 1)
"""


class FenceOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FencedLlama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FencedLlama.answers, FencedLlama.requests = {}, []
        self.tmp = Path(tempfile.mkdtemp(prefix="cranker-test-"))
        (self.tmp / "config.json").write_text(json.dumps({"thermal_pause": False}), encoding="utf-8")
        (self.tmp / "check_c.py").write_text(CHECK, encoding="utf-8")
        self.check = f'"{sys.executable}" "{self.tmp / "check_c.py"}" "{{out}}" "{self.tmp / "seen.jsonl"}"'

    def worker(self, *args):
        env = {**os.environ, "CRANKER_HOME": str(self.tmp), "CRANKER_PORT": str(self.server.server_port),
               "CRANKER_BRIEF": "", "PYTHONUTF8": "1"}
        r = subprocess.run([sys.executable, str(WORKER), *args], cwd=self.tmp, env=env, capture_output=True,
                           text=True, encoding="utf-8", timeout=120)
        self.stdout = r.stdout + r.stderr
        return r.returncode

    def seen(self):
        return [json.loads(l) for l in (self.tmp / "seen.jsonl").read_text("utf-8").splitlines()]

    def test_ask_writes_the_inside_and_the_check_sees_it(self):
        FencedLlama.answers = {"write janus.c": [f"```c\n{C}```\n"]}
        out = self.tmp / "janus.c"
        code = self.worker("ask", "-p", "write janus.c", "-o", str(out), "--check", self.check)
        self.assertEqual(code, 0, self.stdout)
        self.assertIn("check passed", self.stdout)
        self.assertNotIn("check passed after", self.stdout)  # the first round, not a fix
        self.assertEqual(out.read_text("utf-8"), C)
        self.assertEqual(self.seen(), [C])
        self.assertEqual(len(FencedLlama.requests), 1)

    def test_the_fixes_go_to_the_code_not_the_fence(self):
        # both answers fenced: the check's error is about the code, and so is the fix
        FencedLlama.answers = {"write janus.c": [f"```c\n{WRONG}```", f"```c\n{C}```"]}
        out = self.tmp / "janus.c"
        code = self.worker("ask", "-p", "write janus.c", "-o", str(out), "--check", self.check)
        self.assertEqual(code, 0, self.stdout)
        self.assertIn("check passed after 1 fix by the worker", self.stdout)
        self.assertEqual(self.seen(), [WRONG, C])
        fix = FencedLlama.requests[1]["messages"][-1]["content"]
        self.assertIn("main must return 0", fix)
        self.assertNotIn("'`' token", fix)
        self.assertEqual(out.read_text("utf-8"), C)

    def test_batch_unwraps_one_block_and_writes_any_other_answer_as_it_came(self):
        answers = {"one block": f"```c\n{C}```", "prose and a block": f"The file:\n\n```c\n{C}```\n",
                   "two blocks": f"```c\n{C}```\n\n```c\n{WRONG}```\n"}
        FencedLlama.answers = {p: [a] for p, a in answers.items()}
        outs = {p: self.tmp / f"{p.replace(' ', '_')}.md" for p in answers}
        (self.tmp / "tasks.jsonl").write_text(
            "".join(json.dumps({"id": p, "prompt": p, "out": str(o)}) + "\n" for p, o in outs.items()), "utf-8")
        code = self.worker("batch", str(self.tmp / "tasks.jsonl"))
        self.assertEqual(code, 0, self.stdout)
        # byte for byte as write_out has always written them: UTF-8, in text mode ('\n' is the platform's line end)
        expected = {**answers, "one block": C}
        self.assertEqual({p: o.read_bytes() for p, o in outs.items()},
                         {p: a.replace("\n", os.linesep).encode("utf-8") for p, a in expected.items()})


if __name__ == "__main__":
    unittest.main()
