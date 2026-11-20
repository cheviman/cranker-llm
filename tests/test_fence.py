"""An answer that is one fenced code block and nothing else (```c ... ```) is written without its fence, before the
check runs, so the check and its fixes go to the code. For a code file (not a document: a map, a review, notes),
an answer with exactly one block and text around it has the block written to the file and the text to
<out>.notes.md, as the two real answers of 20-11-2026 in fixtures/ needed. Any other answer is written as it came.

unfence() and split_answer() are tried on answers alone; then worker.py ask and batch run as they do for the user,
against test_cut.py's fake llama-server answering what each test gives it for its prompt, with CRANKER_HOME a
temporary folder.

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
from test_worker import Home

_spec = importlib.util.spec_from_file_location("worker", WORKER)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

C = "#include <stdio.h>\n\nint main(void)\n{\n    puts(\"janus\");\n    return 0;\n}\n"
WRONG = C.replace("return 0;", "return 1;")

# what deepseek-v4.1-flash answered on 20-11-2026 (batch --remote), asked for a Python test: its block, and a
# paragraph after it. Written as it came, the check failed with a SyntaxError on line 1 (the fence), and after the
# fix on line 141 (the paragraph)
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRY1 = (FIXTURES / "prose_after_block.try1.answer").read_text(encoding="utf-8")
TRY2 = (FIXTURES / "prose_after_block.try2.answer").read_text(encoding="utf-8")


def block_of(answer):
    """The lines between an answer's first fence line and the next one: what the file should hold"""
    lines = answer.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("```"))
    end = next(i for i, l in enumerate(lines) if i > start and l.startswith("```"))
    return "".join(lines[start + 1:end])


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


class SplitAnswerTests(unittest.TestCase):
    def test_the_20_11_answers_split_into_their_code_and_their_paragraph(self):
        for answer, first in ((TRY1, "What is still uncertain:"), (TRY2, "I have to be honest")):
            with self.subTest(first=first):
                code, around, blocks = worker.split_answer(answer)
                self.assertEqual(blocks, 1)
                self.assertEqual(code, block_of(answer))
                compile(code, "subdir_project.py", "exec")  # what the check failed on: it is Python now
                self.assertTrue(around.startswith(first), around[:80])
                self.assertNotIn("```", around)

    def test_text_before_and_after_both_go_aside(self):
        code, around, blocks = worker.split_answer(f"The file:\n\n```c\n{C}```\n\nIt compiles with gcc.\n")
        self.assertEqual((code, around, blocks), (C, "The file:\n\nIt compiles with gcc.", 1))

    def test_one_block_alone_has_nothing_aside(self):
        self.assertEqual(worker.split_answer(f"```c\n{C}```\n"), (C, "", 1))

    def test_two_blocks_or_a_fence_left_open_are_not_taken_apart(self):
        for text, blocks in ((f"```c\n{C}```\n\nand the test:\n\n```c\n{WRONG}```\n", 2),
                             (f"Here:\n```c\n{C}", 0), (f"```markdown\n# Notes\n\n```c\n{C}```\n```\n", 0), (C, 0)):
            with self.subTest(text=text[:40]):
                self.assertEqual(worker.split_answer(text), (text, "", blocks))

    def test_what_is_a_document_and_what_is_code(self):
        for name in ("map.md", "review.MD", "notes.txt", "build.log", "answer"):
            self.assertTrue(worker.is_document(name), name)
        for name in ("subdir_project.py", "janus.c", "saturn.cmake", "CMakeLists.txt", "Makefile", "page.html"):
            self.assertFalse(worker.is_document(name), name)


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

    def python_check(self, needs=None):
        """A check that compiles the output as Python (what failed on 20-11-2026), and with `needs`, wants that text
        in it. Each run prints which round it is, from a count kept in rounds.txt"""
        (self.tmp / "check_py.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "count = Path(sys.argv[2])\nn = int(count.read_text() or 0) + 1 if count.exists() else 1\n"
            "count.write_text(str(n))\n"
            "text = open(sys.argv[1], encoding='utf-8').read()\n"
            "try:\n    compile(text, sys.argv[1], 'exec')\n"
            "except SyntaxError as e:\n    sys.exit(print(f'round {n}: SyntaxError on line {e.lineno}') or 1)\n"
            "if len(sys.argv) > 3 and sys.argv[3] not in text:\n"
            "    sys.exit(print(f'round {n}: the file must say {sys.argv[3]}') or 1)\n", encoding="utf-8")
        return (f'"{sys.executable}" "{self.tmp / "check_py.py"}" "{{out}}" "{self.tmp / "rounds.txt"}"'
                + (f' "{needs}"' if needs else ""))

    def test_a_code_answer_with_a_paragraph_after_its_block_passes_and_keeps_the_paragraph(self):
        # the 20-11-2026 answer: on main the fence and the paragraph went into the file, a SyntaxError on line 1
        FencedLlama.answers = {"write the test": [TRY1]}
        out = self.tmp / "subdir_project.py"
        code = self.worker("ask", "-p", "write the test", "-o", str(out), "--check", self.python_check())
        self.assertEqual(code, 0, self.stdout)
        self.assertIn("check passed", self.stdout)
        self.assertNotIn("check passed after", self.stdout)
        self.assertEqual(out.read_text("utf-8"), block_of(TRY1))
        notes = self.tmp / "subdir_project.py.notes.md"
        self.assertIn(f"the text around the code kept apart in {notes}", self.stdout)
        self.assertTrue(notes.read_text("utf-8").startswith("## answer 1\n\nWhat is still uncertain:"))
        self.assertEqual(len(FencedLlama.requests), 1)

    def test_a_fix_asks_for_the_block_alone_and_each_answer_keeps_its_note(self):
        # the second 20-11-2026 answer first (its block compiles, but the check wants a line it lacks), then the
        # first one with that line: both came with a paragraph after the block
        fixed = TRY1.replace("```python\n", "```python\n# fixed\n", 1)
        FencedLlama.answers = {"write the test": [TRY2, fixed]}
        out = self.tmp / "subdir_project.py"
        code = self.worker("ask", "-p", "write the test", "-o", str(out), "--check", self.python_check("# fixed"))
        self.assertEqual(code, 0, self.stdout)
        self.assertIn("check passed after 1 fix by the worker", self.stdout)
        self.assertEqual(out.read_text("utf-8"), block_of(fixed))
        fix = FencedLlama.requests[1]["messages"][-1]["content"]
        self.assertIn("round 1: the file must say # fixed", fix)
        self.assertIn("as one fenced code block and nothing before or after it", fix)
        notes = (self.tmp / "subdir_project.py.notes.md").read_text("utf-8")
        self.assertRegex(notes, r"(?s)^## answer 1\n\nI have to be honest.*\n## answer 2\n\nWhat is still uncertain:")
        self.assertFalse((self.tmp / "subdir_project.py.check.1.txt").exists())  # it passed: no round is left

    def test_every_round_of_a_failing_check_keeps_its_output(self):
        # only the last round's output was kept, and the first failure of a run that ended failing was lost
        FencedLlama.answers = {"write the test": ["```python\nprint(1\n```\n"]}
        out = self.tmp / "t.py"
        code = self.worker("ask", "-p", "write the test", "-o", str(out), "--check", self.python_check())
        self.assertEqual(code, 2, self.stdout)
        self.assertIn("CHECK FAILED after 2 fixes", self.stdout)
        for n in (1, 2, 3):
            self.assertIn(f"round {n}: SyntaxError", (self.tmp / f"t.py.check.{n}.txt").read_text("utf-8"))
        note = (self.tmp / "t.py.check.txt").read_text("utf-8")
        self.assertIn("round 3: SyntaxError", note)
        self.assertIn("t.py.check.1.txt, t.py.check.2.txt, t.py.check.3.txt", note)

    def test_two_blocks_in_a_code_file_are_written_as_they_came_and_said(self):
        answer = f"```c\n{C}```\n\nand its test:\n\n```c\n{WRONG}```\n"
        FencedLlama.answers = {"write janus.c": [answer]}
        out = self.tmp / "janus.c"
        self.assertEqual(self.worker("ask", "-p", "write janus.c", "-o", str(out)), 0, self.stdout)
        self.assertEqual(out.read_text("utf-8"), answer)
        self.assertIn("2 fenced blocks: written as they came", self.stdout)
        self.assertFalse((self.tmp / "janus.c.notes.md").exists())


class StreamedFenceTests(unittest.TestCase):
    """The same 20-11-2026 answer through tests/fake_llama.py, streamed a word an event as the servers send it"""

    def test_the_streamed_answer_is_split_the_same_way(self):
        h = Home(self)
        check = h.file("check_py.py", "import sys\ncompile(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1], "
                                      "'exec')\n")
        prompt = f"[[say-file:{FIXTURES / 'prose_after_block.try1.answer'}]] write the test"
        rc, out, err = h.worker("ask", "-o", "subdir_project.py", "-p", prompt,
                                "--check", f'"{sys.executable}" "{check}" {{out}}')
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(h.requests()[0]["stream"])
        self.assertEqual((h.work / "subdir_project.py").read_text(encoding="utf-8"), block_of(TRY1))
        notes = (h.work / "subdir_project.py.notes.md").read_text(encoding="utf-8")
        self.assertIn("What is still uncertain:", notes)


if __name__ == "__main__":
    unittest.main()
