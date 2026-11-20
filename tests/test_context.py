"""1.5.0's tests: the shared context (--context), the brief's note, and the check tried before any request.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import sys
import unittest

from test_worker import Home, check_for, user_text


def passes_anything(h):
    """A check that passes whatever output it is given"""
    ok = h.file("ok.py", "import sys\nsys.exit(0)\n")
    return f'"{sys.executable}" "{ok}" "{{out}}"'


class Context(unittest.TestCase):
    """--context: shared files that go into every request's user message first."""

    def test_batch_puts_the_context_first_in_every_request(self):
        h = Home(self)
        h.file("pack/a.md", "format A")
        h.file("pack/b.md", "format B")
        h.file("data.txt", "one\ntwo\nthree\nfour\n")
        h.file("own1.c", "own one")
        h.file("own2.c", "own two")
        tasks = [
            {"id": "a", "prompt": "[[say:one]]", "out": "1.md", "files": ["own1.c"]},
            {"id": "b", "prompt": "[[say:two]]", "out": "2.md", "files": ["own2.c"]},
        ]
        h.file("tasks.jsonl", "".join(json.dumps(t) + "\n" for t in tasks))
        rc, out, err = h.worker("batch", "tasks.jsonl", "--context", "pack", "--context", "data.txt:2-3")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("context 3 files", out)
        reqs = h.requests()
        self.assertEqual(len(reqs), 2)
        for req in reqs:  # in parallel: in either order
            text = user_text(req)
            own = "own one" if "own one" in text else "own two"
            self.assertLess(text.index("format A"), text.index("format B"))
            self.assertLess(text.index("format B"), text.index("two\nthree"))
            self.assertLess(text.index("two\nthree"), text.index(own))
            self.assertLess(text.index(own), text.index("[[say:"))
            self.assertNotIn("four", text)

    def test_a_tasks_own_context_wins(self):
        h = Home(self)
        h.file("shared.md", "SHARED")
        h.file("mine.md", "MINE")
        tasks = [
            {"id": "a", "prompt": "[[say:a]]", "out": "a.md", "context": ["mine.md"]},
            {"id": "b", "prompt": "[[say:b]]", "out": "b.md"},
        ]
        h.file("tasks.jsonl", "".join(json.dumps(t) + "\n" for t in tasks))
        rc, out, err = h.worker("batch", "tasks.jsonl", "--context", "shared.md")
        self.assertEqual(rc, 0, out + err)
        reqs = h.requests()
        self.assertEqual(len(reqs), 2)
        mine, = (user_text(r) for r in reqs if "[[say:a]]" in user_text(r))
        other, = (user_text(r) for r in reqs if "[[say:b]]" in user_text(r))
        self.assertIn("MINE", mine)
        self.assertNotIn("SHARED", mine)
        self.assertIn("SHARED", other)

    def test_the_context_is_not_part_of_the_work(self):
        h = Home(self)
        h.file("src.c", "int x;\n")
        h.file("pack.md", "PACKED")
        rc, out, err = h.worker("ask", "-f", "src.c", "-o", "a.md", "-p", "[[say:x]]")
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("ask", "-f", "src.c", "-o", "a.md", "-p", "[[say:x]]", "--context", "pack.md")
        self.assertEqual(rc, 0, out + err)
        usage = h.log("usage.jsonl")
        self.assertEqual(len(usage), 2)
        self.assertEqual(usage[0]["work"], usage[1]["work"])
        self.assertIn("PACKED", user_text(h.requests()[1]))

    def test_ask_puts_the_context_before_its_files(self):
        h = Home(self)
        h.file("rules.md", "RULES")
        h.file("src.c", "int x;\n")
        rc, out, err = h.worker("ask", "--context", "rules.md", "-f", "src.c", "-o", "o.md", "-p", "[[say:ok]]")
        self.assertEqual(rc, 0, out + err)
        text = user_text(h.requests()[0])
        self.assertLess(text.index("RULES"), text.index("int x;"))

    def test_a_missing_context_is_refused_before_anything_runs(self):
        h = Home(self)
        h.file("tasks.jsonl", json.dumps({"id": "a", "prompt": "[[say:a]]", "out": "a.md"}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--context", "nope.md")
        self.assertEqual(rc, 1, out)
        self.assertIn("--context nope.md: no such file or folder", out)
        self.assertEqual(len(h.requests()), 0)
        self.assertNotIn("Traceback", err)
        (h.work / "empty").mkdir()
        rc, out, err = h.worker("batch", "tasks.jsonl", "--context", "empty")
        self.assertEqual(rc, 1, out)
        self.assertIn("--context empty: no files in that folder", out)
        self.assertEqual(len(h.requests()), 0)

    def test_an_out_over_a_context_file_is_refused(self):
        h = Home(self)
        h.file("pack.md", "PACKED")
        h.file("tasks.jsonl", json.dumps({"id": "a", "prompt": "[[say:a]]", "out": "pack.md"}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--context", "pack.md")
        self.assertEqual(rc, 1, out)
        self.assertIn("one of the task's own files", out)
        self.assertEqual((h.work / "pack.md").read_text(encoding="utf-8"), "PACKED")
        self.assertEqual(len(h.requests()), 0)


class BriefNote(unittest.TestCase):
    """The note on stderr when a task writes code or asks for changes and no brief was found."""

    def test_a_task_that_writes_code_without_a_brief_is_told(self):
        h = Home(self)
        rc, out, err = h.worker("ask", "-o", "x.c", "-p", "[[say:int x;]]")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("no brief: ", err)
        self.assertIn('--brief ""', err)

    def test_a_map_is_not(self):
        h = Home(self)
        rc, out, err = h.worker("ask", "-o", "m.md", "-p", "[[say:map]] one line per function")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("no brief", err)

    def test_a_prompt_that_asks_for_changes_is(self):
        h = Home(self)
        h.file("tasks.jsonl", json.dumps({"id": "a", "prompt": "[[say:plan]] list the changes to apply",
                                          "out": "plan.md"}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("no brief: ", err)

    def test_not_with_a_brief_or_with_it_turned_off(self):
        h = Home(self)
        h.file(".cranker/brief.md", "Tabs.")
        rc, out, err = h.worker("ask", "-o", "x.c", "-p", "[[say:int x;]]")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("no brief", err)
        h2 = Home(self)
        rc, out, err = h2.worker("ask", "-o", "x.c", "-p", "[[say:int x;]]", "--brief", "")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("no brief", err)


class CheckFirst(unittest.TestCase):
    """The check is tried on a null output (and a known-good one) before any request."""

    def test_a_check_that_passes_an_empty_output_stops_the_run(self):
        h = Home(self)
        okcheck = passes_anything(h)
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", okcheck)
        self.assertEqual(rc, 1, out)
        self.assertIn("passes an empty output", out)
        self.assertEqual(len(h.requests()), 0)
        h.file("tasks.jsonl", json.dumps({"id": "a", "prompt": "[[say:good]]", "out": "x.txt"}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", okcheck)
        self.assertEqual(rc, 1, out)
        self.assertIn("task ", out)
        self.assertIn("passes an empty output", out)
        self.assertEqual(len(h.requests()), 0)

    def test_a_good_check_runs_and_says_how_it_was_tried(self):
        h = Home(self)
        check = check_for(h, "good")
        tasks = [
            {"id": "a", "prompt": "[[say:good]]", "out": "a.txt"},
            {"id": "b", "prompt": "[[say:good]]", "out": "b.txt"},
        ]
        h.file("tasks.jsonl", "".join(json.dumps(t) + "\n" for t in tasks))
        mine = h.file("a.cranker-probe.txt", "mine")  # a file that happens to have the probe's name is not touched
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", check)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("check, tried before any request: fails an empty output (2 tasks)", out)
        self.assertEqual(mine.read_text(encoding="utf-8"), "mine")
        mine.unlink()
        quality = h.log("quality.jsonl")
        self.assertEqual(len(quality), 2)
        for line in quality:
            self.assertEqual(line["proof"], "empty")
        usage = h.log("usage.jsonl")
        self.assertEqual(len(usage), 2)
        for line in usage:
            self.assertEqual(line["proof"], "empty")
        self.assertEqual([p for p in h.work.rglob("*") if "cranker-probe" in p.name], [])

    def test_the_null_and_the_good_outputs(self):
        h = Home(self)
        check = check_for(h, "good")
        h.file("null.txt", "good")
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", check,
                                "--check-null", "null.txt")
        self.assertEqual(rc, 1, out)
        self.assertIn("the null output", out)
        self.assertEqual(len(h.requests()), 0)
        h.file("good.txt", "bad")
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", check,
                                "--check-good", "good.txt")
        self.assertEqual(rc, 1, out)
        self.assertIn("fails the known-good output", out)
        self.assertEqual(len(h.requests()), 0)
        h.file("null.txt", "same")
        h.file("good.txt", "good")
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", check,
                                "--check-null", "null.txt", "--check-good", "good.txt")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(h.log("quality.jsonl")[-1]["proof"], "null+good")

    def test_none_skips_it_and_is_said_as_never_tried(self):
        h = Home(self)
        okcheck = passes_anything(h)
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", okcheck,
                                "--check-null", "none")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(h.log("quality.jsonl")[-1]["proof"], "none")

    def test_a_check_without_out_runs_only_when_said_untried(self):
        h = Home(self)
        ok = h.file("ok.py", "import sys\nsys.exit(0)\n")
        check = f'"{sys.executable}" "{ok}"'  # it reads the output from its own place: it cannot be tried on a copy
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", check)
        self.assertEqual(rc, 1, out)
        self.assertIn("does not name {out}", out)
        self.assertEqual(len(h.requests()), 0)
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", check, "--check-null", "none")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(h.log("quality.jsonl")[-1]["proof"], "none")

    def test_a_batch_tries_the_check_in_each_folder_of_its_outputs(self):
        h = Home(self)
        # passes anything in "lax/", and only "good" elsewhere: tried in the first folder alone, it would pass
        check = h.file("lax.py", "import sys\nfrom pathlib import Path\n"
                                 "text = open(sys.argv[1], encoding='utf-8').read()\n"
                                 "sys.exit(0 if Path(sys.argv[1]).parent.name == 'lax' or text == 'good' else 1)\n")
        tasks = [{"id": "a", "prompt": "[[say:good]]", "out": "strict/a.txt"},
                 {"id": "b", "prompt": "[[say:good]]", "out": "lax/b.txt"}]
        h.file("tasks.jsonl", "".join(json.dumps(t) + "\n" for t in tasks))
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", f'"{sys.executable}" "{check}" "{{out}}"')
        self.assertEqual(rc, 1, out)
        self.assertIn("task b: the check passes an empty output", out)
        self.assertEqual(len(h.requests()), 0)


class Measure(unittest.TestCase):
    """usage says what the right work stood on."""

    def test_usage_splits_the_right_work_by_its_proof(self):
        h = Home(self)
        check = check_for(h, "good")
        okcheck = passes_anything(h)
        rc, out, err = h.worker("ask", "-o", "a.txt", "-p", "[[say:good]]", "--check", check)
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("ask", "-o", "b.txt", "-p", "[[say:good]]", "--check", okcheck,
                                "--check-null", "none")
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("ask", "-o", "c.md", "-p", "[[say:map]]")
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("usage")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("Of the 3 right: 1 passed a check tried before it ran, 1 a check never tried, "
                      "1 had no check", out)


if __name__ == "__main__":
    unittest.main()
