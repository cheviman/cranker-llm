"""1.7.0's tests: a wide batch of changes asks, on its first run, whether what it can reach was measured (20/21-11-2026:
three batches of 25, 14 and 14 against a measure mostly out of reach). --reach FILE says it was.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import unittest

from test_worker import Home

NOTE = "wide batch of changes"


def changes(h, n):
    h.file("tasks.jsonl", "".join(json.dumps({"id": f"c{i}", "prompt": "[[say:int x;]]", "out": f"c{i}.c"}) + "\n"
                                  for i in range(n)))


class Reach(unittest.TestCase):
    def test_a_wide_batch_of_changes_is_asked_on_its_first_run(self):
        h = Home(self)
        changes(h, 8)
        rc, out, err = h.worker("batch", "tasks.jsonl")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"{NOTE}: 8 tasks that write changes, on their first run", err)
        self.assertIn("--reach FILE", err)
        (h.work / "c0.c").unlink()  # the next run of the same batch: not asked again
        rc, out, err = h.worker("batch", "tasks.jsonl")
        self.assertNotIn(NOTE, err)

    def test_not_with_reach_and_it_is_said(self):
        h = Home(self)
        changes(h, 8)
        h.file("reach.txt", "12 of 140 uses can move")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--reach", "reach.txt")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn(NOTE, err)
        self.assertIn("reach measured in reach.txt", out)
        rc, out, err = h.worker("batch", "tasks.jsonl", "--reach", "missing.txt", "--force")
        self.assertEqual(rc, 1, out)
        self.assertIn("--reach missing.txt: no such file", out)

    def test_not_for_a_narrow_batch_or_one_of_maps(self):
        h = Home(self)
        changes(h, 7)
        self.assertNotIn(NOTE, h.worker("batch", "tasks.jsonl")[2])
        h.file("tasks.jsonl", "".join(json.dumps({"id": f"m{i}", "prompt": "[[say:a map]]", "out": f"m{i}.md"}) + "\n"
                                      for i in range(10)))
        self.assertNotIn(NOTE, h.worker("batch", "tasks.jsonl")[2])


class WholeFile(unittest.TestCase):
    """1.8.0's: a task that asks for a large code file whole again is told so before anything is sent (21-11-2026,
    saturn-fullset #119: main.c came back empty after ~40K tokens of reasoning, layer2d.py cut at line 957)"""
    NOTE = "whole-file drafts"

    def test_a_large_file_asked_for_whole_is_told_with_the_split(self):
        h = Home(self)
        h.file("layer2d.py", "x = 1\n" * 1200)
        rc, out, err = h.worker("ask", "-f", "layer2d.py", "-o", "layer2d.new.py", "--max-tokens", "1000",
                                "-p", "[[say:x = 1]]", "--brief", "")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"{self.NOTE}: layer2d.new.py from layer2d.py, 1200 lines", err)
        self.assertIn("more than its max_tokens 1000", err)
        self.assertIn("FILE::name", err)

    def test_not_for_a_range_a_small_file_a_document_or_another_file(self):
        h = Home(self)
        h.file("big.py", "x = 1\n" * 1200)
        h.file("small.py", "x = 1\n" * 50)
        h.file("tasks.jsonl", "".join(json.dumps(t) + "\n" for t in [
            {"id": "range", "prompt": "[[say:x]]", "files": ["big.py:1-100"], "out": "big.py.part"},
            {"id": "small", "prompt": "[[say:x]]", "files": ["small.py"], "out": "small_new.py"},
            {"id": "map", "prompt": "[[say:x]]", "files": ["big.py"], "out": "big.md"},
            {"id": "test", "prompt": "[[say:x]]", "files": ["big.py"], "out": "test_big.py"}]))
        rc, out, err = h.worker("batch", "tasks.jsonl", "--brief", "")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn(self.NOTE, err)
        h.file("tasks.jsonl", json.dumps({"id": "whole", "prompt": "[[say:x]]", "files": ["big.py"],
                                          "out": "out/big.py"}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--brief", "")
        self.assertIn(f"{self.NOTE}: whole: big.py from big.py, 1200 lines", err)


if __name__ == "__main__":
    unittest.main()
