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


if __name__ == "__main__":
    unittest.main()
