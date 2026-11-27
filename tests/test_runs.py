"""the runs log (runs.jsonl) and the day's line in `usage`, against the same stand-ins as test_worker.py:
one fake llama-server here.

    py -m unittest discover -s tests -v

Standard library only.
"""
import time
import unittest

from test_worker import Home


class Runs(unittest.TestCase):
    """the runs log (runs.jsonl) and the day's line in `usage`."""

    def test_an_ask_leaves_one_line(self):
        h = Home(self)
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:hello]]")
        self.assertEqual(rc, 0, out + err)
        runs = h.log("runs.jsonl")
        self.assertEqual(len(runs), 1)
        (run,) = runs
        self.assertEqual(run["cmd"], "ask")
        self.assertEqual(run["requests"], 1)
        (used,) = h.log("usage.jsonl")
        self.assertEqual(run["in"], used["in"])
        self.assertEqual(run["out"], used["out"])
        self.assertEqual(run["answer"], used["answer"])
        self.assertEqual(run["claude_usd"], {"Opus 5.5": round((used["in"] * 4.00 + used["out"] * 20.00) / 1e6, 6),
                                             "Fable 5.2": round((used["in"] * 8.00 + used["out"] * 40.00) / 1e6, 6)})
        self.assertEqual(run["prices"], {"Opus 5.5": [4.0, 20.0], "Fable 5.2": [8.0, 40.0]})
        self.assertEqual(run["rc"], 0)

    def test_a_batch_is_one_line(self):
        h = Home(self)
        tasks = h.file("tasks.jsonl",
                       '{"id": "a", "prompt": "[[say:one]]", "out": "a.md"}\n'
                       '{"id": "b", "prompt": "[[say:two]]", "out": "b.md"}\n'
                       '{"id": "c", "prompt": "[[say:three]]", "out": "c.md"}\n')
        rc, out, err = h.worker("batch", tasks)
        self.assertEqual(rc, 0, out + err)
        runs = h.log("runs.jsonl")
        self.assertEqual(len(runs), 1)
        (run,) = runs
        self.assertEqual(run["cmd"], "batch")
        self.assertEqual(run["requests"], 3)
        used = h.log("usage.jsonl")
        self.assertEqual(len(used), 3)
        self.assertEqual(run["in"], sum(u["in"] for u in used))
        self.assertEqual(run["out"], sum(u["out"] for u in used))

    def test_nothing_sent_no_line(self):
        h = Home(self)
        h.worker("usage")
        h.worker("status")
        h.worker("ask")
        self.assertEqual(h.log("runs.jsonl"), [])

    def test_session_and_tag_are_kept(self):
        h = Home(self)
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:hello]]",
                                CLAUDE_CODE_SESSION_ID="abc123", CRANKER_USAGE_TAG="bench")
        self.assertEqual(rc, 0, out + err)
        (run,) = h.log("runs.jsonl")
        self.assertEqual(run["session"], "abc123")
        self.assertEqual(run["tag"], "bench")

    def test_usage_shows_the_day(self):
        h = Home(self)
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:hello]]")
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("usage")
        self.assertEqual(rc, 0, out + err)
        lines = out.splitlines()
        header = next(i for i, l in enumerate(lines) if l.startswith("runs (runs.jsonl) at Claude price"))
        self.assertTrue(lines[header + 1].startswith(time.strftime("%Y-%m-%d")), lines[header + 1])


if __name__ == "__main__":
    unittest.main()
