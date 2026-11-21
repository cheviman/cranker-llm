"""1.7.0's tests: "variants", one task tried several ways side by side, and the one that passes its check with the
best score kept (20/21-11-2026: three tries of a task with other roles and efforts, the best one picked by hand).

    py -m unittest discover -s tests -v

Standard library only.
"""
import importlib.util
import json
import sys
import unittest

from test_worker import Home, SCRIPTS, WORKER

_spec = importlib.util.spec_from_file_location("worker", WORKER)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
_spec = importlib.util.spec_from_file_location("local_share", SCRIPTS / "local_share.py")
local_share = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local_share)


def scored_check(h):
    """Passes an output that starts with "good"; "good N" also prints "score: N" """
    script = h.file("scored.py", "import sys\ntext = open(sys.argv[1], encoding='utf-8').read().split()\n"
                                 "if not text or text[0] != 'good':\n    print('not good:', text)\n    sys.exit(1)\n"
                                 "print('measured')\n"
                                 "if len(text) > 1:\n    print('score:', text[1])\n")
    return f'"{sys.executable}" "{script}" "{{out}}"'


def task(h, variants, **more):
    h.file("tasks.jsonl", json.dumps({"id": "t", "prompt": "[[say:unused]]", "out": "t.txt", "retries": 0,
                                      "variants": variants, **more}) + "\n")


class Variants(unittest.TestCase):
    def test_the_one_that_passes_with_the_best_score_is_kept(self):
        h = Home(self)
        task(h, [{"prompt": "[[say:bad]]"}, {"prompt": "[[say:good 7]]"}, {"prompt": "[[say:good 9]]", "effort": "low"}])
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", scored_check(h))
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "t.txt").read_text(encoding="utf-8"), "good 9")
        self.assertIn("variant 3 of 3 kept", out)
        self.assertIn("v1 (its own prompt): CHECK FAILED", out)
        self.assertIn("v2 (its own prompt): passed, score 7", out)
        self.assertIn("v3 (effort low, its own prompt): passed, score 9", out)
        self.assertEqual([(h.work / f"t.v{k}.txt").read_text(encoding="utf-8") for k in (1, 2, 3)],
                         ["bad", "good 7", "good 9"])
        self.assertEqual(len(h.requests()), 3)
        quality = h.log("quality.jsonl")
        self.assertEqual(len(quality), 1)  # one task, not three
        self.assertEqual((quality[0]["chosen"], quality[0]["variants"], quality[0]["result"]), (3, 3, "pass"))
        usage = h.log("usage.jsonl")
        self.assertEqual(sorted(r.get("variant") for r in usage if r["cmd"] == "batch"), [1, 2, 3])
        self.assertEqual(len({r["work"] for r in usage}), 1)  # one piece of work
        self.assertEqual(usage[-1]["cmd"], "variants")
        self.assertEqual(usage[-1]["chosen"], 3)
        self.assertFalse((h.work / "t.txt.running.json").exists())

    def test_with_no_score_the_first_that_passes(self):
        h = Home(self)
        task(h, [{"prompt": "[[say:bad]]"}, {"prompt": "[[say:good]]", "role": None}, {"prompt": "[[say:good]]"}])
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", scored_check(h))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("variant 2 of 3 kept", out)

    def test_none_passes(self):
        h = Home(self)
        task(h, [{"prompt": "[[say:bad]]"}, {"prompt": "[[say:worse]]"}])
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", scored_check(h))
        self.assertEqual(rc, 2, out + err)
        self.assertIn("BAD t: no variant passed its check", out)
        self.assertEqual((h.work / "t.txt").read_text(encoding="utf-8"), "bad")  # the first one's, to look at
        self.assertIn("no variant passed", (h.work / "t.txt.check.txt").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((h.work / "t.txt.failed.json").read_text(encoding="utf-8"))["why"], "check")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", scored_check(h))
        self.assertIn("failed before", out)  # unchanged: not run again
        self.assertEqual(len(h.requests()), 2)
        self.assertEqual(h.log("quality.jsonl")[-1]["result"], "fail")

    def test_variants_are_checked_before_anything_runs(self):
        h = Home(self)
        task(h, [{"prompt": "[[say:good]]"}])
        rc, out, err = h.worker("batch", "tasks.jsonl")
        self.assertEqual(rc, 1, out)
        self.assertIn("\"variants\" needs a check", out)
        task(h, [{"out": "elsewhere.txt"}])
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", scored_check(h))
        self.assertEqual(rc, 1, out)
        self.assertIn("a variant cannot set out", out)
        task(h, "draft")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", scored_check(h))
        self.assertEqual(rc, 1, out)
        self.assertIn("is a list of objects", out)
        self.assertEqual(h.requests(), [])

    def test_usage_counts_the_kept_one_as_the_work_and_the_others_as_spent(self):
        h = Home(self)
        task(h, [{"prompt": "[[say:good 1]]"}, {"prompt": "[[say:good 5]]"}])
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", scored_check(h))
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("usage")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("1 pieces, 1 right at the first try", out)
        self.assertIn("1 were variants not kept", out)
        self.assertNotIn("variants  ", out)  # the line that says which was kept is no request in the tables


class PieceRuns(unittest.TestCase):
    """The same accounting in worker.py (usage) and in local_share.py (the hooks' scale)"""
    RUNS = [{"work": "w", "cmd": "batch", "variant": 1, "answer": 5, "in": 10, "out": 5},
            {"work": "w", "cmd": "batch", "variant": 2, "answer": 5, "in": 10, "out": 5, "check": "fail"},
            {"work": "w", "cmd": "batch", "variant": 2, "answer": 5, "in": 10, "out": 5},
            {"work": "w", "cmd": "variants", "chosen": 2, "answer": 0, "in": 0, "out": 0}]

    def test_both_count_the_kept_variants_runs_as_the_tries(self):
        for module in (worker, local_share):
            rows = [dict(r) for r in self.RUNS]
            tries, beside = module.piece_runs(rows)
            self.assertEqual([r.get("variant") for r in tries], [2, 2], module.__name__)
            self.assertEqual([r.get("variant") for r in beside], [1], module.__name__)

    def test_none_kept_counts_them_all(self):
        rows = [dict(r) for r in self.RUNS[:3]] + [{"work": "w", "cmd": "variants", "chosen": None}]
        tries, beside = worker.piece_runs(rows)
        self.assertEqual((len(tries), beside), (3, []))

    def test_mark_tries(self):
        rows = [dict(r) for r in self.RUNS]
        worker.mark_tries(rows)
        self.assertEqual(rows[0]["_why"], worker.NOT_KEPT)
        self.assertEqual((rows[2]["_tries"], rows[2]["_why"]), (2, None))  # right by its second try: effective


if __name__ == "__main__":
    unittest.main()
