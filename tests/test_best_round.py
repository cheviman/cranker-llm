"""1.8.0's tests: a task with a check that ends failing leaves in out the best of its rounds, not the last
(21-11-2026: saturn-fullset's rules.py had a second round that was almost right, and the third came back empty and was
written over it), and every round's draft stays beside it.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import unittest

from test_worker import PY, Home

WORDS = ["alpha", "beta", "gamma", "delta"]


def check_words(h):
    """A check that wants the four words: one error line for each one missing"""
    script = h.file("words.py", "import sys\ntext = open(sys.argv[1], encoding='utf-8').read().split()\n"
                                f"missing = [w for w in {WORDS!r} if w not in text]\n"
                                "for w in missing:\n    print('error: missing', w)\n"
                                "sys.exit(1 if missing else 0)\n")
    return f'"{PY}" "{script}" "{{out}}"'


def batch(h, rounds):
    h.file("tasks.jsonl", json.dumps({"id": "rules", "prompt": f"[[rounds:{'||'.join(rounds)}]]",
                                      "out": "rules.txt"}))
    return h.worker("batch", "tasks.jsonl", "--check", check_words(h))


class BestRound(unittest.TestCase):
    def test_an_empty_last_round_does_not_take_the_place_of_a_near_miss(self):
        h = Home(self)
        rc, out, err = batch(h, ["alpha", "alpha beta gamma", ""])
        self.assertEqual(rc, 2, out + err)
        self.assertEqual((h.work / "rules.txt").read_text(encoding="utf-8"), "alpha beta gamma")
        self.assertIn("out holds round 2's draft, the best of 3 (the last came back empty)", out)
        self.assertIn("each round's draft beside it (rules.rN.txt)", out)
        self.assertIn("BAD rules", out)  # not CUT: what is in out is whole, and fails its check
        drafts = [(h.work / f"rules.r{n}.txt").read_text(encoding="utf-8") for n in (1, 2, 3)]
        self.assertEqual(drafts, ["alpha", "alpha beta gamma", ""])
        note = (h.work / "rules.txt.check.txt").read_text(encoding="utf-8")
        self.assertIn("out holds round 2's draft", note)
        self.assertIn("missing delta", note)  # its own check's output, not the empty one's
        self.assertNotIn("missing alpha", note)
        self.assertEqual(h.log("quality.jsonl")[-1].get("kept_round"), 2)

    def test_a_cut_last_round_does_not_either(self):
        h = Home(self)
        rc, out, err = batch(h, ["alpha", "alpha beta", "!cut"])
        self.assertEqual(rc, 2, out + err)
        self.assertEqual((h.work / "rules.txt").read_text(encoding="utf-8"), "alpha beta")
        self.assertIn("out holds round 2's draft, the best of 3 (the last came back empty)", out)

    def test_a_better_last_round_stays(self):
        h = Home(self)
        rc, out, err = batch(h, ["alpha", "alpha beta", "alpha beta gamma"])
        self.assertEqual(rc, 2, out + err)
        self.assertEqual((h.work / "rules.txt").read_text(encoding="utf-8"), "alpha beta gamma")
        self.assertNotIn("out holds round", out)
        self.assertIn("each round's draft beside it (rules.rN.txt)", out)
        self.assertIsNone(h.log("quality.jsonl")[-1].get("kept_round"))

    def test_the_fewest_errors_win_and_the_fix_starts_from_them(self):
        h = Home(self)
        rc, out, err = batch(h, ["alpha beta gamma", "alpha", "beta"])
        self.assertEqual(rc, 2, out + err)
        self.assertEqual((h.work / "rules.txt").read_text(encoding="utf-8"), "alpha beta gamma")
        self.assertIn("1 line of errors in its check against 3", out)
        third = h.requests()[-1]["messages"]  # its fix: from round 1's draft, the best so far, not round 2's
        self.assertEqual(third[-2]["content"], "alpha beta gamma")

    def test_a_task_that_passes_leaves_no_drafts(self):
        h = Home(self)
        rc, out, err = batch(h, ["alpha", "alpha beta gamma delta"])
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(sorted(p.name for p in h.work.glob("rules*")), ["rules.txt"])

    def test_nor_one_that_had_a_single_round(self):
        h = Home(self)
        h.file("tasks.jsonl", json.dumps({"id": "rules", "prompt": "[[say:alpha]]", "out": "rules.txt", "retries": 0}))
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", check_words(h))
        self.assertEqual(rc, 2, out + err)
        self.assertNotIn("each round's draft beside it (", out)
        self.assertEqual(sorted(p.name for p in h.work.glob("rules.r*")), [])


if __name__ == "__main__":
    unittest.main()
