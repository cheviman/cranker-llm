"""1.7.0's tests: --check-bad ("check_bad"), a known-bad output the check must fail before any request is sent.
On 20/21-11-2026 three checks that failed the null output still passed wrong work (OutRunners).

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import sys
import unittest

from test_worker import Home, check_for


def not_empty(h):
    """A weak check: anything that is not empty passes (it applies and compiles, and runs nothing)"""
    script = h.file("not_empty.py", "import sys\nsys.exit(0 if open(sys.argv[1], encoding='utf-8').read() else 1)\n")
    return f'"{sys.executable}" "{script}" "{{out}}"'


class CheckBad(unittest.TestCase):
    def test_a_check_that_passes_the_known_bad_output_stops_the_run(self):
        h = Home(self)
        h.file("bad.txt", "the patch that broke the game")
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", not_empty(h),
                                "--check-bad", "bad.txt")
        self.assertEqual(rc, 1, out)
        self.assertIn("the check passes the known-bad output bad.txt", out)
        self.assertIn("has to run the result", out)
        self.assertEqual(h.requests(), [])

    def test_a_check_that_fails_it_runs_and_its_proof_says_so(self):
        h = Home(self)
        h.file("bad.txt", "wrong")
        h.file("worse.txt", "worse")
        h.file("good.txt", "good")
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", check_for(h, "good"),
                                "--check-bad", "bad.txt", "--check-bad", "worse.txt", "--check-good", "good.txt")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(h.log("quality.jsonl")[-1]["proof"], "empty+bad+good")
        self.assertEqual(h.log("usage.jsonl")[-1]["proof"], "empty+bad+good")

    def test_every_bad_output_of_a_task_is_tried(self):
        h = Home(self)
        check = check_for(h, "good")
        h.file("b1.txt", "wrong")
        h.file("b2.txt", "good")  # the check passes this one: it is not a bad output to it
        h.file("tasks.jsonl", json.dumps({"id": "a", "prompt": "[[say:good]]", "out": "a.txt",
                                          "check_bad": ["b1.txt", "b2.txt"]}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", check)
        self.assertEqual(rc, 1, out)
        self.assertIn("task a: the check passes the known-bad output b2.txt", out)
        self.assertEqual(h.requests(), [])

    def test_a_batch_says_it_was_tried_with_it(self):
        h = Home(self)
        h.file("bad.txt", "wrong")
        tasks = [{"id": i, "prompt": "[[say:good]]", "out": f"{i}.txt"} for i in ("a", "b")]
        h.file("tasks.jsonl", "".join(json.dumps(t) + "\n" for t in tasks))
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", check_for(h, "good"), "--check-bad", "bad.txt")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("check, tried before any request: fails an empty output and fails the known-bad output "
                      "(2 tasks)", out)

    def test_it_needs_a_check(self):
        h = Home(self)
        h.file("bad.txt", "wrong")
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check-bad", "bad.txt")
        self.assertEqual(rc, 1, out)
        self.assertIn("--check-bad try a --check", out)

    def test_usage_counts_it_as_tried(self):
        h = Home(self)
        h.file("bad.txt", "wrong")
        rc, out, err = h.worker("ask", "-o", "x.txt", "-p", "[[say:good]]", "--check", check_for(h, "good"),
                                "--check-null", "none", "--check-bad", "bad.txt")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(h.log("quality.jsonl")[-1]["proof"], "bad")
        rc, out, err = h.worker("usage")
        self.assertIn("Of the 1 right: 1 passed a check tried before it ran", out)


if __name__ == "__main__":
    unittest.main()
