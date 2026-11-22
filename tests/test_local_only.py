"""--local keeps every request of a run on the local worker, against the same stand-ins as test_worker.py: one fake
llama-server here, one elsewhere. On 22-11-2026 `batch --local` sent the check fixes of 4 of 81 reviews to the
drafter's own model elsewhere (remote.escalate), which is what the review by another family was there to avoid.

    py -m unittest discover -s tests -v
"""
import json
import unittest

from test_worker import Home, SCRIPTS, check_for

HOLD = ("import sys, time; sys.path.insert(0, sys.argv[1]); import worker\n"
        "lock = worker.agent_turn(1)\nprint('held', flush=True)\ntime.sleep(60)")


class LocalOnly(unittest.TestCase):
    """--local: first answers, fixes, spill and escalation all stay here; a line's own way out is refused."""
    KEY = {"CRANKER_REMOTE_KEY": "its-key"}

    def home(self, **config):
        h = Home(self, **config)
        # the user's config of that day: spill on, roles elsewhere, remote.escalate after 1 failed answer
        h.save(remote={"url": h.elsewhere(), "model": "base-model", "roots": [str(h.work)], "spill": True,
                       "plain": {"reasoning_effort": "none"}, "roles": {"draft": "draft-model", "here": "local"},
                       "escalate": "draft", "escalate_after": 1})
        return h

    def review_tasks(self, h, n=2, **more):
        check = check_for(h, "right")
        return h.file("review.jsonl", "".join(
            json.dumps({"id": f"rev{i}", "prompt": "[[say:wrong]] [[then:right]]", "out": f"rev{i}.txt",
                        "check": check, "effort": "low", **more}) + "\n" for i in range(n)))

    def test_ask_local_keeps_the_fixes_here(self):
        h = self.home()
        rc, out, err = h.worker("ask", "--local", "-o", "x.txt", "--check", check_for(h, "right"),
                                "-p", "[[say:wrong]] [[then:right]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("check passed after 1 fix", out)
        self.assertNotIn("escalated", out)
        self.assertNotIn("remote.spill", out)
        self.assertIn("--local: remote.escalate (draft) is not used", out + err)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (2, 0))

    def test_batch_local_keeps_the_fixes_here(self):
        h = self.home()
        rc, out, err = h.worker("batch", self.review_tasks(h), "--local", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("done: 2 ok", out)
        self.assertNotIn("escalated", out)
        self.assertNotIn("remote.spill", out)
        self.assertIn("--local: remote.escalate (draft) is not used", out)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (4, 0))
        for i in range(2):
            self.assertEqual((h.work / f"rev{i}.txt").read_text(encoding="utf-8"), "right")

    def test_a_lines_escalate_to_a_local_role_is_not_refused(self):
        h = self.home()
        rc, out, err = h.worker("batch", self.review_tasks(h, escalate="here"), "--local", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("done: 2 ok", out)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (4, 0))

    def test_a_lines_own_escalate_elsewhere_is_refused(self):
        h = self.home()
        rc, out, err = h.worker("batch", self.review_tasks(h, escalate="draft"), "--local", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("--local, and its escalate draft is a model elsewhere", out + err)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 0))
        rc, out, err = h.worker("batch", self.review_tasks(h), "--local", "--escalate", "draft", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("--local, and its escalate draft is a model elsewhere", out + err)
        rc, out, err = h.worker("ask", "--local", "--escalate", "draft", "-o", "x.txt", "--check",
                                check_for(h, "right"), "-p", "[[say:wrong]]", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("--local, and its escalate draft is a model elsewhere", out + err)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 0))

    def test_a_lines_own_role_elsewhere_is_refused(self):
        h = self.home()
        rc, out, err = h.worker("batch", self.review_tasks(h, role="draft"), "--local", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("--local, and its role draft is a model elsewhere", out + err)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 0))

    def test_an_escalation_is_not_called_a_spill(self):
        # without --local the fix escalates, as the config says: its line says so, and no longer "(remote.spill)"
        h = self.home()
        rc, out, err = h.worker("batch", self.review_tasks(h, n=1), **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("fixes from fix 1 on by draft-model as draft (escalated)", out)
        line = next(l for l in out.splitlines() if l.startswith("ok  rev0"))
        self.assertNotIn("remote.spill", line)
        self.assertIn("; 0 of them on 127.0.0.1 (remote.spill)", out)  # the first answer ran here
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (1, 1))

    def test_local_and_remote_together_are_refused(self):
        h = self.home()
        for cmd in (("ask", "-p", "x"), ("batch", str(self.review_tasks(h)))):
            rc, out, err = h.worker(*cmd, "--local", "--remote", **self.KEY)
            self.assertEqual(rc, 1, out + err)
            self.assertIn("--local and --remote", out)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 0))

    def test_a_full_local_worker_waits_instead_of_spilling(self):
        # without --local the same batch goes there (test_spill); with it every task waits for room here, and
        # with no room before context_wait it is not run: nothing leaves, neither the answer nor its fixes
        h = self.home(context_wait=0)
        holder = h.start("-c", HOLD, SCRIPTS)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        rc, out, err = h.worker("batch", self.review_tasks(h), "--local", **self.KEY)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("WAIT ", out)
        self.assertNotIn("remote.spill", out)
        rc, out, err = h.worker("ask", "--local", "-o", "x.txt", "--check", check_for(h, "right"),
                                "-p", "[[say:wrong]] [[then:right]]", **self.KEY)
        self.assertEqual(rc, 2, out + err)
        self.assertEqual(len(h.requests(h.away)), 0)
        self.assertEqual(len(h.requests()), 0)


if __name__ == "__main__":
    unittest.main()
