"""A request the server elsewhere refuses (HTTP 400), against the same stand-ins as test_worker.py: one fake
llama-server here, one elsewhere. On 27-11-2026 two tasks ended in FAIL that way, a role's request and an escalated
fix, with the local worker there to answer them.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import unittest

from test_worker import Home, SCRIPTS, check_for

REFUSED = "[[fail-there:400:provider rejected the request: invalid request error]]"


class Refused(unittest.TestCase):
    """HTTP 400 from the server elsewhere: the request is the local worker's, the task does not fail."""
    KEY = {"CRANKER_REMOTE_KEY": "its-key"}

    def home(self, home=None, **remote):
        h = home or Home(self)
        h.save(remote={"url": h.elsewhere(), "model": "base-model", "roots": [str(h.work)],
                       "plain": {"reasoning_effort": "none"}, **remote})
        return h

    def test_a_refused_escalated_fix_is_done_here(self):
        h = self.home(roles={"big": "big-model"}, escalate="big")
        rc, out, err = h.worker("ask", "-o", "x.txt", "--check", check_for(h, "right"),
                                "-p", f"[[say:wrong]] [[then:right]] {REFUSED}", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "x.txt").read_text(encoding="utf-8"), "right")
        self.assertIn("check passed after 1 fix", out)
        self.assertEqual(len(h.requests(h.away)), 1)  # the fix, refused there
        self.assertEqual(len(h.requests()), 2)  # the first answer, and the fix again
        self.assertIn("refused", err)
        self.assertIn("HTTP 400", err)
        # the role above did none of it: neither the task's line nor quality.jsonl says it was escalated
        self.assertNotIn("(escalated)", out)
        self.assertIn("127.0.0.1 refused a request, answered by the local worker", out)
        self.assertNotIn("escalated", h.log("quality.jsonl")[-1])
        self.assertNotIn("remote", h.log("usage.jsonl")[-1])

    def test_a_refused_fix_stops_the_escalation_for_the_next_ones(self):
        h = self.home(roles={"big": "big-model"}, escalate="big")
        rc, out, err = h.worker("ask", "-o", "x.txt", "--check", check_for(h, "right"), "--retries", "3",
                                "-p", f"[[rounds:wrong||worse||right]] {REFUSED}", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "x.txt").read_text(encoding="utf-8"), "right")
        self.assertEqual(len(h.requests(h.away)), 1)  # asked once: the second fix does not go there to be refused
        self.assertEqual(len(h.requests()), 3)

    def test_a_refused_role_request_is_answered_here(self):
        h = self.home(roles={"review": "big-model"})
        rc, out, err = h.worker("ask", "--role", "review", "-o", "r.md", "-p", f"{REFUSED} [[say:here]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "r.md").read_text(encoding="utf-8"), "here")
        self.assertEqual(len(h.requests(h.away)), 1)
        self.assertEqual(len(h.requests()), 1)
        self.assertIn("refused", err)
        self.assertIn("127.0.0.1 refused a request, answered by the local worker", out)
        self.assertNotIn("big-model as review", out)

    def test_a_batch_keeps_its_other_tasks_there(self):
        h = self.home(roles={"review": "big-model"})
        tasks = h.file("tasks.jsonl",
                       json.dumps({"id": "a", "prompt": f"{REFUSED} [[say:A]]", "out": "a.md", "role": "review"}) + "\n" +
                       json.dumps({"id": "b", "prompt": "[[say:B]]", "out": "b.md", "role": "review"}) + "\n")
        rc, out, err = h.worker("batch", tasks, **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("FAIL", out)
        self.assertIn("done: 2 ok", out)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "A")
        self.assertEqual((h.work / "b.md").read_text(encoding="utf-8"), "B")
        self.assertEqual(len(h.requests(h.away)), 2)
        self.assertEqual(len(h.requests()), 1)

    def test_remote_asked_for_by_name_still_fails(self):
        # --remote is the user's word for that server: its refusal is the answer, not a reason to run it here
        h = self.home()
        rc, out, err = h.worker("ask", "--remote", "-o", "a.md", "-p", f"{REFUSED} [[say:here]]", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("HTTP 400", out)
        self.assertEqual(len(h.requests()), 0)

    def test_with_the_local_worker_not_there_it_says_both(self):
        h = self.home(Home(self, serve=False), roles={"review": "big-model"})
        rc, out, err = h.worker("ask", "--role", "review", "-o", "r.md", "-p", f"{REFUSED} [[say:here]]", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("HTTP 400", out)
        self.assertIn("the local worker", out)

    def test_a_refused_spill_waits_for_the_local_worker(self):
        h = Home(self, context_wait=30)
        h.save(remote={"url": h.elsewhere(), "model": "qwen-there", "roots": [str(h.work)], "spill": True})
        holder = h.start("-c", "import sys, time; sys.path.insert(0, sys.argv[1]); import worker\n"
                         "lock = worker.agent_turn(1)\nprint('held', flush=True)\ntime.sleep(3)", SCRIPTS)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", f"{REFUSED} [[say:from here]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "from here")
        self.assertEqual(len(h.requests(h.away)), 1)
        self.assertEqual(len(h.requests()), 1)

    def test_a_refused_spill_over_the_local_context_says_both(self):
        # nowhere to run it: over the local context, and the server elsewhere refused it. The line says the two
        h = Home(self, ctx=6000)
        h.save(remote={"url": h.elsewhere(), "model": "qwen-there", "roots": [str(h.work)], "spill": True})
        rc, out, err = h.worker("ask", "-o", "a.md", "--max-tokens", "7000", "-p", f"{REFUSED} [[say:x]]", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("over the context", out)
        self.assertIn("HTTP 400", out)
        self.assertEqual(len(h.requests()), 0)


if __name__ == "__main__":
    unittest.main()
