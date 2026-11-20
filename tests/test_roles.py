"""remote.roles' tests, against the same stand-ins as test_worker.py: one fake llama-server here, one elsewhere.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import unittest

from test_worker import Home, check_for


class Roles(unittest.TestCase):
    """remote.roles: a role's model, its own fields, and its escalation, sent to the server elsewhere."""
    KEY = {"CRANKER_REMOTE_KEY": "its-key"}

    def home(self, serve=True, **remote):
        h = Home(self, serve=serve)
        h.save(remote={"url": h.elsewhere(), "model": "base-model", "roots": [str(h.work)],
                       "plain": {"reasoning_effort": "none"}, **remote})
        return h

    def test_a_role_sends_ask_to_its_model_without_remote(self):
        h = self.home(roles={"review": "big-model"})
        rc, out, err = h.worker("ask", "--role", "review", "-o", "r.md", "-p", "[[say:found]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "r.md").read_text(encoding="utf-8"), "found")
        self.assertEqual(len(h.requests(h.away)), 1)
        self.assertEqual(h.requests(h.away)[0]["model"], "big-model")
        self.assertEqual(len(h.requests()), 0)
        self.assertIn("big-model as review", out)
        last = h.log("usage.jsonl")[-1]
        self.assertEqual(last["model"], "big-model")
        self.assertEqual(last["role"], "review")
        self.assertEqual(last["remote"], "127.0.0.1")

    def test_a_role_sets_its_own_fields_there(self):
        h = self.home(roles={"draft": {"model": "draft-model", "plain": {"reasoning_effort": "low"}}})
        rc, out, err = h.worker("ask", "--role", "draft", "-o", "d.md", "-p", "[[say:ok]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(len(h.requests(h.away)), 1)
        req = h.requests(h.away)[0]
        self.assertEqual(req["model"], "draft-model")
        self.assertEqual(req["reasoning_effort"], "low")

    def test_a_local_role_stays_here(self):
        h = self.home(roles={"volume": "local"})
        rc, out, err = h.worker("ask", "--role", "volume", "-o", "v.md", "-p", "[[say:here]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("local as volume", out)
        self.assertEqual(len(h.requests()), 1)
        self.assertEqual(len(h.requests(h.away)), 0)

    def test_local_and_a_role_elsewhere_are_refused(self):
        h = self.home(roles={"review": "big-model"})
        rc, out, err = h.worker("ask", "--local", "--role", "review", "-p", "x", **self.KEY)
        self.assertEqual(rc, 1, out)
        self.assertIn("is a model elsewhere", out)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 0))

    def test_an_unknown_role_is_refused(self):
        h = self.home(roles={"review": "big-model"})
        rc, out, err = h.worker("ask", "--role", "nope", "-p", "x", **self.KEY)
        self.assertEqual(rc, 1, out)
        self.assertIn("no role 'nope' in remote.roles", out)
        self.assertEqual(len(h.requests()), 0)
        self.assertEqual(len(h.requests(h.away)), 0)

    def test_a_file_outside_the_roots_does_not_leave_for_a_role(self):
        h = self.home(roles={"review": "big-model"})
        outside = h.dir / "outside.c"
        outside.write_text("int x;\n", encoding="utf-8")
        rc, out, err = h.worker("ask", "--role", "review", "-o", "o.md", "-f", f"{outside}:1-1", "-p", "x",
                                **self.KEY)
        self.assertEqual(rc, 1, out)
        self.assertIn("is not under remote.roots", out)
        self.assertEqual(len(h.requests(h.away)), 0)

    def test_a_batch_sends_each_task_where_its_role_says(self):
        h = self.home(roles={"review": "big-model"})
        tasks = h.file("tasks.jsonl",
                       json.dumps({"id": "a", "prompt": "[[say:A]]", "out": "a.md", "role": "review"}) + "\n" +
                       json.dumps({"id": "b", "prompt": "[[say:B]]", "out": "b.md", "role": "review"}) + "\n" +
                       json.dumps({"id": "c", "prompt": "[[say:C]]", "out": "c.md"}) + "\n")
        rc, out, err = h.worker("batch", tasks, **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("roles: 2 as review (big-model)", out)
        self.assertEqual(len(h.requests(h.away)), 2)
        self.assertEqual(len(h.requests()), 1)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "A")
        self.assertEqual((h.work / "b.md").read_text(encoding="utf-8"), "B")
        self.assertEqual((h.work / "c.md").read_text(encoding="utf-8"), "C")

    def test_a_failed_check_escalates_its_fix_to_the_role_above(self):
        h = self.home(roles={"big": "big-model"})
        rc, out, err = h.worker("ask", "-o", "x.txt", "--check", check_for(h, "right"), "--escalate", "big",
                                "-p", "[[say:wrong]] [[then:right]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("check passed after 1 fix", out)
        self.assertIn("fixes from fix 1 on by big-model as big (escalated)", out)
        self.assertEqual(len(h.requests()), 1)
        self.assertEqual(len(h.requests(h.away)), 1)
        self.assertEqual(h.requests(h.away)[0]["model"], "big-model")
        self.assertEqual((h.work / "x.txt").read_text(encoding="utf-8"), "right")
        self.assertEqual(h.log("quality.jsonl")[-1]["escalated"], "big")

    def test_the_configs_escalate_needs_no_flag(self):
        # with remote.escalate alone, the task had no "escalate" of its own and the escalation raised a KeyError
        h = self.home(roles={"big": "big-model"}, escalate="big")
        rc, out, err = h.worker("ask", "-o", "x.txt", "--check", check_for(h, "right"),
                                "-p", "[[say:wrong]] [[then:right]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("fixes from fix 1 on by big-model as big (escalated)", out)
        self.assertEqual(h.log("quality.jsonl")[-1]["escalated"], "big")
        rc, out, err = h.worker("ask", "-o", "y.txt", "--check", check_for(h, "right"), "--escalate", "",
                                "-p", "[[say:wrong]] [[then:right]]", **self.KEY)  # "" turns it off for this run
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("escalated", out)
        self.assertEqual(len(h.requests(h.away)), 1)

    def test_usage_prices_each_model_apart(self):
        h = self.home(price=[1, 2], roles={"review": {"model": "big-model", "price": [10, 20]}})
        rc, out, err = h.worker("ask", "--remote", "-p", "[[say:a]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("ask", "--role", "review", "-p", "[[say:b]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        rc, out, err = h.worker("usage", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("By model:", out)
        self.assertIn("base-model: 1 requests", out)
        self.assertIn("big-model (as review): 1 requests", out)
        self.assertIn("at $10 / $20", out)


if __name__ == "__main__":
    unittest.main()
