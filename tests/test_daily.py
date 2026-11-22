"""remote.prices, remote.daily, and the fall back to the local worker, against the same stand-ins as test_worker.py:
one fake llama-server here, one elsewhere.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import time
import unittest

from test_worker import Home


class Daily(unittest.TestCase):
    """remote.prices, remote.daily, and the fall back to the local worker."""
    KEY = {"CRANKER_REMOTE_KEY": "its-key"}

    def home(self, **remote):
        h = Home(self, context_wait=0)
        h.save(remote={"url": h.elsewhere(), "model": "base-model", "roots": [str(h.work)],
                       "plain": {"reasoning_effort": "none"}, **remote})
        return h

    def spent(self, h, usd, days_ago=0):
        """A line in usage.jsonl as a request elsewhere that cost usd, today or days_ago"""
        logs = h.dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - days_ago * 86400))
        with open(logs / "usage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": ts, "cmd": "ask", "model": "big-model", "in": 1, "out": 1,
                                "remote": "127.0.0.1", "usd": usd}) + "\n")

    def test_a_request_there_is_priced_with_its_cached_part(self):
        h = self.home(roles={"review": "big-model"},
                      prices={"big-model": [1000000, 2000000, 100000]})
        rc, out, err = h.worker("ask", "--role", "review", "-o", "r.md",
                                "-p", "[[cached:2]] [[say:found it]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        last = h.log("usage.jsonl")[-1]
        self.assertEqual(last["cached"], 2)
        expected = round((last["in"] - 2) * 1 + 2 * 0.1 + last["out"] * 2, 6)
        self.assertAlmostEqual(last["usd"], expected, places=4)

    def test_a_spent_day_sends_a_role_here(self):
        h = self.home(roles={"review": "big-model"}, daily=7)
        self.spent(h, 7.5)
        rc, out, err = h.worker("ask", "--role", "review", "-o", "r.md", "-p", "[[say:here]]",
                                **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "r.md").read_text(encoding="utf-8"), "here")
        self.assertEqual(len(h.requests(h.away)), 0)
        self.assertEqual(len(h.requests()), 1)
        self.assertIn("takes no more requests today", err)
        self.assertIn("remote.daily", err)

    def test_another_days_spending_does_not_count(self):
        h = self.home(roles={"review": "big-model"}, daily=7)
        self.spent(h, 100, days_ago=2)
        rc, out, err = h.worker("ask", "--role", "review", "-o", "r.md", "-p", "[[say:here]]",
                                **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(len(h.requests(h.away)), 1)
        self.assertEqual(len(h.requests()), 0)

    def test_remote_is_not_sent_once_the_day_is_spent(self):
        h = self.home(daily=7)
        self.spent(h, 7.5)
        rc, out, err = h.worker("ask", "--remote", "-p", "[[say:x]]", **self.KEY)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("not sent to 127.0.0.1: remote.daily", out + err)
        self.assertEqual(len(h.requests(h.away)), 0)

    def test_a_spent_quota_there_sends_the_role_here(self):
        h = self.home(roles={"review": "big-model"})
        rc, out, err = h.worker("ask", "--role", "review", "-o", "r.md",
                                "-p", "[[fail-there:429:free_limit_reached for today]] [[say:here]]",
                                **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "r.md").read_text(encoding="utf-8"), "here")
        self.assertEqual(len(h.requests(h.away)), 1)
        self.assertEqual(len(h.requests()), 1)
        self.assertIn("takes no more requests today", err)

    def test_status_says_what_is_left_today(self):
        h = self.home(daily=7)
        self.spent(h, 2.5)
        self.spent(h, 100, days_ago=2)
        rc, out, err = h.worker("status", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("today $2.50 of $7.00 (remote.daily", out)

    def test_a_batch_says_what_it_costs_before_it_runs(self):
        h = self.home(roles={"review": "big-model"}, prices={"big-model": [1000000, 0]})
        tasks = h.file("tasks.jsonl",
                       json.dumps({"id": "a", "prompt": "[[say:x]]", "out": "a.md", "role": "review"}) + "\n" +
                       json.dumps({"id": "b", "prompt": "[[say:x]]", "out": "b.md", "role": "review"}) + "\n")
        rc, out, err = h.worker("batch", tasks, **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("elsewhere: 2 to big-model (~", err)
        self.assertIn("at least $", err)


if __name__ == "__main__":
    unittest.main()