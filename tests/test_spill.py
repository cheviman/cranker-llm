"""remote.spill's tests, against the same stand-ins as test_worker.py: one fake llama-server here, one elsewhere."""
import json
import time
import unittest

from test_worker import Home, SCRIPTS


class Spill(unittest.TestCase):
    """remote.spill: a full local worker sends the request to the remote server by itself."""
    KEY = {"CRANKER_REMOTE_KEY": "its-key"}

    def home(self, serve=True, **config):
        h = Home(self, serve=serve, context_wait=0, **config)
        h.save(remote={"url": h.elsewhere(), "model": "qwen-there", "roots": [str(h.work)], "spill": True})
        return h

    def test_a_full_local_worker_sends_ask_there(self):
        h = self.home()
        holder = h.start("-c", "import sys, time; sys.path.insert(0, sys.argv[1]); import worker\n"
                         "lock = worker.agent_turn(1)\nprint('held', flush=True)\ntime.sleep(60)", SCRIPTS)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:from afar]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "from afar")
        self.assertIn("on 127.0.0.1 (remote.spill", out)
        self.assertEqual(len(h.requests(h.away)), 1)
        self.assertEqual(len(h.requests()), 0)
        self.assertEqual(h.log("usage.jsonl")[-1]["remote"], "127.0.0.1")

    def test_with_room_it_stays_here(self):
        h = self.home()
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:from here]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(len(h.requests()), 1)
        self.assertEqual(len(h.requests(h.away)), 0)

    def test_a_file_outside_the_roots_and_local_stay_here(self):
        h = self.home()
        holder = h.start("-c", "import sys, time; sys.path.insert(0, sys.argv[1]); import worker\n"
                         "lock = worker.agent_turn(1)\nprint('held', flush=True)\ntime.sleep(60)", SCRIPTS)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        outside = h.dir / "not-listed.c"
        outside.write_text("int x;\n", encoding="utf-8")
        self.assertEqual(h.worker("ask", "-f", str(outside), "-p", "x", **self.KEY)[0], 2)
        self.assertEqual(h.worker("ask", "--local", "-p", "x", **self.KEY)[0], 2)
        self.assertEqual(len(h.requests(h.away)), 0)

    def test_a_batch_runs_here_and_there_at_once(self):
        h = self.home(ctx=6000)
        tasks = h.file("tasks.jsonl", "".join(
            json.dumps({"id": str(i), "prompt": f"[[sleep:2]] [[say:answer {i}]]",
                        "out": f"{i}.md", "max_tokens": 2000}) + "\n" for i in range(4)))
        t0 = time.time()
        rc, out, err = h.worker("batch", tasks, **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("done: 4 ok", out)
        self.assertIn("4 of them may go to 127.0.0.1 as well (remote.spill)", out)
        for i in range(4):
            self.assertEqual((h.work / f"{i}.md").read_text(encoding="utf-8"), f"answer {i}")
        local, remote = h.requests(), h.requests(h.away)
        self.assertTrue(local)
        self.assertTrue(remote)
        self.assertEqual(len(local) + len(remote), 4)
        self.assertIn(f"; {len(remote)} of them on 127.0.0.1 (remote.spill)", out)
        self.assertLess(time.time() - t0, 7)

    def test_a_batch_shares_even_with_room_here(self):
        h = self.home()  # the whole context: every task would fit here
        tasks = h.file("tasks.jsonl", "".join(
            json.dumps({"id": str(i), "prompt": f"[[sleep:2]] [[say:answer {i}]]", "out": f"{i}.md"}) + "\n"
            for i in range(4)))
        rc, out, err = h.worker("batch", tasks, **self.KEY)
        self.assertEqual(rc, 0, out + err)
        local, remote = h.requests(), h.requests(h.away)
        self.assertEqual((len(local) + len(remote), len(remote)), (4, 2), out)  # remote.parallel is 2
        self.assertIn("; 2 of them on 127.0.0.1 (remote.spill)", out)
        rc, out, err = h.worker("batch", tasks, "--force", "--local", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(len(h.requests(h.away)), 2, "--local sent a task there")

    def test_a_failure_there_sends_it_back_here(self):
        h = Home(self, context_wait=30)
        h.save(remote={"url": h.elsewhere(), "model": "qwen-there", "roots": [str(h.work)], "spill": True})
        holder = h.start("-c", "import sys, time; sys.path.insert(0, sys.argv[1]); import worker\n"
                         "lock = worker.agent_turn(1)\nprint('held', flush=True)\ntime.sleep(3)", SCRIPTS)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:from here]]", CRANKER_REMOTE_KEY="another")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "from here")
        self.assertIn("remote.spill off for the rest of this run", err)
        self.assertEqual(len(h.requests()), 1)

    def test_the_local_server_not_running(self):
        h = self.home(serve=False)
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:from afar]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "from afar")
        outside = h.dir / "not-listed.c"
        outside.write_text("int x;\n", encoding="utf-8")
        tasks = h.file("tasks.jsonl",
                       json.dumps({"id": "0", "prompt": "x", "out": "0.md", "files": [str(outside)]}) + "\n" +
                       json.dumps({"id": "1", "prompt": "[[say:from afar]]", "out": "1.md"}) + "\n")
        rc, out, err = h.worker("batch", tasks, **self.KEY)
        self.assertEqual(rc, 2)
        self.assertIn("WAIT ", out)
        self.assertIn("the local worker is not running", out)
        self.assertFalse((h.work / "0.md").exists())
        self.assertEqual((h.work / "1.md").read_text(encoding="utf-8"), "from afar")
