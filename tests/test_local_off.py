"""The local worker turned off (`worker.py local off`, "local": false): fully remote, the GPU's memory free. Against
the same stand-ins as test_worker.py: one fake llama-server here, one elsewhere.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import unittest

from test_worker import Home, SCRIPTS


class LocalOff(unittest.TestCase):
    KEY = {"CRANKER_REMOTE_KEY": "its-key"}

    def home(self, serve=True, **remote):
        """A local server that still answers (the runs must not go to it) and a remote one; the local worker off"""
        h = Home(self, serve=serve)
        h.save(remote={"url": h.elsewhere(), "model": "base-model", "roots": [str(h.work)], **remote})
        rc, out, err = h.worker("local", "off")
        self.assertEqual(rc, 0, out + err)
        self.assertIs(json.loads((h.dir / "config.json").read_text(encoding="utf-8"))["local"], False)
        return h

    def test_ask_goes_to_the_remote_without_remote(self):
        h = self.home()
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:from afar]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "from afar")
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 1))
        self.assertEqual(h.log("usage.jsonl")[-1]["remote"], "127.0.0.1")

    def test_a_batch_goes_there_too(self):
        h = self.home()
        h.file("tasks.jsonl", "\n".join(json.dumps({"id": i, "prompt": f"[[say:{i}]]", "out": f"{i}.md"})
                                        for i in ("a", "b")))
        rc, out, err = h.worker("batch", "tasks.jsonl", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "b.md").read_text(encoding="utf-8"), "b")
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 2))

    def test_a_file_that_may_not_leave_is_refused_not_run_here(self):
        h = self.home()
        outside = h.dir / "outside.c"
        outside.write_text("int x;\n", encoding="utf-8")
        rc, out, err = h.worker("ask", "-o", "o.md", "-f", str(outside), "-p", "x", **self.KEY)
        self.assertEqual(rc, 1, out)
        self.assertIn("the local worker is off", out)
        self.assertIn("is not under remote.roots", out)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 0))

    def test_what_needs_the_local_worker_is_refused(self):
        h = self.home(roles={"volume": "local"})
        for args in (["start"], ["agent", "-p", "x"], ["ask", "--local", "-p", "x"],
                     ["ask", "--role", "volume", "-p", "x"]):
            rc, out, err = h.worker(*args, **self.KEY)
            self.assertEqual(rc, 1, f"{args}: {out} {err}")
            self.assertIn("the local worker is off", out, args)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 0))

    def test_status_says_so(self):
        h = self.home(serve=False)
        rc, out, err = h.worker("status")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("local worker off", out)

    def test_on_brings_it_back(self):
        h = self.home()
        rc, out, err = h.worker("local", "on")
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("local", json.loads((h.dir / "config.json").read_text(encoding="utf-8")))
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:here]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (1, 0))

    def test_the_environment_turns_it_off_for_one_run(self):
        h = Home(self)
        h.save(remote={"url": h.elsewhere(), "model": "base-model", "roots": [str(h.work)]})
        rc, out, err = h.worker("ask", "-o", "a.md", "-p", "[[say:x]]", CRANKER_LOCAL="0", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((len(h.requests()), len(h.requests(h.away))), (0, 1))

    def test_the_read_guard_sends_the_map_there_or_lets_the_read_through(self):
        h = self.home(serve=False)
        h.save(read_guard={"roots": [str(h.dir)]}, local=False)  # save() writes its own copy over `local off`'s
        inside, outside = h.file("big.txt", "x\n" * 500), h.dir / "big.txt"
        outside.write_text("x\n" * 500, encoding="utf-8")

        def guard(path):
            event = {"tool_name": "Read", "tool_input": {"file_path": str(path)}, "cwd": str(h.work)}
            return h.run(SCRIPTS / "read_guard.py", stdin=json.dumps(event))[1]

        said = guard(inside)
        self.assertIn("deny", said)
        self.assertIn("local worker is off", said)
        self.assertNotIn("start it first", said)
        self.assertEqual(guard(outside), "", "nobody can map a file that may not leave: the Read goes through")


if __name__ == "__main__":
    unittest.main()
