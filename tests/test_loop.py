"""LoopWatch and the worker's stop of an answer that goes round in a loop, against tests/fake_llama.py: the real
answer of 20-11-2026 (a review that listed `file:N: fine.` for every N, until max_tokens) and the [[loop:]] prompt
the fake streams over and over.

    py -m unittest discover -s tests -v

Standard library only."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

from test_worker import Home, FAKE, get

WORKER = Path(__file__).resolve().parent.parent / "scripts" / "worker.py"
_spec = importlib.util.spec_from_file_location("worker", WORKER)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

REVIEW = Path(__file__).resolve().parent / "fixtures" / "loop_review99.answer.md"


class LoopWatchTests(unittest.TestCase):
    def feed(self, text, messages=None, piece=5):
        messages = messages or [{"role": "user", "content": "review it"}]
        watch = worker.LoopWatch(messages)
        for i in range(0, len(text), piece):
            said = watch.feed(text[i:i + piece])
            if said:
                return said, text[:i + piece].count("\n") + 1
        return None, None

    def test_the_20_11_review_trips_inside_its_loop(self):
        said, line = self.feed(REVIEW.read_text(encoding="utf-8"))
        self.assertIsNotNone(said)
        self.assertIn("sdk/tests/subdir_project.py:#: fine.", said)
        self.assertTrue(146 <= line <= 230, line)

    def test_the_review_before_its_loop_does_not_trip(self):
        text = "".join(REVIEW.read_text(encoding="utf-8").splitlines(keepends=True)[:144])
        said, _ = self.feed(text)
        self.assertIsNone(said)

    def test_a_long_map_of_different_lines_does_not_trip(self):
        verbs = ["fills", "reads", "clears", "locks", "frees"]
        nouns = ["input", "output", "state", "queue", "buffer", "cache", "table"]
        fields = ["count", "value", "flag"]
        text = "".join(
            f"`fn_{i}` (line {i * 7}): {verbs[i % 5]} the {nouns[i % 7]} table and returns its {fields[i % 3]}.\n"
            for i in range(300))
        said, _ = self.feed(text)
        self.assertIsNone(said)

    def test_the_same_paragraph_over_and_over_trips(self):
        paragraph = " ".join(f"w{i}" for i in range(60)) + "\n\n"
        said, _ = self.feed(paragraph * 20)
        self.assertIsNotNone(said)
        self.assertIn("word for word", said)

    def test_lines_the_request_holds_may_be_copied(self):
        source = "".join(f"    PORT_BIT( 0x{1 << (i % 16):04x}, IP_ACTIVE_LOW, IPT_UNUSED )\n" for i in range(40))
        said, _ = self.feed(source, messages=[{"role": "user", "content": "Rewrite this:\n" + source}])
        self.assertIsNone(said)
        said, _ = self.feed(source)
        self.assertIsNotNone(said)
        self.assertIn("PORT_BIT", said)

    def test_a_table_of_numbers_does_not_count(self):
        # numbers and signs only: neither its lines nor its runs of words count, as in the measure of the thresholds
        said, _ = self.feed("    0x00, 0x00, 0x00, 0x00,\n" * 200)
        self.assertIsNone(said)


class LoopRunTests(unittest.TestCase):
    def test_ask_stops_the_20_11_review_and_says_so(self):
        h = Home(self)
        rc, out, err = h.worker("ask", "-o", "review99.md", "--max-tokens", "6000", "-p",
                                f"[[say-file:{REVIEW}]] review it")
        self.assertEqual(rc, 2, out + err)
        self.assertTrue(out.startswith("REPEATING: the answer went round in a loop ("), out)
        self.assertIn("sdk/tests/subdir_project.py:#: fine.", out)
        self.assertIn("next: don't raise --max-tokens", out)
        self.assertNotIn("cut by max_tokens", out)
        written = (h.work / "review99.md").read_text(encoding="utf-8")
        review = REVIEW.read_text(encoding="utf-8")
        self.assertLess(len(written), len(review))
        self.assertTrue(review.startswith(written))
        self.assertTrue(h.requests()[0]["stream"])
        self.assertTrue(h.log("usage.jsonl")[-1]["repeat"])
        self.assertTrue(h.log("usage.jsonl")[-1]["cut"])

    def test_the_request_is_stopped_at_the_server(self):
        h = Home(self, serve=False)
        streams = h.dir / "streams.jsonl"
        h.start(FAKE, "--port", h.port, "--record", h.record, "--streams", streams)
        h.wait(lambda: get(h.port, "/health"), "the fake server did not come up")
        rc, out, err = h.worker("ask", "-o", "loop.md", "--max-tokens", "60000", "-p", "[[loop:a.py:{n}: fine.]]")
        self.assertEqual(rc, 2, out + err)
        self.assertIn("REPEATING", out)
        h.wait(lambda: streams.exists() and streams.read_text(encoding="utf-8").strip(),
               "the fake did not end its stream")
        row = json.loads(streams.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(row["closed"])
        self.assertLess(row["sent"], 60000)

    def test_with_a_check_a_loop_is_not_sent_back_as_a_fix(self):
        h = Home(self)
        check = h.file("check_ok.py", "import sys\nsys.exit(0)\n")
        rc, out, err = h.worker("ask", "-o", "review99.md", "--max-tokens", "6000",
                                "--check", f'"{sys.executable}" "{check}" {{out}}', "--check-null", "none",
                                "-p", f"[[say-file:{REVIEW}]] review it")
        self.assertEqual(rc, 2, out + err)
        self.assertEqual(len(h.requests()), 1)
        self.assertTrue((h.work / "review99.md.check.txt").read_text(encoding="utf-8").startswith("REPEATING"))
        self.assertTrue((h.work / "review99.md.check.1.txt").exists())

    def test_batch_marks_a_loop_and_runs_it_again_only_changed(self):
        h = Home(self)
        tasks = h.file("tasks.jsonl",
                       json.dumps({"id": "r", "prompt": f"[[say-file:{REVIEW}]] review it", "out": "r.md",
                                   "max_tokens": 6000}) + "\n")
        rc, out, err = h.worker("batch", tasks)
        self.assertEqual(rc, 2, out + err)
        self.assertTrue(any(l.startswith("LOOP r: REPEATING") for l in out.splitlines()), out)
        self.assertIn("loops: 1 answers went round in a loop", out)
        self.assertEqual(json.loads((h.work / "r.md.failed.json").read_text(encoding="utf-8"))["why"], "repeat")
        rc, out, err = h.worker("batch", tasks)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("failed before", out)
        self.assertEqual(len(h.requests()), 1)

    def test_the_server_elsewhere_streams_and_stops_too(self):
        h = Home(self)
        h.save(remote={"url": h.elsewhere(), "model": "m", "roots": [str(h.work)]})
        rc, out, err = h.worker("ask", "-o", "review99.md", "--max-tokens", "6000", "--remote",
                                "-p", f"[[say-file:{REVIEW}]] review it", CRANKER_REMOTE_KEY="its-key")
        self.assertEqual(rc, 2, out + err)
        self.assertTrue(out.startswith("REPEATING"), out)
        req = h.requests(h.away)[0]
        self.assertTrue(req["stream"])
        self.assertEqual(req["stream_options"], {"include_usage": True})


if __name__ == "__main__":
    unittest.main()
