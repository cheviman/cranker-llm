"""1.7.0's tests: a batch stopped from outside (TaskStop, Ctrl+C) and the next one. On 20-11-2026 a stopped batch
left outputs that the next one took for done: an empty cut answer, and a first answer whose fix was running.

    py -m unittest discover -s tests -v

Standard library only.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from test_worker import Home, PY, WORKER

_spec = importlib.util.spec_from_file_location("worker", WORKER)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def slow_check(h):
    """Passes "good"; with the file "slow" there, it waits a minute on a task's output (not on the probe), so a
    batch can be stopped while it runs"""
    script = h.file("slow.py", "import sys, time\nfrom pathlib import Path\n"
                               "text = open(sys.argv[1], encoding='utf-8').read()\n"
                               "if Path('slow').exists() and 'cranker-probe' not in sys.argv[1]:\n"
                               "    time.sleep(60)\n"
                               "sys.exit(text != 'good')\n")
    return f'"{PY}" "{script}" "{{out}}"'


def tasks_file(h, *ids, prompt="[[say:good]]"):
    h.file("tasks.jsonl", "".join(json.dumps({"id": i, "prompt": prompt, "out": f"{i}.txt"}) + "\n" for i in ids))


class Resume(unittest.TestCase):
    def test_a_batch_stopped_while_a_check_ran_is_run_again(self):
        h = Home(self)
        check = slow_check(h)
        tasks_file(h, "a")
        h.file("slow", "")
        p = h.start(WORKER, "batch", "tasks.jsonl", "--check", check)
        h.wait(lambda: (h.work / "a.txt").exists() and (h.work / "a.txt").read_text(encoding="utf-8"),
               "the first answer was not written")
        p.kill()  # as TaskStop does: no finally, no failed mark
        p.wait(timeout=30)
        self.assertTrue((h.work / "a.txt.running.json").exists())
        self.assertFalse((h.work / "a.txt.failed.json").exists())
        (h.work / "slow").unlink()
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", check)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("1 were left unfinished by a batch that was stopped", out)
        self.assertIn("1 to run", out)
        self.assertEqual(len(h.requests()), 2)
        self.assertFalse((h.work / "a.txt.running.json").exists())

    def test_one_another_batch_is_running_is_left_to_it(self):
        h = Home(self)
        tasks_file(h, "a", "b")
        started = worker._started(os.getpid())  # this test's process: alive, as another batch would be
        (h.work / "a.txt.running.json").write_text(json.dumps({"pid": os.getpid(), "since": str(started)}),
                                                   encoding="utf-8")
        rc, out, err = h.worker("batch", "tasks.jsonl")
        self.assertEqual(rc, 2, out + err)
        self.assertIn("1 are being run by another batch now", out)
        self.assertEqual(len(h.requests()), 1)  # b only
        self.assertFalse((h.work / "a.txt").exists())

    def test_a_mark_whose_process_is_gone_is_run_again(self):
        h = Home(self)
        tasks_file(h, "a")
        gone = subprocess.run([PY, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        h.file("a.txt", "half of")
        (h.work / "a.txt.running.json").write_text(json.dumps({"pid": int(gone.stdout), "since": "1"}),
                                                   encoding="utf-8")
        rc, out, err = h.worker("batch", "tasks.jsonl")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("left unfinished", out)
        self.assertEqual((h.work / "a.txt").read_text(encoding="utf-8"), "good")

    def test_an_empty_output_with_no_mark_is_run_again(self):
        h = Home(self)
        tasks_file(h, "a", "b")
        h.file("a.txt", "")
        h.file("b.txt", "done before")
        rc, out, err = h.worker("batch", "tasks.jsonl")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("1 had an empty output and no mark of a failure, so they run again: a", out)
        self.assertEqual((h.work / "a.txt").read_text(encoding="utf-8"), "good")
        self.assertEqual((h.work / "b.txt").read_text(encoding="utf-8"), "done before")

    def test_an_ended_batch_leaves_no_marks_and_no_temporary_files(self):
        h = Home(self)
        h.file("tasks.jsonl", json.dumps({"id": "a", "prompt": "[[say:good]]", "out": "a.txt"}) + "\n"
               + json.dumps({"id": "b", "prompt": "[[say:bad]]", "out": "b.txt"}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--check", slow_check(h), "--retries", "0")
        self.assertEqual(rc, 2, out + err)
        names = sorted(p.name for p in h.work.iterdir())
        self.assertNotIn("a.txt.running.json", names)
        self.assertNotIn("b.txt.running.json", names)
        self.assertIn("b.txt.failed.json", names)
        self.assertEqual([n for n in names if "cranker-tmp" in n], [])


class CheckTimeout(unittest.TestCase):
    def test_a_check_that_hangs_in_a_process_it_started_is_stopped_at_check_timeout(self):
        h = Home(self, check_timeout=3)
        hang = h.file("hang.py", "import subprocess, sys\n"
                                 "subprocess.run([sys.executable, '-c', 'import time; time.sleep(120)'])\n")
        t0 = time.time()
        rc, out, err = h.worker("ask", "-o", "a.txt", "-p", "[[say:good]]", "--check", f'"{PY}" "{hang}" "{{out}}"',
                                "--retries", "0")
        self.assertLess(time.time() - t0, 60)  # not the two minutes of the process under it
        self.assertEqual(rc, 2, out + err)
        self.assertIn("the check took more than 3 s, and was stopped",
                      (h.work / "a.txt.check.txt").read_text(encoding="utf-8"))


class WriteAtomic(unittest.TestCase):
    def test_a_write_that_fails_leaves_the_old_file_and_no_temporary_one(self):
        h = Home(self, serve=False)
        target = h.file("out.c", "the old file")
        with mock.patch.object(worker.os, "replace", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                worker.write_atomic(target, "the new fi")
        self.assertEqual(target.read_text(encoding="utf-8"), "the old file")
        self.assertEqual([p.name for p in h.work.iterdir()], ["out.c"])
        worker.write_atomic(target, "the new file")
        self.assertEqual(target.read_text(encoding="utf-8"), "the new file")


if __name__ == "__main__":
    unittest.main()
