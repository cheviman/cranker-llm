"""cranker-llm's tests: the worker and its hooks against tests/fake_llama.py, a stand-in for llama-server, so they
need no GPU and no model, and run in a couple of minutes.

    py -m unittest discover -s tests          (python3 elsewhere; -v names each test, -k NAME runs some)

Every test has a CRANKER_HOME of its own in a temporary folder and a port of its own: the real worker, its config,
its logs and a server that may be running are never touched. What the fake answers is scripted in the prompt (see
fake_llama.py). The agent itself (Claude Code on the local model) is not run here: only its relay and its lock.
Standard library only.
"""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
WORKER = SCRIPTS / "worker.py"
FAKE = REPO / "tests" / "fake_llama.py"
PY = sys.executable
RAISES = "raise_exception"  # what the fake's own chat template has, and the patched copy must not


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


class Home:
    """A CRANKER_HOME of its own, and with serve=True a fake server on its port"""

    def __init__(self, test, serve=True, **config):
        self.dir = Path(tempfile.mkdtemp(prefix="cranker-test-"))
        test.addCleanup(self.close)
        self.work = self.dir / "work"  # where the worker is called from: outside any repository, with no brief
        self.work.mkdir()
        self.port = free_port()
        self.record = self.dir / "requests.jsonl"
        (self.dir / "model.gguf").write_bytes(b"not a model")
        self.config = {"model": str(self.dir / "model.gguf"), "port": self.port, "thermal_pause": False, **config}
        self.save()
        self.started = []
        if serve:
            self.start(FAKE, "--port", self.port, "--record", self.record)
            self.wait(lambda: get(self.port, "/health"), "the fake server did not come up")

    def save(self, bom=False, **config):
        self.config.update(config)
        (self.dir / "config.json").write_text(json.dumps(self.config), encoding="utf-8-sig" if bom else "utf-8")

    @staticmethod
    def wait(what, error, seconds=20):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if what():
                return
            time.sleep(0.1)
        raise AssertionError(error)

    def env(self, **more):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CRANKER_", "CLAUDE_CODE_SESSION"))}
        return {**env, "CRANKER_HOME": str(self.dir), **more}

    def start(self, script, *args, **env):
        p = subprocess.Popen([PY, str(script), *map(str, args)], cwd=self.work, env=self.env(**env),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace")
        self.started.append(p)  # ended in close(), whatever the test did with it
        return p

    def run(self, script, *args, stdin="", **env):
        """(exit code, stdout, stderr)"""
        p = self.start(script, *args, **env)
        out, err = p.communicate(stdin, timeout=180)
        return p.returncode, out.strip(), err.strip()

    def worker(self, *args, **env):
        return self.run(WORKER, *args, **env)

    def file(self, name, text, bom=False):
        path = self.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")
        return path

    def requests(self):
        if not self.record.exists():
            return []
        return [json.loads(l) for l in self.record.read_text(encoding="utf-8").splitlines()]

    def log(self, name):
        path = self.dir / "logs" / name
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def launcher(self, load=0):
        """What `start` runs in llama-server's place: the fake, taking `load` seconds to answer /health"""
        if os.name == "nt":
            path = self.dir / "fake-llama-server.cmd"
            path.write_text(f'@"{PY}" "{FAKE}" --load {load} %*\n', encoding="utf-8")
        else:
            path = self.dir / "fake-llama-server"
            path.write_text(f'#!/bin/sh\nexec "{PY}" "{FAKE}" --load {load} "$@"\n', encoding="utf-8")
            path.chmod(0o755)
        self.save(llama_server=str(path))

    def close(self):
        get(self.port, "/quit")  # also one that `start` left detached
        for p in self.started:
            if p.poll() is None:
                p.kill()
            p.communicate()
        shutil.rmtree(self.dir, ignore_errors=True)


def check_for(home, word):
    """A check that passes when the output is exactly `word`"""
    script = home.file("check.py", "import sys\ntext = open(sys.argv[1], encoding='utf-8').read()\n"
                                   f"print('got', repr(text))\nsys.exit(text != {word!r})\n")
    return f'"{PY}" "{script}" "{{out}}"'


def user_text(request):
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


class Ask(unittest.TestCase):
    def test_the_answer_goes_to_the_file_and_the_usage_to_the_log(self):
        h = Home(self)
        src = h.file("src.c", "one\ntwo\nthree\nfour\n")
        rc, out, err = h.worker("ask", "-f", f"{src}:2-3", "-o", "map.md", "-p", "[[say:hello there]]")
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual((h.work / "map.md").read_text(encoding="utf-8"), "hello there")
        self.assertRegex(out, r"^-> map\.md: in \d+ tok .* out 2 tok")
        sent = user_text(h.requests()[0])
        self.assertIn("(lines 2-3) ===\ntwo\nthree\n=== END FILE", sent)
        self.assertNotIn("four", sent)
        (used,) = h.log("usage.jsonl")
        self.assertEqual((used["cmd"], used["out"], used["answer"], used["files"]), ("ask", 2, 2, 1))

    def test_number_gives_the_model_each_line_with_its_number(self):
        h = Home(self)
        src = h.file("src.c", "one\ntwo\nthree\nfour\n")
        h.worker("ask", "-f", f"{src}:2-3", "--number", "-p", "x")
        self.assertIn("\n2\ttwo\n3\tthree\n=== END FILE", user_text(h.requests()[-1]))
        self.assertIn("every line starts with its number", user_text(h.requests()[-1]))
        h.worker("ask", "-f", src, "--number", "-p", "x")
        self.assertIn("===\n1\tone\n2\ttwo\n3\tthree\n4\tfour\n=== END FILE", user_text(h.requests()[-1]))
        h.worker("ask", "-f", src, "-p", "x")  # without it the file goes as it is
        self.assertIn("===\none\ntwo\nthree\nfour\n\n=== END FILE", user_text(h.requests()[-1]))

    def test_a_cut_answer_exits_2_and_says_what_to_do(self):
        h = Home(self)
        rc, out, _ = h.worker("ask", "-o", "x.md", "-p", "[[say:half]] [[cut]]")
        self.assertEqual(rc, 2)
        self.assertIn("FINISH=length", out)
        self.assertIn("next: send it again with something changed", out)

    def test_thinking_gets_a_budget_and_an_empty_answer_is_recovered(self):
        h = Home(self)
        rc, out, _ = h.worker("ask", "--think", "--max-tokens", "9000", "-o", "x.md", "-p", "[[empty-think]]")
        self.assertEqual(rc, 0)
        self.assertIn("RECOVERED", out)
        self.assertEqual((h.work / "x.md").read_text(encoding="utf-8"), "recovered")
        first, second = h.requests()
        self.assertEqual(first["reasoning_budget_tokens"], 9000 - 2048)
        self.assertTrue(first["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(second["chat_template_kwargs"]["enable_thinking"])
        self.assertIn("half a thought", user_text(second))

    def test_a_failed_check_is_fixed_by_the_worker(self):
        h = Home(self)
        rc, out, _ = h.worker("ask", "-o", "x.txt", "-p", "[[say:bad]] [[then:good]]", "--check", check_for(h, "good"))
        self.assertEqual(rc, 0, out)
        self.assertIn("check passed after 1 fix by the worker", out)
        self.assertEqual((h.work / "x.txt").read_text(encoding="utf-8"), "good")
        self.assertFalse((h.work / "x.txt.check.txt").exists())
        fix = h.requests()[1]["messages"]
        self.assertEqual(fix[2], {"role": "assistant", "content": "bad"})
        self.assertIn("got 'bad'", fix[3]["content"])  # the check's own output goes back to the model
        self.assertEqual([(q["result"], q["tries"]) for q in h.log("quality.jsonl")], [("fixed", 1)])

    def test_a_check_still_failing_is_left_with_its_errors(self):
        h = Home(self)
        rc, out, _ = h.worker("ask", "-o", "x.txt", "-p", "[[say:bad]]", "--retries", "1", "--check",
                              check_for(h, "good"))
        self.assertEqual(rc, 2)
        self.assertIn("CHECK FAILED after 1 fix: x.txt.check.txt", out)
        self.assertIn("got 'bad'", (h.work / "x.txt.check.txt").read_text(encoding="utf-8"))
        self.assertEqual(len(h.requests()), 2)

    def test_the_brief_goes_into_the_system_prompt(self):
        h = Home(self)
        h.file(".cranker/brief.md", "Tabs, never spaces.")
        system = lambda r: r["messages"][0]["content"]
        h.worker("ask", "-p", "x")
        self.assertIn("Tabs, never spaces.", system(h.requests()[-1]))
        h.worker("ask", "-p", "x", "--brief", "")
        self.assertNotIn("Tabs, never spaces.", system(h.requests()[-1]))
        rc, out, err = h.worker("ask", "-p", "x", "--brief", "no-such-brief.md")
        self.assertEqual(rc, 1)
        self.assertIn("cannot be read", out)
        self.assertNotIn("Traceback", err)

    def test_an_output_over_one_of_its_own_files_is_refused(self):
        h = Home(self)
        src = h.file("src.c", "int main(void) { return 0; }\n")
        rc, out, _ = h.worker("ask", "-f", f"{src}:1-1", "-o", src, "-p", "[[say:int ma]] [[cut]]")
        self.assertEqual(rc, 1)
        self.assertIn("one of the task's own files", out)
        self.assertEqual(src.read_text(encoding="utf-8"), "int main(void) { return 0; }\n")
        self.assertEqual(h.requests(), [])

    def test_a_server_that_goes_away_is_said_in_a_line(self):
        h = Home(self)
        rc, out, err = h.worker("ask", "-o", "x.md", "-p", "[[die]]")
        self.assertEqual(rc, 1)
        self.assertIn("the server did not answer", out)
        self.assertNotIn("Traceback", err)

    def test_http_errors(self):
        h = Home(self, context_wait=0)
        rc, out, _ = h.worker("ask", "-p", "[[fail:400:the request is wrong]]")
        self.assertEqual(rc, 1)
        self.assertIn("HTTP 400", out)
        rc, out, _ = h.worker("ask", "-p", "[[fail:500:Context size has been exceeded]]")  # retried while it waits
        self.assertEqual(rc, 1)
        self.assertIn("still no room after 0 s", out)


class SharedContext(unittest.TestCase):
    def test_a_request_that_does_not_fit_beside_the_running_one_is_not_sent(self):
        h = Home(self, ctx=2000, context_wait=0)
        first = h.start(WORKER, "ask", "--max-tokens", "600", "-o", "a.md", "-p", "[[sleep:3]]")
        h.wait(lambda: any(s["is_processing"] for s in get(h.port, "/slots")), "the first request never ran")
        rc, out, _ = h.worker("ask", "--max-tokens", "600", "-o", "b.md", "-p", "x")
        self.assertEqual(rc, 2)  # not run, as in batch and agent: 1 is an error
        self.assertRegex(out, r"not run: no room: \d+ tokens needed, \d+ held by 1 running requests")
        self.assertEqual(first.wait(timeout=30), 0)
        self.assertEqual(len(h.requests()), 1)

    def test_one_over_the_whole_context_fails_at_once(self):
        h = Home(self, ctx=2000)
        rc, out, _ = h.worker("ask", "--max-tokens", "1500", "-p", "x")
        self.assertEqual(rc, 1)
        self.assertIn("over the context (2000) alone", out)

    def test_while_an_agent_has_the_worker_a_task_says_whose_it_is(self):
        h = Home(self, context_wait=0)
        holder = h.start("-c", "import sys, time; sys.path.insert(0, sys.argv[1]); import worker\n"
                         "lock = worker.agent_turn(1)\nworker.agent_note('running', 600, 'Write a test for it')\n"
                         "print('held', flush=True)\ntime.sleep(60)", SCRIPTS)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        rc, out, _ = h.worker("ask", "-p", "x")
        self.assertEqual(rc, 2)
        self.assertRegex(out, r'not run: an agent started outside Claude Code, pid \d+ \(.*"Write a test for it\.\.\."\)'
                              r" has the worker")
        self.assertIn("taken: an agent", h.worker("status")[1])
        self.assertEqual(h.requests(), [])


class Batch(unittest.TestCase):
    def tasks(self, h, *lines, bom=False):
        return h.file("tasks.jsonl", "".join((l if isinstance(l, str) else json.dumps(l)) + "\n" for l in lines), bom)

    def test_it_runs_reports_and_resumes(self):
        h = Home(self)
        tasks = self.tasks(h, {"id": "a", "prompt": "[[say:one]]", "out": "a.md"},
                           {"id": "b", "prompt": "[[say:half]] [[cut]]", "out": "b.md"},
                           {"id": "c", "prompt": "[[say:bad]] [[then:good]]", "out": "c.md"},
                           bom=True)  # as Windows PowerShell 5.1 writes it
        run = ("batch", tasks, "--check", check_for(h, "good"), "--retries", "1")
        rc, out, _ = h.worker(*run)
        self.assertEqual(rc, 2, out)
        self.assertIn("3 tasks, 0 already done, 3 to run", out)
        self.assertRegex(out, r"BAD a: .*CHECK FAILED after 1 fix: a\.md\.check\.txt")
        self.assertRegex(out, r"ok  c: .*check passed after 1 fix")
        self.assertIn("done: 1 ok, 2 failed or cut", out)
        self.assertIn("2 of them left an output", out)
        sent = len(h.requests())
        rc, out, _ = h.worker(*run)  # the same batch again, as after a `not run`: nothing is sent
        self.assertEqual(rc, 2)
        self.assertIn("3 tasks, 1 already done, 0 to run", out)
        self.assertIn("2 failed before (cut, or failing their check) and their tasks have not changed", out)
        self.assertIn("not run again: a, b.", out)
        self.assertEqual(len(h.requests()), sent)

    def test_a_failed_task_runs_again_once_its_line_changes(self):
        h = Home(self)
        cut = {"id": "a", "prompt": "[[say:half]] [[cut]]", "out": "a.md"}
        done = {"id": "b", "prompt": "[[say:whole]]", "out": "b.md"}
        self.assertEqual(h.worker("batch", self.tasks(h, cut, done))[0], 2)
        self.assertTrue((h.work / "a.md.failed.json").exists())
        rc, out, _ = h.worker("batch", self.tasks(h, {**cut, "prompt": "[[say:all of it]]"}, done))
        self.assertEqual(rc, 0, out)
        self.assertIn("2 tasks, 1 already done, 1 to run", out)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "all of it")
        self.assertFalse((h.work / "a.md.failed.json").exists())
        self.assertIn("2 tasks, 2 already done, 0 to run", h.worker("batch", self.tasks(h, cut, done))[1])

    def test_what_all_the_tasks_share_is_given_once(self):
        h = Home(self)
        tasks = self.tasks(h, {"prompt": "first", "out": "a.md"}, {"prompt": "second", "out": "b.md", "system": "own"})
        rc, out, _ = h.worker("batch", tasks, "--system-file", h.file("system.txt", "shared system"),
                              "--prompt-prefix", h.file("rules.txt", "the rules"))
        self.assertEqual(rc, 0, out)
        sent = {user_text(r): r["messages"][0]["content"] for r in h.requests()}
        self.assertEqual(sent, {"the rules\n\nfirst": "shared system", "the rules\n\nsecond": "own"})

    def test_the_tasks_are_checked_before_any_runs(self):
        h = Home(self)
        good = {"id": "a", "prompt": "x", "out": "a.md"}
        for lines, error in (
                ((good, '{"id": "b", "prompt": "x" "out": "b.md"}'), "tasks.jsonl, line 2: not JSON"),
                ((good, {"id": "b", "out": "b.md"}), 'line 2: a task is an object with at least "prompt" and "out"'),
                ((good, {"id": "b", "prompt": "x", "out": "a.md"}), 'tasks a and b write the same "out"'),
                ((good, {"id": "b", "prompt": "x", "out": "s.c", "files": ["s.c:1-9"]}), "task b: its \"out\" is one"),
        ):
            rc, out, err = h.worker("batch", self.tasks(h, *lines))
            self.assertEqual(rc, 1, out)
            self.assertIn(error, out)
            self.assertNotIn("Traceback", err)
        self.assertEqual(h.requests(), [])

    def test_ctrl_c_ends_it(self):
        h = Home(self)
        tasks = self.tasks(h, *({"id": str(i), "prompt": "[[sleep:3]]", "out": f"{i}.md"} for i in range(6)))
        t0 = time.time()  # 9 s if it ran to the end; before, Ctrl+C did not stop the tasks still waiting
        rc, out, _ = h.run("-c", "import sys, threading, _thread; sys.path.insert(0, sys.argv[1]); import worker\n"
                                 "sys.argv = ['worker.py', 'batch', sys.argv[2], '--jobs', '2']\n"
                                 "threading.Timer(1.5, _thread.interrupt_main).start()\nworker.main()", SCRIPTS, tasks)
        self.assertEqual(rc, 130)
        self.assertIn("interrupted: 0 of 6 tasks had ended", out)
        self.assertLess(time.time() - t0, 6)


class Server(unittest.TestCase):
    def test_start_patches_the_template_and_stop_stops_it(self):
        h = Home(self, serve=False)
        h.launcher()
        rc, out, _ = h.worker("start", "--timeout", "60")
        self.assertEqual(rc, 0, out)
        self.assertIn("patched copy", out)
        self.assertIn("ready at", out)
        self.assertNotIn(RAISES, get(h.port, "/props")["chat_template"])
        pid, started = (h.dir / "logs" / "server.pid").read_text().split(" ", 1)
        self.assertTrue(pid.isdigit() and started)
        self.assertIn("already running", h.worker("start")[1])
        self.assertIn("running at", h.worker("status")[1])
        self.assertEqual(h.worker("stop")[0], 0)
        h.wait(lambda: not get(h.port, "/health"), "the server still answers after stop")

    def test_two_starts_at_once_leave_one_server_that_stop_stops(self):
        h = Home(self, serve=False, template="model")
        h.launcher(load=3)
        first = h.start(WORKER, "start", "--timeout", "60")
        time.sleep(1)
        rc, out, err = h.worker("start", "--timeout", "60")
        self.assertEqual((rc, first.wait(timeout=60)), (0, 0), out + err)
        self.assertIn("another session is starting the server", err)
        self.assertIn("already running", out)
        h.worker("stop")
        h.wait(lambda: not get(h.port, "/health"), "a second server was left running, with no pid file to stop it by")

    def test_a_start_cut_short_is_waited_for_not_started_again(self):
        h = Home(self, serve=False, template="model")
        h.launcher(load=4)
        first = h.start(WORKER, "start", "--timeout", "60")
        h.wait(lambda: (h.dir / "logs" / "server.pid").exists(), "the first start never launched the server")
        first.kill()
        pid_file = (h.dir / "logs" / "server.pid").read_text()
        rc, out, err = h.worker("start", "--timeout", "60")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("a server is already starting", err)
        self.assertEqual((h.dir / "logs" / "server.pid").read_text(), pid_file)
        h.worker("stop")
        h.wait(lambda: not get(h.port, "/health"), "the server still answers after stop")

    def test_stop_spares_a_process_that_got_the_servers_pid(self):
        h = Home(self, serve=False)
        other = h.start("-c", "import time; time.sleep(60)")
        (h.dir / "logs").mkdir()
        (h.dir / "logs" / "server.pid").write_text(f"{other.pid} 12345")  # the server's start time: not this one's
        rc, out, _ = h.worker("stop")
        self.assertIn("was not running", out)
        time.sleep(1)
        self.assertIsNone(other.poll(), "stop killed a process that only had the old server's pid")

    def test_stop_says_when_the_server_is_not_its_own(self):
        h = Home(self)  # a server on the port, and no pid file
        rc, out, _ = h.worker("stop")
        self.assertEqual(rc, 1)
        self.assertIn("this worker did not start it", out)
        self.assertTrue(get(h.port, "/health"))


class ConfigAndHooks(unittest.TestCase):
    def guard(self, h, path, **tool_input):
        event = {"tool_name": "Read", "tool_input": {"file_path": str(path), **tool_input}, "cwd": str(h.work)}
        return "deny" in h.run(SCRIPTS / "read_guard.py", stdin=json.dumps(event))[1]

    def test_a_config_with_a_bom_is_read(self):
        h = Home(self, serve=False)
        big = h.file("big.txt", "x\n" * 500)
        h.save(bom=True, read_guard={"roots": [str(h.work)]})
        rc, out, err = h.worker("config")
        self.assertEqual(rc, 0, err)
        self.assertIn(f'"port": {h.port}', out)
        self.assertTrue(self.guard(h, big), "with a BOM in the config the read guard let the Read through")

    def test_a_broken_config_is_said_in_a_line(self):
        h = Home(self, serve=False)
        (h.dir / "config.json").write_text('{"port": ', encoding="utf-8")
        rc, out, err = h.worker("status")
        self.assertEqual(rc, 1)
        self.assertIn("is not valid JSON", err)
        self.assertNotIn("Traceback", err)

    def test_the_read_guard(self):
        root = Path(tempfile.gettempdir()).anchor  # C:\ or /
        h = Home(self, serve=False, read_guard={"roots": [root]})
        big, small = h.file("big.txt", "x\n" * 500), h.file("small.txt", "x\n" * 100)
        self.assertTrue(self.guard(h, big), f"a root that is a drive's root ({root}) matched nothing under it")
        self.assertFalse(self.guard(h, small))
        self.assertFalse(self.guard(h, big, offset=1, limit=500))
        self.assertEqual([g["lines"] for g in h.log("read_guard.jsonl")], [500])
        h.save(read_guard={"roots": [str(h.dir / "elsewhere")]})
        self.assertFalse(self.guard(h, big))

    def test_usage_adds_up_and_survives_a_line_cut_short(self):
        h = Home(self)
        h.worker("ask", "-o", "a.md", "-p", "[[say:one two three]]", CLAUDE_CODE_SESSION_ID="abcdef123456")
        h.worker("ask", "-o", "b.md", "-p", "[[say:]]")  # an empty answer: consumed, not effective
        (h.dir / "logs" / "read_guard.jsonl").write_text('{"ts": "2026-11-19 10:00:00", "path": "a", "lines": 400}\n'
                                                         '{"ts": "2026-1', encoding="utf-8")
        rc, out, err = h.worker("usage")
        self.assertEqual(rc, 0, err)
        self.assertIn("2 requests", out)
        self.assertIn("1 runs were not effective (1 gave nothing back)", out)
        self.assertIn("2 pieces, 1 right at the first try, 0 by try 2, 0 took more tries, 1 left wrong", out)
        self.assertIn("read guard: 1 whole-file reads", out)
        self.assertIn("1 requests by session abcdef from", h.worker("usage", "--session", "abcdef")[1])

    def test_the_scale_hook_notes_a_session_below_it_once(self):
        h = Home(self, serve=False)
        (h.dir / "logs").mkdir()
        (h.dir / "logs" / "usage.jsonl").write_text(json.dumps(
            {"ts": "2026-11-19 10:00:00", "cmd": "ask", "cwd": "x", "in": 900, "out": 100, "answer": 100, "s": 1,
             "session": "s1"}) + "\n", encoding="utf-8")
        transcript = h.file("s1.jsonl", json.dumps(
            {"timestamp": "2026-11-19T13:00:00Z", "cwd": "x",
             "message": {"id": "m1", "usage": {"input_tokens": 200_000, "output_tokens": 500}}}) + "\n")
        event = json.dumps({"session_id": "s1", "transcript_path": str(transcript), "cwd": str(h.work),
                            "hook_event_name": "PostToolUse"})
        rc, out, _ = h.run(SCRIPTS / "local_share.py", "--hook", stdin=event)
        note = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("200,500 tokens of Claude and 1,000 effective tokens of the local worker", note)
        self.assertEqual(h.run(SCRIPTS / "local_share.py", "--hook", stdin=event)[1], "")  # not again for 15 minutes


class Scale(unittest.TestCase):
    """The scale is on work that came out right, and the hook says what the user was saying by hand"""

    def session(self, h, claude, local, sid="s1"):
        """A session that has spent `claude` tokens of Claude and has `local` tokens of the worker's work done"""
        (h.dir / "logs").mkdir(exist_ok=True)
        (h.dir / "logs" / "usage.jsonl").write_text(json.dumps(
            {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": "ask", "cwd": "x", "in": local - 100, "out": 100,
             "answer": 100, "s": 1, "session": sid}) + "\n", encoding="utf-8")
        transcript = h.file(f"{sid}.jsonl", json.dumps(
            {"timestamp": "2026-11-19T13:00:00Z", "cwd": "x",
             "message": {"id": "m1", "usage": {"input_tokens": claude, "output_tokens": 0}}}) + "\n")
        return lambda event, cwd=h.work: self.said(h, {"session_id": sid, "transcript_path": str(transcript),
                                                       "cwd": str(cwd), "hook_event_name": event})

    def said(self, h, event):
        out = h.run(SCRIPTS / "local_share.py", "--hook", stdin=json.dumps(event))[1]
        if not out:
            return ""
        answer = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(answer["hookEventName"], event["hook_event_name"])
        return answer["additionalContext"]

    def test_every_run_is_consumed_and_work_is_effective_when_right_by_the_second_try(self):
        h = Home(self)
        src = h.file("src.c", "one\ntwo\nthree\n")
        h.worker("ask", "-f", src, "-o", "a.md", "-p", "[[say:first try]]")
        h.worker("ask", "-f", f"{src}:1-2", "-o", "a.md", "-p", "[[say:second try]] narrower, another prompt")
        h.worker("ask", "-f", h.file("other.c", "x\n"), "-o", "a.md", "-p", "[[say:other work]]")  # same out only
        for n in range(3):
            h.worker("ask", "-o", "b.md", "-p", f"[[say:try {n}]]")  # a task that took three runs
        h.worker("ask", "-o", "c.md", "-p", "[[say:half]] [[cut]]")
        h.worker("ask", "-o", "f.md", "-p", "[[say:bad]] [[then:good]]", "--check", check_for(h, "good"))
        first, second, other, b1, b2, b3, cut, bad, fixed = runs = h.log("usage.jsonl")
        self.assertEqual((first["out"], first["dest"]), (2, str((h.work / "a.md").resolve())))  # tokens, and where
        self.assertEqual(first["work"], second["work"])
        self.assertNotEqual(first["work"], other["work"])
        self.assertEqual((cut.get("cut"), bad["check"], fixed["check"]), (True, "fail", "pass"))
        rc, out, err = h.worker("usage")
        self.assertEqual(rc, 0, err)
        spent = sum(r["in"] + r["out"] for r in runs)  # consumed: every run, the three of b.md too
        effective = sum(r["in"] + r["out"] for r in (first, second, other, bad, fixed))  # right by the second try
        self.assertIn(f"consumed {spent:,} tokens (in + out), effective {effective:,}: 4 runs were not effective "
                      "(3 were of work that took more than 2 tries, 1 were cut)", out)
        self.assertIn("5 pieces, 1 right at the first try, 2 by try 2, 1 took more tries, 1 left wrong", out)

    def test_the_hook_tells_volume_from_quality(self):
        h = Home(self, serve=False, scale={"floor": 0})
        hook = self.session(h, claude=10_000, local=1_000)
        again = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": "batch", "cwd": "x", "in": 9_000, "out": 1_000,
                 "answer": 1_000, "s": 1, "session": "s1", "dest": "x.md", "work": "abc"}
        with open(h.dir / "logs" / "usage.jsonl", "a", encoding="utf-8") as f:  # one task, five runs
            f.write("".join(json.dumps(again) + "\n" for _ in range(5)))
        for said in (hook("PostToolUse"), hook("UserPromptSubmit")):
            self.assertIn("10,000 tokens of Claude and 1,000 effective tokens of the local worker (0.10 to 1)", said)
            self.assertIn("consumed 51,000 in all (5.10 to 1): 50,000 went on work that was not right by try 2", said)
            self.assertIn("Those tokens were spent all the same", said)
            self.assertIn("of its 1 pieces of work, 0 were right at the first try, 0 by try 2, 1 took more tries and "
                          "0 were left wrong", said)
            self.assertIn("The worker has had the volume: what is short is quality. Do not send more", said)

    def test_the_rule_is_said_when_the_session_starts_and_after_a_compaction(self):
        h = Home(self, serve=False, scale={"roots": [str(Path(tempfile.gettempdir()))]})
        start = {"session_id": "new", "transcript_path": str(h.work / "new.jsonl"), "cwd": str(h.work),
                 "hook_event_name": "SessionStart", "source": "startup"}
        rule = self.said(h, start)
        self.assertIn("this machine has a local worker", rule)
        self.assertIn("not a quota", rule)
        h.save(scale={"roots": []})
        self.assertEqual(self.said(h, start), "")  # not one of the listed folders, and it never used the worker
        hook = self.session(h, claude=0, local=1_000)
        self.assertIn("this machine has a local worker", hook("SessionStart"))  # it did: after a compaction, say
        h.save(scale={"remind": False})
        self.assertEqual(hook("SessionStart"), "")

    def test_every_prompt_is_reminded_only_while_claude_does_more_than_the_worker(self):
        h = Home(self, serve=False, scale={"floor": 0})
        hook = self.session(h, claude=200_000, local=1_000)
        for _ in range(2):  # at every prompt, not every 15 minutes
            said = hook("UserPromptSubmit")
            self.assertIn("1,000 effective tokens of the local worker (0.01 to 1)", said)
            self.assertIn("below the 1 to 1 the user wants at the least", said)
            self.assertIn("Before you start on this request yourself, send the worker its part of it", said)
        hook = self.session(h, claude=200_000, local=300_000)  # more than Claude, short of 5 to 1
        self.assertEqual(hook("UserPromptSubmit"), "")
        (h.dir / "logs" / "local_share.json").unlink()
        said = hook("PostToolUse")  # the note, at its own pace
        self.assertIn("The scale the user set is 5 effective local tokens", said)
        self.assertIn("Give the worker real load", said)
        self.assertEqual(hook("PostToolUse"), "")
        hook = self.session(h, claude=200_000, local=1_200_000)  # at the scale: nothing to say
        (h.dir / "logs" / "local_share.json").unlink()
        self.assertEqual((hook("UserPromptSubmit"), hook("PostToolUse")), ("", ""))


def load(path, h):
    """One of the scripts as a module in this process, on the test's own home"""
    with mock.patch.dict(os.environ, {"CRANKER_HOME": str(h.dir)}):
        spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class RangeServer:
    """Serves one blob in ranges and counts the GETs; with `limit`, only that many get an answer"""

    def __init__(self, data):
        self.gets, self.limit = 0, None
        lock, this = threading.Lock(), self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()

            def do_GET(self):
                with lock:
                    this.gets += 1
                    refuse = this.limit is not None and this.gets > this.limit
                if refuse:
                    return self.send_error(500)
                first, last = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                self.send_response(206)
                self.send_header("Content-Length", str(last - first + 1))
                self.end_headers()
                self.wfile.write(data[first:last + 1])

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/blob.gguf"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Setup(unittest.TestCase):
    def download(self, setup, *args, **kw):
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            setup.download(*args, **kw)
        return said.getvalue()

    def test_a_download_that_was_cut_goes_on_from_its_pieces(self):
        h = Home(self, serve=False)
        setup = load(SCRIPTS / "setup.py", h)
        setup.RETRY_WAIT = 0
        data = os.urandom(1 << 20)
        sha = hashlib.sha256(data).hexdigest()
        server = RangeServer(data)
        self.addCleanup(server.close)
        out = h.dir / "models" / "blob.gguf"
        state = h.dir / "models" / "blob.gguf.part.json"
        server.limit = 3  # the connection goes after three of the eight pieces
        with self.assertRaises(RuntimeError):
            self.download(setup, server.url, out, len(data), sha, parts=8, jobs=2)
        self.assertFalse(out.exists())
        self.assertEqual(len(json.loads(state.read_text(encoding="utf-8"))["done"]), 3)
        server.limit, server.gets = None, 0
        said = self.download(setup, server.url, out, len(data), sha, parts=8, jobs=2)
        self.assertIn("3 of 8 pieces are here", said)
        self.assertEqual(server.gets, 5, "it downloaded again pieces it already had")
        self.assertEqual(out.read_bytes(), data)
        self.assertEqual(sorted(p.name for p in out.parent.iterdir()), ["blob.gguf", "blob.gguf.sha256"])

    def test_a_wrong_checksum_starts_over_the_next_time(self):
        h = Home(self, serve=False)
        setup = load(SCRIPTS / "setup.py", h)
        data = os.urandom(1 << 16)
        server = RangeServer(data)
        self.addCleanup(server.close)
        out = h.dir / "models" / "blob.gguf"
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.download(setup, server.url, out, len(data), "0" * 64, parts=4, jobs=2)
        self.assertEqual([p.name for p in out.parent.iterdir()], ["blob.gguf.part"])


class Bench(unittest.TestCase):
    def test_the_lines_step_scores_a_map(self):
        bench = load(REPO / "bench" / "bench.py", Home(self, serve=False))
        want = {"load": 10, "save": 20, "main": 30}
        text = "`load` (line 10): reads it\nsave (line 21): writes it\n`main` (lines 30-40): runs it\nother (line 5): x"
        self.assertEqual(bench.score_lines(text, want), (3, 2))


class Agent(unittest.TestCase):
    """Its relay and its environment, in this process: the worker's module is loaded on a home of its own"""

    def load(self, h):
        return load(WORKER, h)

    def post(self, url, body):
        req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_the_relay_caps_the_thinking_drops_the_old_and_counts_the_cuts(self):
        h = Home(self)
        worker = self.load(h)
        relay = worker.AgentRelay(f"http://127.0.0.1:{h.port}", budget=4096, stall_limit=2)
        relay.start()
        self.addCleanup(relay.stop)
        thought = {"type": "thinking", "thinking": "..."}
        turn = lambda text: {"role": "assistant", "content": [thought, {"type": "text", "text": text}]}
        body = {"model": "x", "max_tokens": 8000, "thinking": {"type": "adaptive"},
                "messages": [{"role": "user", "content": "[[cut]]"}, turn("one"), {"role": "user", "content": "go on"},
                             turn("two"), {"role": "user", "content": "go on"}]}
        self.assertEqual(self.post(relay.url, body)["stop_reason"], "max_tokens")
        self.assertFalse(relay.stalled)
        self.post(relay.url, body)
        self.assertTrue(relay.stalled, "two turns in a row cut at the output cap did not stall the agent")
        self.assertEqual((relay.requests, relay.cuts), (2, 2))
        sent = h.requests()[-1]
        self.assertEqual(sent["thinking"], {"type": "enabled", "budget_tokens": 4096})
        first, last = (m["content"] for m in sent["messages"] if m["role"] == "assistant")
        self.assertNotIn(thought, first)
        self.assertIn(thought, last)

    def test_the_agent_is_kept_on_this_machine(self):
        h = Home(self, serve=False)
        worker = self.load(h)
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_API_KEY": "the-callers-own",
                                          "CLAUDE_CODE_OAUTH_TOKEN": "real"}):
            env = worker.agent_env("http://127.0.0.1:1")
        self.assertNotIn("CLAUDE_CODE_USE_BEDROCK", env)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)
        self.assertEqual((env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_API_KEY"]), ("http://127.0.0.1:1", "local"))


if __name__ == "__main__":
    unittest.main()
