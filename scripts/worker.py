"""cranker-llm: a llama.cpp server on this machine that Claude Code hands volume work to.

    worker.py start | stop | status | config
    worker.py ask -p "instruction" [-f file[:a-b] ...] [--number] [-o out] [--think] [--print] [--check CMD]
                  [--brief FILE] [--remote | --local]
    worker.py batch tasks.jsonl [--jobs N] [--force] [--check CMD] [--system-file F] [--prompt-prefix F] [--number]
                  [--remote | --local]
    worker.py agent --dir folder [--dir folder ...] -p "task" [-o out] [--write folder]
    worker.py usage [--since YYYY-MM-DD] [--all]

check: a command that exits 0 when an output is right ({out} is its path); the worker hands the model its own output
and the errors until it passes or check_retries run out. An answer cut by max_tokens is asked for again with 1.5 times
the room, once; cut again, the task needs splitting. brief: .cranker/brief.md in the caller's folder or above it,
how the user works in that project, goes into every request's system prompt (the worker loads no CLAUDE.md).
out: an answer that is one fenced code block and nothing else is written without its fence (-o, a batch's "out").

ask: the files go to the model, not through the caller's context; the answer is written to -o and only a one-line
summary is printed (--print also prints the answer). From Windows PowerShell 5.1 a prompt with double quotes in it is
split into several arguments: write it to a file and pass --prompt-file. --number gives the model every line with
its number in the file, for answers that cite lines (a map, a review); not for a rewrite.
batch: one JSON object per line: {"id", "prompt", "files": [...], "out", "system"?, "think"?, "max_tokens"?,
"check"?, "retries"?, "number"?}.
Tasks whose "out" already exists are skipped unless --force, so a batch can resume. One that left its output cut or
failing its check (out.failed.json beside it) runs again once its line has changed. An "out" is never one of the
task's own files.

Exit codes of ask, batch and agent: 0 done and right; 2 an output that is not right (cut, its check still failing,
an agent stopped) or a task that was not run because the worker was taken (the last lines say which, and what to do);
1 an error (a wrong argument or task file, the server not running or gone).
agent: Claude Code itself, headless, on the local model: Read, Grep and Glob, and with --write also Write and Edit
inside that folder only.
--remote (ask, batch): the run goes to the server under "remote" in the config, any OpenAI-compatible endpoint,
instead of the local one. The files leave the machine: only from the folders under remote.roots, with the key from
an environment variable (CRANKER_REMOTE_KEY), and only when the user asks for it.
remote.spill (true in the config, the user's say): with the local worker full (an agent has it, its context has no
room, the request is over its context, or it is not running) a request whose files are all under remote.roots goes
there by itself instead of waiting, and a batch's tasks run here and there at the same time. A failure there (a
quota, its limits, the network) sends the rest back here. --local keeps a run here.

Settings: CRANKER_HOME/config.json (~/.cranker-llm by default; scripts/setup.py writes it), and the
environment variables in ENV_OVERRIDES. Standard library only.
"""

import argparse
import atexit
import collections
import concurrent.futures
import hashlib
import http.client
import http.server
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

WINDOWS = os.name == "nt"
HOME = Path(os.environ.get("CRANKER_HOME") or Path.home() / ".cranker-llm")
CONFIG_FILE = HOME / "config.json"
LOGS = HOME / "logs"
TEMPLATES = HOME / "templates"
PID_FILE = LOGS / "server.pid"   # "pid start-time" of the server this worker started (_read_pid)
START_LOCK = LOGS / "start.lock"
SERVER_LOG = LOGS / "server.log"
USAGE_LOG = LOGS / "usage.jsonl"
AGENT_LOCK = LOGS / "agent.lock"
AGENT_INFO = LOGS / "agent.json"  # who holds AGENT_LOCK (agent_note)

DEFAULTS = {
    "llama_server": "",        # path to llama-server; "" = HOME/bin, then PATH
    "model": "",               # path to the .gguf
    "template": "auto",        # auto: the model's own, patched if it refuses a late system message; model; or a path
    "alias": "cranker-llm",
    "host": "127.0.0.1",
    "port": 8090,
    "ctx": 65536,              # one pool for all the slots (-kvu)
    "parallel": 8,
    "cache_type": "q8_0",
    "spec": "ngram-mod",       # drafts from the context: much faster when the answer repeats it (rewrites, edits)
    "threads": min(8, max(1, (os.cpu_count() or 2) // 2)),
    "max_power_w": None,       # refuse to start while the NVIDIA power limit is above this (see the README)
    "thermal_pause": True,     # batch: wait before a task while the GPU slows itself down for heat
    "extra": [],               # more llama-server options
    "system": ("You are a worker another agent hands tasks to. Follow the instructions exactly and output only what "
               "is asked, with no preamble. When unsure, say so instead of guessing."),
    # Qwen3's recommended sampling, without and with thinking
    "sampling": {"plain": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
                 "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},
    "max_tokens": 4096,
    "max_tokens_thinking": 10000,  # the thinking counts: 4096 went on it alone in a 260-line diff's review
    "context_wait": 1800,          # s a request waits for the shared context to have room
    "check_retries": 2,            # a task with a check: how many times the worker fixes its own output
    "check_timeout": 300,          # s one run of a check may take
    "agent_think_budget": 4096,    # tokens an agent's turn may think before it must act (of its 8,000 per turn)
}
ENV_OVERRIDES = {"CRANKER_SERVER": "llama_server", "CRANKER_MODEL": "model",
                 "CRANKER_TEMPLATE": "template", "CRANKER_PORT": "port", "CRANKER_CTX": "ctx",
                 "CRANKER_PARALLEL": "parallel", "CRANKER_SPEC": "spec", "CRANKER_EXTRA": "extra"}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:  # utf-8-sig: Windows PowerShell 5.1 writes a BOM, and json refuses it
            own = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
            if not isinstance(own, dict):
                raise ValueError("not an object")
        except ValueError as e:
            sys.exit(f"{CONFIG_FILE} is not valid JSON ({e}): fix it, or delete it and run scripts/setup.py")
        cfg.update(own)
    for env, key in ENV_OVERRIDES.items():
        v = os.environ.get(env)
        if v is None or v == "":
            continue
        if isinstance(DEFAULTS[key], int) and not isinstance(DEFAULTS[key], bool):
            v = int(v)
        elif key == "extra":
            v = v.split()
        cfg[key] = v
    return cfg


CFG = load_config()
URL = f"http://{CFG['host']}:{CFG['port']}"


def server_binary():
    if CFG["llama_server"]:
        return Path(CFG["llama_server"])
    local = HOME / "bin" / ("llama-server.exe" if WINDOWS else "llama-server")
    if local.exists():
        return local
    found = shutil.which("llama-server")
    return Path(found) if found else local


# ---------------------------------------------------------------------------------------------------- the server

def _get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if body.get("stream") and "text/event-stream" in (r.headers.get("Content-Type") or ""):
            return read_stream(r, body["messages"])  # leaving the with closes the connection: a loop stops there
        return json.loads(r.read().decode("utf-8"))


def healthy():
    try:
        return _get("/health", timeout=2).get("status") == "ok"
    except Exception:
        return False


def template_file():
    """The chat template to pass, or None for the model's own"""
    t = CFG["template"]
    if t == "model":
        return None
    if t == "auto":
        patched = TEMPLATES / (Path(CFG["model"]).stem + ".jinja")
        return patched if patched.exists() else None
    return Path(t)


def server_cmd():
    tf = template_file()
    return [
        str(server_binary()),
        "-m", str(CFG["model"]),
        "--host", CFG["host"],
        "--port", str(CFG["port"]),
        "-a", CFG["alias"],
        "-ngl", "999",
        "-c", str(CFG["ctx"]),
        "-np", str(CFG["parallel"]),
        "-kvu",  # the slots share one context pool: one long request can use all of it while the others are idle
        "-fa", "on",
        "-ctk", CFG["cache_type"],
        "-ctv", CFG["cache_type"],
        "-t", str(CFG["threads"]),
        "--jinja",
        *(["--chat-template-file", str(tf)] if tf else []),
        "--reasoning-format", "deepseek",
        "--metrics",
        *(["--spec-type", CFG["spec"]] if CFG["spec"] else []),
        "--log-file", str(SERVER_LOG),
        *CFG["extra"],
    ]


def nvidia_smi(query):
    """One value per GPU from nvidia-smi, or None without an NVIDIA GPU"""
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines() if r.returncode == 0 else None
    except Exception:
        return None


def power_limit():
    v = nvidia_smi("power.limit")
    try:
        return float(v[0]) if v else None
    except ValueError:
        return None


THERMAL_BITS = 0x20 | 0x40  # SW / HW thermal slowdown


def thermal_slowdown():
    """True while the GPU slows itself down for heat. On an RTX 3090 under an LLM it comes with the core cool and
    the fan at 100 %: the GDDR6X at its limit, which an LLM loads harder than a game does."""
    v = nvidia_smi("clocks_event_reasons.active")
    try:
        return bool(v) and bool(int(v[0], 16) & THERMAL_BITS)
    except ValueError:
        return False


def cool_down(max_wait=90):
    if not CFG["thermal_pause"]:
        return 0.0
    t0 = time.time()
    while thermal_slowdown() and time.time() - t0 < max_wait:
        time.sleep(3)
    return time.time() - t0


def _started(pid):
    """When the process with this pid started, as a value that only has to compare equal, or None when there is no
    such process (or no way to tell here). The pid file keeps it next to the pid: a pid alone is not a process. Once
    the server is gone (a crash, a reboot) the system hands its pid to another one, and `stop` kills what it names."""
    if WINDOWS:
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32")
        times = [wintypes.FILETIME() for _ in range(4)]
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k.GetProcessTimes.argtypes = (wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4)
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            code = wintypes.DWORD()
            if not k.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
                return None
            if not k.GetProcessTimes(h, *(ctypes.byref(t) for t in times)):
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            k.CloseHandle(h)
    try:  # Linux: the 22nd field of /proc/PID/stat, in clock ticks since the boot
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        pass
    try:  # macOS, the BSDs
        r = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
        return " ".join(r.stdout.split()) or None
    except Exception:
        return None


def _read_pid():
    """(pid, ours) from the pid file. ours: True while that pid is still the server this worker started, False when
    it is gone or is another process now, None when the file has no start time to tell by (an older worker wrote
    it). (None, False) without a pid file."""
    try:
        pid, _, started = PID_FILE.read_text().strip().partition(" ")
        pid = int(pid)
    except (OSError, ValueError):
        return None, False
    if not started:
        return pid, None
    return pid, str(_started(pid)) == started


def _launch():
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "server.stdout.log", "w", encoding="utf-8") as out:
        kw = dict(stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True)
        if WINDOWS:
            flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
            try:  # outlive the caller's job object (Claude Code's shell) when it allows it
                p = subprocess.Popen(server_cmd(), creationflags=flags | 0x01000000, **kw)  # BREAKAWAY_FROM_JOB
            except OSError:
                p = subprocess.Popen(server_cmd(), creationflags=flags, **kw)
        else:
            p = subprocess.Popen(server_cmd(), start_new_session=True, **kw)
    started = _started(p.pid)
    PID_FILE.write_text(f"{p.pid} {started}" if started is not None else str(p.pid))
    return p


def _wait_ready(timeout, p=None):
    """p: the server just launched. None: one found already starting, the pid file's."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p is not None and p.poll() is not None:
            print(f"llama-server exited with {p.returncode}; see {LOGS / 'server.stdout.log'}")
            return False
        if p is None and not _read_pid()[1]:
            print(f"the server that was starting is gone; see {LOGS / 'server.stdout.log'}")
            return False
        if healthy():
            return True
        time.sleep(1)
    print(f"not ready after {timeout} s; see {LOGS / 'server.stdout.log'}"
          + ("" if p is not None else ". If it hangs there, `stop` ends it and `start` launches a new one"))
    return False


LATE_SYSTEM_RAISE = re.compile(r"\{\{-?\s*raise_exception\(\s*['\"]System message must be at the beginning\.?['\"]\s*\)\s*-?\}\}")
LATE_SYSTEM_RENDER = "{{- '<|im_start|>system\\n' + content + '<|im_end|>' + '\\n' }}"


def fix_template():
    """Qwen3's chat template raises on a system message after the first one, and Claude Code sends them (its
    reminders): the agent's first request fails with HTTP 500. The patched copy renders it in place as a system
    turn. Returns the patched file, or None when the running model's template needs no patch."""
    try:
        tpl = _get("/props").get("chat_template", "")
    except Exception:
        return None
    if not LATE_SYSTEM_RAISE.search(tpl):
        return None
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    out = TEMPLATES / (Path(CFG["model"]).stem + ".jinja")
    patched = LATE_SYSTEM_RAISE.sub(lambda m: LATE_SYSTEM_RENDER, tpl)
    patched += ("\n{#- cranker-llm: a system message after the first one is rendered in place instead of raising "
                "(Claude Code sends them) #}\n")
    out.write_text(patched, encoding="utf-8", newline="")
    return out


def cmd_start(args):
    if healthy():
        print(f"already running at {URL}")
        return 0
    limit = power_limit()
    cap = CFG["max_power_w"]
    if cap and limit is not None and limit > cap and not args.ignore_power_limit:
        print(f"GPU power limit is {limit:.0f} W, above max_power_w ({cap} W) in {CONFIG_FILE}. "
              f"Lower it from an admin shell (nvidia-smi -pl <watts>), or start --ignore-power-limit")
        return 3
    if not CFG["model"] or not Path(CFG["model"]).exists():
        print(f"model missing: '{CFG['model']}'. Run scripts/setup.py, or set \"model\" in {CONFIG_FILE}")
        return 1
    if not server_binary().exists():
        print(f"llama-server missing: '{server_binary()}'. Run scripts/setup.py, or set \"llama_server\" in "
              f"{CONFIG_FILE}")
        return 1
    # one start at a time, across sessions: two that both found nothing running each launched a server, and the
    # second one's pid file left the first server with nobody to stop it
    t0 = time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = open(START_LOCK, "a+")
    noted = False
    try:
        while True:
            try:
                _try_lock(lock)
                break
            except OSError:
                if time.time() - t0 > args.timeout:
                    print(f"another start still has not finished after {args.timeout} s; see "
                          f"{LOGS / 'server.stdout.log'}")
                    return 1
                if not noted:
                    print("another session is starting the server: waiting for it", file=sys.stderr)
                    noted = True
                time.sleep(1)
        try:
            return _start(args, t0)
        finally:
            _unlock(lock)
    finally:
        lock.close()


def _start(args, t0):
    if healthy():  # the start this one waited for
        print(f"already running at {URL}")
        return 0
    pid, ours = _read_pid()
    p = None
    if ours:  # its own start was cut short while the model loaded: a second server would load the model again
        print(f"a server is already starting (pid {pid}): waiting for it", file=sys.stderr)
    else:
        p = _launch()
    if not _wait_ready(args.timeout, p):
        return 1
    if CFG["template"] == "auto" and template_file() is None:
        patched = fix_template()
        if patched:
            print(f"the model's chat template refuses a late system message (Claude Code sends them): "
                  f"patched copy {patched}; restarting with it")
            _stop()
            time.sleep(2)
            if not _wait_ready(args.timeout, _launch()):
                return 1
    print(f"ready at {URL} in {time.time() - t0:.0f} s (pid {_read_pid()[0]}); {_props_line()}")
    return 0


def _stop():
    pid, ours = _read_pid()
    PID_FILE.unlink(missing_ok=True)
    if pid is None:
        return "not running"
    if ours is False:  # never kill by a pid alone: it is another process's by now
        return f"was not running (the server this worker started, pid {pid}, is gone)"
    if WINDOWS:
        r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        return (r.stdout or r.stderr).strip() or "stopped"
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return "stopped"
    except ProcessLookupError:
        return "was not running"


def cmd_stop(args):
    ours = _read_pid()[1]
    print(_stop())
    if ours is False and healthy():
        print(f"a server still answers at {URL}: this worker did not start it (no pid file for it), so it is not "
              "this worker's to stop")
        return 1
    return 0


def _props_line():
    try:
        p = _get("/props")
        n_ctx = p.get("default_generation_settings", {}).get("n_ctx")
        return f"model {Path(p.get('model_path', '?')).name}, n_ctx {n_ctx}, slots {p.get('total_slots')}"
    except Exception as e:
        return f"(no props: {e})"


def _vram_line():
    v = nvidia_smi("memory.used,memory.total")
    return f"VRAM {v[0].replace(', ', ' of ')} MiB" if v else ""


def _remote_line():
    """The server elsewhere, when one is configured: where, which model, from which folders, and whether its key is
    set (never the key)"""
    r = {**REMOTE_DEFAULTS, **(CFG.get("remote") or {})}
    if not r["url"]:
        return
    key = "no key" if not r["key_env"] else f"key in {r['key_env']} ({'set' if remote_key(r) else 'not set'})"
    roots = ", ".join(map(str, r["roots"])) or "no folder listed yet: nothing can be sent"
    spill = ("; spill on: a request the local worker has no room for goes there by itself" if r["spill"] else "")
    roles = ", ".join(f"{name} {v if isinstance(v, str) else (v or {}).get('model')}"
                      for name, v in (r["roles"] or {}).items())
    roles = (f"; roles: {roles}" + (f"; escalate: {r['escalate']} after {r['escalate_after']} failed answer"
                                    + ("s" if r["escalate_after"] != 1 else "") if r["escalate"] else "")
             if roles else "")
    print(f"remote (--remote): {r['model']} at {urlsplit(r['url']).hostname}; {key}; files may leave from: {roots}"
          + spill + roles)


def cmd_status(args):
    _remote_line()
    if not healthy():
        print(f"not running ({URL}). {_vram_line()}")
        return 1
    print(f"running at {URL}; {_props_line()}; {_vram_line()}")
    used, busy = used_tokens()
    if busy:
        print(f"busy: {busy} requests running, {used} tokens of the context held (an agent started now waits for "
              "them, and then makes every other task wait for it)")
    if agent_running():
        print(f"taken: {agent_holder()} has the worker; asks and batches wait for it")
    return 0


def cmd_config(args):
    print(f"home   {HOME}\nconfig {CONFIG_FILE} ({'found' if CONFIG_FILE.exists() else 'missing: defaults'})")
    print(f"server {server_binary()}")
    print(json.dumps(CFG, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------- shared context

def _try_lock(f):
    """A non-blocking exclusive lock on the file (its first byte on Windows); OSError if another holds it"""
    if WINDOWS:
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(f):
    try:
        if WINDOWS:
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


_AGENT_PROBE = threading.Lock()


def agent_running():
    """True while another process's agent holds AGENT_LOCK (see agent_turn). One probe at a time in a process: a
    batch's threads probing together took each other's test lock for an agent's."""
    with _AGENT_PROBE:
        try:
            LOGS.mkdir(parents=True, exist_ok=True)
            f = open(AGENT_LOCK, "a+")
        except OSError:
            return False
        try:
            _try_lock(f)
        except OSError:
            f.close()
            return True
        _unlock(f)
        f.close()
        return False


def agent_note(state, timeout, task):
    """The agent leaves its name where the tasks that wait for it can read it (agent_holder): they waited half an
    hour and failed without a word on whose agent had the worker. The lock itself cannot hold it: on Windows a
    locked byte is not readable."""
    note = {"session": os.environ.get("CLAUDE_CODE_SESSION_ID"), "cwd": os.getcwd(), "pid": os.getpid(),
            "since": time.time(), "timeout": timeout, "state": state, "task": " ".join(task.split())[:70]}
    try:
        AGENT_INFO.write_text(json.dumps(note), encoding="utf-8")
    except OSError:
        pass


def agent_holder():
    """Whose agent has the worker, for a message: 'an agent of session 601d636d (vs-cheviman\\saturn-fullset; 12
    min in, 33 min left at most; "Write a test for...")'. Only means something while agent_running(). 'an agent'
    alone when there is no note, or it is too old to be this agent's (one that was killed leaves its note)."""
    try:
        a = json.loads(AGENT_INFO.read_text(encoding="utf-8"))
        age = time.time() - a["since"]
        if age > 2 * a["timeout"] + 60:
            return "an agent"
        sid = a.get("session")
        if sid and sid == os.environ.get("CLAUDE_CODE_SESSION_ID"):
            who = "this session's own agent"
        elif sid:
            who = f"an agent of session {sid[:8]}"
        else:
            who = f"an agent started outside Claude Code, pid {a.get('pid')}"
        about = [project_of(a)]
        if a.get("state") == "running":
            about.append(f"{_span(age)} in, {_span(max(0, a['timeout'] - age))} left at most")
        else:
            about.append("waiting for the server to be idle")
        if a.get("task"):
            about.append(f"\"{a['task']}...\"")
        return f"{who} ({'; '.join(about)})"
    except (OSError, ValueError, KeyError, TypeError):
        return "an agent"


def _span(seconds):
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


class WorkerError(Exception):
    pass


class WorkerBusy(WorkerError):
    """The request was never sent: the worker was taken for the whole wait. Nothing is wrong with the task."""


def needed_tokens(body):
    """What a chat request takes of the context: its messages (the server's tokenizer) and its answer at most"""
    text = "\n".join(m["content"] for m in body["messages"])
    try:
        return len(_post("/tokenize", {"content": text}, timeout=120)["tokens"]) + 64 + body.get("max_tokens", 0)
    except Exception:
        return None


ANSWER_MARGIN = 4096  # what a running request may still add when the server does not say (n_remain -1); else all of
# it: capped at 4096, a batch of four thinking tasks (7,000 tokens each) grew past a 64K pool and every request failed


def used_tokens():
    """The context the running requests hold, and what they may still add: (tokens, requests)"""
    total = busy = 0
    try:
        for s in _get("/slots"):
            if not s.get("is_processing"):
                continue
            busy += 1
            nt = s.get("next_token")
            nt = (nt[0] if nt else {}) if isinstance(nt, list) else (nt or {})
            remain = nt.get("n_remain", -1)
            total += ((s.get("n_prompt_tokens") or 0) + (nt.get("n_decoded") or 0)
                      + (ANSWER_MARGIN if remain is None or remain < 0 else remain))
    except Exception:
        pass
    return total, busy


ADMIT_LOCK = LOGS / "admit.lock"
ADMIT_MARGIN = 1024  # context kept free beyond the running requests' answers: speculative drafts take a few tokens
ADMIT_SEEN = 30      # s a request holds the admission until the server shows it running (/slots)
_ADMIT = threading.Lock()
# What a full pool fails the running requests with (HTTP 500), retried like "no room". With the pool full, the
# server halves its batch down to a few tokens, and the ngram drafts then fall outside it: "speculative batch
# index 8 is not inside the current sub-batch [0, 8)"
POOL_FULL = ("context", "speculative batch index", "free space in the kv cache", "memory slot")


class _Admission:
    """One request at a time, among this process's threads and the other worker processes, from the room check
    until the server runs it. /slots shows a request only once the server takes it, so a batch's threads that
    all found the server idle in the same second went in together: 88K of answers on a 64K pool."""

    def __enter__(self):
        _ADMIT.acquire()
        self.f = None
        try:
            LOGS.mkdir(parents=True, exist_ok=True)
            self.f = open(ADMIT_LOCK, "a+")
            while True:
                try:
                    _try_lock(self.f)
                    break
                except OSError:
                    time.sleep(0.2)
        except Exception:
            if self.f:
                self.f.close()
            self.f = None  # the thread lock alone
        return self

    def __exit__(self, *exc):
        if self.f:
            _unlock(self.f)
            self.f.close()
        _ADMIT.release()


def _running_tasks():
    """The tasks the server runs and has counted (their prompt known), or None if /slots does not answer"""
    try:
        return {s.get("id_task") for s in _get("/slots") if s.get("is_processing") and s.get("n_prompt_tokens")}
    except Exception:
        return None


def _post_seen(path, body):
    """Sends the request and returns its future once the server runs it, it ended, or ADMIT_SEEN s passed"""
    before = _running_tasks()
    fut = concurrent.futures.Future()

    def send():  # a daemon thread: Ctrl+C does not wait for the answer
        try:
            fut.set_result(_post(path, body))
        except BaseException as e:
            fut.set_exception(e)

    threading.Thread(target=send, daemon=True).start()
    t0 = time.time()
    while not fut.done() and time.time() - t0 < ADMIT_SEEN:
        now = _running_tasks()
        if now is None or before is None or now - before:
            break
        time.sleep(0.2)
    return fut


def post_with_room(path, body, spill=False):
    """The slots share one context pool (-kvu). The server does not turn a request away when the pool is full:
    decoding fails and every running request fails with it (HTTP 500, POOL_FULL). So a request is sent only where
    the running ones leave room for it (counted with its answer at most) and ADMIT_MARGIN more, one at a time
    (_Admission), else it waits. One over the whole context is an error at once: split it, use line ranges or lower
    --max-tokens. spill: the request may go elsewhere (remote.spill), so where it would wait, or the server is not
    there, or it is over the context, LocalFull says why at once."""
    ctx = CFG["ctx"]
    need = needed_tokens(body)
    if need is not None and need > ctx - ADMIT_MARGIN:
        if spill:
            raise LocalFull(f"{need} tokens with the answer, over the local context ({ctx})")
        raise WorkerError(f"{need} tokens with the answer, over the context ({ctx}) alone: "
                          f"split it, use line ranges or lower --max-tokens")
    t0 = time.time()
    noted = False
    while True:
        fut = None
        with _Admission():
            used, busy = used_tokens()
            agent = agent_running()
            if agent:
                error = f"{agent_holder()} has the worker (an agent takes the context alone)"
            elif busy == 0 or need is None or need + used <= ctx - ADMIT_MARGIN:
                fut = _post_seen(path, body)
            else:
                error = f"no room: {need} tokens needed, {used} held by {busy} running requests"
        if fut is not None:
            try:
                return fut.result()
            except (urllib.error.HTTPError, StreamError) as e:
                # a streamed request whose decoding fails once it has started says so in the stream, not in its status
                code = "in the stream" if isinstance(e, StreamError) else f"HTTP {e.code}"
                try:
                    text = e.text if isinstance(e, StreamError) else e.read().decode("utf-8", "replace")
                except Exception:
                    text = str(e)
                if not any(k in text.lower() for k in POOL_FULL):
                    raise WorkerError(f"{code}: {text[:500]}")
                error = f"{code}: {text[:300]}"
            except (OSError, http.client.HTTPException) as e:  # the connection, not an answer: the server went away
                if spill:
                    raise LocalFull(f"the local server did not answer ({type(e).__name__})")
                raise WorkerError(f"the server did not answer ({type(e).__name__}: {e}). `status` says whether it "
                                  f"still runs; its log is {SERVER_LOG}")
        if spill:
            raise LocalFull(error)
        if time.time() - t0 > CFG["context_wait"]:
            if fut is not None:
                raise WorkerError(f"{error} -- still no room after {CFG['context_wait']} s")
            raise WorkerBusy(f"not run: {error}, still after {_span(CFG['context_wait'])} of waiting. Nothing is "
                             "wrong with the task: run it again when "
                             + ("that agent ends, or ask its session" if agent else "the worker has room"))
        if not noted:
            print(f"waiting, up to {_span(CFG['context_wait'])}: {error}", file=sys.stderr)
            noted = True
        time.sleep(15)


# ------------------------------------------------------------------------------------------------------- usage

_USAGE_WRITE = threading.Lock()
_server_model = None


def server_model():
    """The .gguf the server runs (once per process)"""
    global _server_model
    if _server_model is None:
        try:
            _server_model = Path(_get("/props").get("model_path", "")).name or "?"
        except Exception:
            _server_model = "?"
    return _server_model


def count_tokens(text):
    try:
        return len(_post("/tokenize", {"content": text}, timeout=60)["tokens"])
    except Exception:
        return None


def log_usage(cmd, tokens_in, tokens_out, answer, seconds, where=None, **extra):
    """One line per request in usage.jsonl: 'in' is what the model read, 'out' what it wrote (its thinking too),
    'answer' what comes back to the caller. CRANKER_USAGE_TAG marks runs that are not work (benchmarks). where: the
    server elsewhere the request went to (None: this machine's), never its key; its model, and its role if any."""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd, "cwd": os.getcwd(),
           "model": where["model"] if where else server_model(),
           "in": tokens_in or 0, "out": tokens_out or 0, "answer": answer if answer is not None else tokens_out or 0,
           "s": round(seconds, 1), **extra}
    if where:
        rec["remote"] = where["host"]  # not this machine's model: usage tells them apart, with what it cost
        if where.get("role"):
            rec["role"] = where["role"]
    tag = os.environ.get("CRANKER_USAGE_TAG")
    if tag:
        rec["tag"] = tag
    session = os.environ.get("CLAUDE_CODE_SESSION_ID")  # the calling Claude Code session (scripts/local_share.py)
    if session:
        rec["session"] = session
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with _USAGE_WRITE, open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def remote_prices():
    """USD per million tokens (in, out) by model: remote.price for the server's own model, a role's "price" for its
    model. The usage of a model with no price set counts as free"""
    conf = {**REMOTE_DEFAULTS, **(CFG.get("remote") or {})}
    prices = {conf["model"]: conf["price"]} if conf.get("model") and conf.get("price") else {}
    for v in (conf.get("roles") or {}).values():
        if isinstance(v, dict) and v.get("model") and v.get("price"):
            prices[v["model"]] = v["price"]
    return prices


def cmd_usage(args):
    """What went through the local model, by command, project and day. 'kept off Claude' is in - answer: the files
    and prompts the model read, less the answers that come back. For the agent it is an upper bound: its 'in' counts
    Claude Code's own prompt again on every turn."""
    if not USAGE_LOG.exists():
        print(f"nothing logged yet ({USAGE_LOG})")
        return 0
    session = args.session
    if session == "":
        print("--session: not called from a Claude Code session (CLAUDE_CODE_SESSION_ID is not set): give its id")
        return 1
    every = []
    for line in USAGE_LOG.read_text(encoding="utf-8").splitlines():
        try:
            every.append(json.loads(line))
        except ValueError:
            continue
    mark_tries(every)  # over the whole log: a task sent again today is one more try of yesterday's
    recs = []
    for r in every:
        if args.since and r["ts"][:10] < args.since:
            continue
        if r.get("tag") and not args.all:
            continue
        if session and not str(r.get("session", "")).startswith(session):
            continue
        recs.append(r)
    only = f" by session {session[:8]}" if session else ""
    if not recs:
        print(f"nothing logged{only}" + (f" since {args.since}" if args.since else "")
              + ("" if args.all else " (--all adds the benchmarks)"))
        return 0

    def table(title, key):
        rows = {}
        for r in recs:
            a = rows.setdefault(key(r), [0, 0, 0, 0, 0.0, 0])
            a[0] += 1
            a[1] += r["in"]
            a[2] += r["out"]
            a[3] += r["answer"]
            a[4] += r["s"]
            a[5] += 0 if run_failed(r) else r["in"] - r["answer"]
        print(f"\n{title:<34} {'requests':>8} {'in':>11} {'out':>9} {'answer':>9} {'kept off Claude':>16} {'hours':>6}")
        for k, (n, i, o, ans, s, kept) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
            print(f"{str(k)[:34]:<34} {n:>8} {i:>11,} {o:>9,} {ans:>9,} {kept:>16,} {s / 3600:>6.1f}")
        if len(rows) > 1:
            n, i, o, ans, s, kept = (sum(v[j] for v in rows.values()) for j in range(6))
            print(f"{'total':<34} {n:>8} {i:>11,} {o:>9,} {ans:>9,} {kept:>16,} {s / 3600:>6.1f}")

    print(f"{USAGE_LOG}: {len(recs)} requests{only} from {recs[0]['ts']} to {recs[-1]['ts']}"
          + ("" if args.all else " (benchmarks left out: --all)"))
    table("command", lambda r: r["cmd"] + (" (remote)" if r.get("remote") else "")
          + (f" [{r['tag']}]" if r.get("tag") else ""))
    table("project (the caller's folder)", project_of)
    table("day", lambda r: r["ts"][:10])
    cost_table([r for r in recs if not run_failed(r)], project_of)
    quality_table(args.since, args.all, session)
    if not session:
        guard_line(args.since)
    lost = [r for r in recs if run_failed(r)]
    if lost:
        spent = sum(r["in"] + r["out"] for r in recs)
        gone = sum(r["in"] + r["out"] for r in lost)
        why = {}
        for r in lost:
            why[not_effective(r)] = why.get(not_effective(r), 0) + 1
        print(f"\nconsumed {spent:,} tokens (in + out), effective {spent - gone:,}: {len(lost)} runs were not "
              f"effective ({', '.join(f'{n} {w}' for w, n in why.items())}), {gone:,} tokens and "
              f"{sum(r['s'] for r in lost) / 3600:.1f} hours. They were spent all the same, and count as consumed "
              "('in', 'out', 'hours'); they do not count as kept off Claude or saved")
    pieces = pieces_line(recs)
    if pieces:
        print("\n" + pieces)
    away = [r for r in recs if r.get("remote")]
    if away:
        prices = remote_prices()
        cost = sum(r["in"] * prices.get(r["model"], (0, 0))[0] + r["out"] * prices.get(r["model"], (0, 0))[1]
                   for r in away) / 1e6
        models = sorted({r["model"] for r in away})
        one = prices.get(models[0]) if len(models) == 1 else None
        print(f"\nremote: {len(away)} of these requests went to a server elsewhere "
              f"({', '.join(sorted({str(r['remote']) for r in away}))}), {sum(r['in'] for r in away):,} tokens in and "
              f"{sum(r['out'] for r in away):,} out: ${cost:,.2f} at the "
              + (f"price in the config (${one[0]:g} in / ${one[1]:g} out per million). What they read left this "
                 "machine" if one else "prices in the config. What they read left this machine. By model:"))
        for model in [] if one else models:
            rows = [r for r in away if r["model"] == model]
            roles = sorted({r["role"] for r in rows if r.get("role")})
            i, o = sum(r["in"] for r in rows), sum(r["out"] for r in rows)
            price = prices.get(model)
            print(f"  {model}" + (f" (as {', '.join(roles)})" if roles else "") + f": {len(rows)} requests, {i:,} in, "
                  f"{o:,} out, " + (f"${(i * price[0] + o * price[1]) / 1e6:,.2f} at ${price[0]:g} / ${price[1]:g}"
                                    if price else "no price in the config (remote.price, or the role's \"price\")"))
    if any(r["cmd"] == "agent" for r in recs):
        print("\nagent: 'in' counts Claude Code's own prompt on every turn, so its 'kept off Claude' is an upper bound")
    return 0


def work_id(out, files):
    """The piece of work a request is: its output and the files it is about, without their line ranges, so that the
    same file sent again in a narrower range, or with another prompt, is the same work done again"""
    paths = sorted(os.path.normcase(os.path.abspath(m.group(1) if (m := FILE_RANGE.match(f)) else f))
                   for f in files or [])
    return hashlib.sha256(json.dumps([os.path.normcase(os.path.abspath(out)), paths]).encode("utf-8")).hexdigest()[:12]


def tries_asked():
    """By which try a piece of work has to be right to be effective (scale.tries in the config: the second)"""
    return int((CFG.get("scale") or {}).get("tries", 2))


def _wrong(r):
    if r.get("stopped") or r.get("error") or not r.get("answer"):
        return "gave nothing back"  # an agent the worker stopped or that ended in an error, an empty answer
    if r.get("repeat"):
        return "went round in a loop"
    if r.get("cut"):
        return "were cut"
    if r.get("check") == "fail":
        return "failed their check"
    return None


def mark_tries(rows):
    """Two yardsticks, kept apart. Consumption: every run's tokens are spent and counted, and a task run four times
    is four runs. Quality: a piece of work (work_id: an output and its files) is effective when its output was right
    by the first or second try (tries_asked); one that took more, or was left wrong, is not, whatever it consumed.
    Marks each run with its piece's tries and with why it is not effective (None when it is)."""
    most = tries_asked()
    pieces = {}
    for r in rows:
        if r.get("work"):
            pieces.setdefault(r["work"], []).append(r)
    for runs in pieces.values():
        why = _wrong(runs[-1]) or (f"were of work that took more than {most} tries" if len(runs) > most else None)
        for r in runs:
            r["_tries"], r["_why"] = len(runs), why


def not_effective(r):
    """Why a run is not effective work, or None when it is. A run with no piece of work to its name (an answer that
    was only printed) is judged by itself."""
    return r.get("_why") if "_tries" in r else _wrong(r)


def run_failed(r):
    return not_effective(r) is not None


def pieces_line(recs):
    """The quality yardstick in a line: the pieces of work by the try they were right at"""
    most = tries_asked()
    pieces = {}
    for r in recs:
        if r.get("work"):
            pieces[r["work"]] = r  # its last run among these
    if not pieces:
        return None
    first = sum(1 for r in pieces.values() if not r["_why"] and r["_tries"] == 1)
    by = sum(1 for r in pieces.values() if not r["_why"] and r["_tries"] > 1)
    more = sum(1 for r in pieces.values() if r["_why"] and r["_tries"] > most and not _wrong(r))
    wrong = len(pieces) - first - by - more
    return (f"quality, by piece of work (an output and the files it is about): {len(pieces)} pieces, {first} right at "
            f"the first try, {by} by try {most}, {more} took more tries, {wrong} left wrong (cut, looping, failing "
            f"their check, nothing back). Effective is the first two: right by try {most}")


# USD per million tokens (input, output): Anthropic's API list prices on 2026-11-03. Cache reads are $0.20 on both.
PRICES = {"Opus 5.5": (4.00, 20.00), "Fable 5.2": (8.00, 40.00)}


def project_of(r):
    return os.sep.join(Path(r.get("cwd", "?")).parts[-2:])


def quality_table(since=None, include_tagged=False, session=None):
    """The tasks that came with a check: how many were right the first time, how many the worker fixed itself and
    how many were left for Claude to fix. The rework Claude does on outputs without a check is not seen here."""
    if not QUALITY_LOG.exists():
        return
    recs = []
    for line in QUALITY_LOG.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (since and r["ts"][:10] < since) or (r.get("tag") and not include_tagged):
            continue
        if session and not str(r.get("session", "")).startswith(session):
            continue
        recs.append(r)
    if not recs:
        return
    rows = {}
    for r in recs:
        a = rows.setdefault(project_of(r), {"pass": 0, "fixed": 0, "fail": 0, "tries": 0})
        a[r["result"]] += 1
        a["tries"] += r["tries"]
    print(f"\n{'checked tasks (the rework)':<34} {'tasks':>8} {'first time':>11} {'by worker':>10} "
          f"{'for Claude':>11} {'fixes':>6}")
    for k, a in sorted(rows.items(), key=lambda kv: -sum(kv[1][x] for x in ("pass", "fixed", "fail"))):
        n = a["pass"] + a["fixed"] + a["fail"]
        print(f"{str(k)[:34]:<34} {n:>8} {a['pass'] / n:>10.0%} {a['fixed'] / n:>10.0%} {a['fail'] / n:>11.0%} "
              f"{a['tries']:>6}")
    print("first time: passed its check with no fix. by worker: the worker fixed it from the check's errors. "
          "for Claude: still\nfailing after its retries. fixes: the worker's extra requests.")


def guard_line(since=None):
    """The read guard's refusals (scripts/read_guard.py, the plugin's PreToolUse hook on Read)"""
    log = LOGS / "read_guard.jsonl"
    if not log.exists():
        return
    hits = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            h = json.loads(line)
            if not since or h["ts"][:10] >= since:
                hits.append(h)
        except (ValueError, KeyError, TypeError):  # a line cut short by a hook that was killed
            continue
    if hits:
        print(f"\nread guard: {len(hits)} whole-file reads of big files sent to the worker instead "
              f"({sum(h['lines'] for h in hits):,} lines in all, {len({h['path'] for h in hits})} files)")


def cost_table(recs, key):
    """What the same work would have cost on Claude at API list prices. read: the tokens kept off Claude
    (in - answer) at the input price. +write: also the answers at the output price, as if Claude had written them
    (names, comments, drafts); for maps and summaries it would only have read the files, so the truth lies between.
    Left out: Claude Code sends what it has read again on every later turn (at the cache-read price), so a file read
    early in a long session costs more than once. On a subscription the same tokens come off its usage limits."""
    rows = {}
    for r in recs:
        a = rows.setdefault(key(r), [0, 0])
        a[0] += r["in"] - r["answer"]
        a[1] += r["answer"]
    rows["total"] = [sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values())]
    cols = [(f"{m} read", lambda kept, ans, p=p: kept * p[0] / 1e6) for m, p in PRICES.items()]
    cols += [(f"{m} +write", lambda kept, ans, p=p: kept * p[0] / 1e6 + ans * p[1] / 1e6) for m, p in PRICES.items()]
    print(f"\n{'USD saved at API list prices':<34}" + "".join(f" {c:>16}" for c, _ in cols))
    for k, (kept, ans) in sorted(rows.items(), key=lambda kv: (kv[0] == "total", -kv[1][0])):
        print(f"{str(k)[:34]:<34}" + "".join(f" {f(kept, ans):>16,.2f}" for _, f in cols))
    print("read: what the model read instead of Claude, less the answers Claude reads back, at the input price. "
          "+write: also\nthe answers at the output price, as if Claude had written them. Not counted: Claude Code "
          "sends what it read\nagain on every later turn. Prices per million tokens: "
          + ", ".join(f"{m} ${i:.2f} in / ${o:.2f} out" for m, (i, o) in PRICES.items()) + ".")


# ----------------------------------------------------------------------------------------------------- ask, batch

FILE_RANGE = re.compile(r"^(.*):(\d+)-(\d+)$")


NUMBERED = "every line starts with its number in the file and a tab, which are not part of the file"


def read_file_spec(spec, number=False):
    """number: every line with its number in the file in front of it. A model asked for 'name (line N)' counts the
    lines by eye and gets them wrong over a few hundred; with the number there it copies it. Not for a rewrite: the
    numbers would come back in the code."""
    m = FILE_RANGE.match(spec)
    path, a, b = (m.group(1), int(m.group(2)), int(m.group(3))) if m else (spec, None, None)
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    label = path if a is None else f"{path} (lines {a}-{b})"
    if a is not None or number:
        lines = text.splitlines()[(a or 1) - 1:b]
        if number:
            lines = [f"{n}\t{line}" for n, line in enumerate(lines, a or 1)]
            label += f" ({NUMBERED})"
        text = "\n".join(lines)
    return f"=== FILE: {label} ===\n{text}\n=== END FILE ===\n"


# The thinking counts inside max_tokens, and Qwen thinks a lot: two diagnoses with 14,000 tokens spent them all
# thinking and came back empty. llama-server takes a thinking budget per request (reasoning_budget_tokens): past it,
# it closes the thinking with BUDGET_MESSAGE and the model answers in the room left.
ANSWER_ROOM = 2048
BUDGET_MESSAGE = "\n\nThe thinking budget is used up: stop thinking and give the answer now."
EFFORTS = ("xhigh", "medium", "low")  # Qwen3.8's template (reasoning_effort); xhigh is its default
RECOVER = ("\n\n=== YOUR REASONING ON THIS, CUT OFF BEFORE YOU ANSWERED ===\n{thinking}\n=== END OF REASONING ===\n"
           "Your reasoning above was cut off before you answered. From it, give the answer now, as the instructions "
           "above ask. Where it did not reach a conclusion, say what is still uncertain instead of guessing.")


def think_budget(max_tokens):
    return max(256, max_tokens - min(ANSWER_ROOM, max_tokens // 3))


# ----- a server elsewhere (--remote): any OpenAI-compatible endpoint, for a run the caller asks to send there

REMOTE = None  # set by use_remote: this run's requests go to the "remote" server of the config, not the local one
REMOTE_DEFAULTS = {
    "url": "",                        # its base URL, up to /v1
    "model": "",                      # the model's name there
    "key_env": "CRANKER_REMOTE_KEY",  # the environment variable its key is read from ("" = no key); never the config
    "roots": [],                      # the folders files may leave this machine from: none until they are listed
    "parallel": 2,                    # a batch's requests at once (hosted plans limit them)
    "wait": 300,                      # s a request waits while the server limits it (HTTP 429) before it is not run
    "extra": {},                      # fields for every request
    "plain": {},                      # fields for a request that should not think, e.g. {"reasoning_effort": "none"}
    "think": None,                    # fields for one that thinks; None: {"reasoning_effort": the effort}
    "price": [0, 0],                  # USD per million tokens, in and out, for usage
    "spill": False,                   # with the local worker full, a request that may leave goes there by itself
    "stream": True,                   # the answer as it is written, so a loop is stopped (LoopWatch); False: a
                                      # server that refuses stream or stream_options
    "roles": {},                      # {"draft": "glm-5.3", "review": {"model": ..., "price": [...]}, "x": "local"}
    "escalate": None,                 # the role a checked task's fixes go to once it failed (--escalate)
    "escalate_after": 1,              # how many failed answers before that: 1, its first fix (its second try)
}
EFFORT_ELSEWHERE = {"xhigh": "high", "medium": "medium", "low": "low"}  # OpenAI's names for Qwen's efforts
# in a limited request's error: the plan's quota or the account, which no wait of a few minutes gets past
QUOTA_WORDS = ("quota", "limit_reached", "card_required", "billing", "credit")
LOCAL_HELP = "keep this run on the local worker, even with remote.spill on"
ROLE_HELP = "a role of remote.roles: the model the user chose for this kind of task (draft, review...)"
ESCALATE_HELP = "with a check: the role its fixes go to once it failed (default: remote.escalate; '' none)"
REMOTE_HELP = ("send this run to the server under \"remote\" in the config instead of the local one: the files leave "
               "this machine, so only from the folders under remote.roots, and only when the user asks for it")


def _where(p):
    """Where a path really is: a link inside a listed folder that points out of it is out of it"""
    return os.path.normcase(os.path.realpath(os.path.expanduser(str(p))))


REMOTE_HOW = f"{CONFIG_FILE}: \"remote\": {{\"url\": \"https://.../v1\", \"model\": \"...\", \"roots\": [\"...\"]}}"


class NotUnderRoots(WorkerError):
    """A file that may not leave this machine (may_leave)"""


def may_leave(paths, roots):
    """Refuses any of these that is not under one of remote.roots. It is asked of every file the worker reads into
    a remote request, not only the -f ones: the prompt, system and prefix files, a batch's task file and the brief
    go out too."""
    roots = [_where(p) for p in roots]
    for p in paths:
        p = m.group(1) if (m := FILE_RANGE.match(str(p))) else p
        if not any(_where(p) == root or _where(p).startswith(root.rstrip(os.sep) + os.sep) for root in roots):
            raise NotUnderRoots(f"--remote: {p} is not under remote.roots. Files leave this machine only from the "
                                f"folders the user listed there ({REMOTE_HOW})")


def remote_key(r):
    """The server's key: its environment variable, or on Windows the user's own environment, where setx writes it
    (a Claude Code started before the setx, and every command it runs, does not have it). Never the config.
    CRANKER_USER_ENV=0 reads the process's environment only."""
    if not r["key_env"]:
        return None
    key = os.environ.get(r["key_env"])
    if not key and WINDOWS and os.environ.get("CRANKER_USER_ENV") != "0":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, r["key_env"])[0]
        except OSError:
            key = None
    return key or None


# ----- remote.roles: Claude orchestrates and each stage goes to the model the user chose for it (20-11-2026): the
# draft to one, its review to another of another family, which errs in other things, the volume to the local one.
# A role is a model's name on the remote server, an object that also sets its own fields there (a model that refuses
# "reasoning_effort": "none" needs its own "plain"; "price" for usage), or "local". Listing a role is the user's
# standing yes for the runs that name it, as spill is: no --remote, but the files still leave only from remote.roots.
LOCAL = "local"
ROLE_FIELDS = ("model", "plain", "think", "extra", "price")
ROLES_HOW = (f"{CONFIG_FILE}: \"remote\": {{..., \"roles\": {{\"draft\": \"glm-5.3\", \"review\": {{\"model\": "
             "\"kimi-k3\", \"plain\": {}, \"price\": [2.66, 13.31]}, \"volume\": \"local\"}}")


def role_of(name):
    """What remote.roles says of a role: LOCAL, or the fields of the remote server it sets (its model at least)"""
    roles = (CFG.get("remote") or {}).get("roles") or {}
    if name not in roles:
        listed = ", ".join(roles) or "none listed"
        raise WorkerError(f"no role {name!r} in remote.roles ({listed}): {ROLES_HOW}")
    v = roles[name]
    if v == LOCAL:
        return LOCAL
    fields = {"model": v} if isinstance(v, str) else dict(v) if isinstance(v, dict) else {}
    if not fields.get("model") or not isinstance(fields["model"], str):
        raise WorkerError(f"remote.roles.{name}: a model's name, \"local\", or an object with its \"model\": "
                          f"{ROLES_HOW}")
    return {k: fields[k] for k in ROLE_FIELDS if k in fields}


def server_for(role, files):
    """Where a task with this role goes: None (no role: as ever), LOCAL, or the remote server with the role's model
    and fields, once the caller's folder and every file of the task may leave (remote.roots)"""
    if not role:
        return None
    if role_of(role) == LOCAL:
        return LOCAL
    return remote_server(files, role)


def remote_server(files=None, role=None):
    """The server under "remote" in the config, with its key and host, for a run that sends there: refused (with
    why) when it has no URL or model, is not https, a file (files, with the caller's folder) is not under
    remote.roots, or its key is not set. role: one of remote.roles, whose fields (its model...) take the place of the
    server's own."""
    r = {**REMOTE_DEFAULTS, **(CFG.get("remote") or {})}
    if role:
        r.update(role_of(role), role=role)
    if not r["url"] or not r["model"]:
        raise WorkerError(f"--remote: no server to send to. Set it in {REMOTE_HOW}")
    parts = urlsplit(r["url"])
    if parts.scheme != "https" and parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise WorkerError(f"--remote: {r['url']} is not https: the files and the key would cross the network in the "
                          "clear")
    if files is not None:
        may_leave([os.getcwd(), *(f for f in files if f)], r["roots"])
    r["key"] = remote_key(r)
    if r["key_env"] and not r["key"]:
        raise WorkerError(f"--remote: the server's key is read from the environment variable {r['key_env']}, and "
                          "it is not set (the key is never written in the config)")
    r["host"] = parts.hostname
    return r


def use_remote(files):
    """--remote: the run goes to the server under "remote" in the config instead of the local one. The worker's
    promise is that nothing leaves the machine, so this is never a default: it takes the flag, and the caller's
    folder and every file read into the run (may_leave) have to be under one of remote.roots, which is empty until
    the user lists folders. The key comes from an environment variable, so it is in no file and no log."""
    global REMOTE
    REMOTE = remote_server(list(files))


# ----- remote.spill: with the local worker full, a request that may leave goes to the server elsewhere by itself

class LocalFull(WorkerError):
    """The local worker has no room for the request now: post_with_room(spill=True) says so instead of waiting"""


SPILL = {"server": None, "off": None, "slots": None}  # the server, why it is off (once off, for the whole run)
_SPILL_LOCK = threading.Lock()
_ON = threading.local()  # this thread's task: .spill, the server it may go to (spill_for); .down, the local is not up


def spill_server():
    """The remote with remote.spill on, or None: spill is off, the server cannot be used (no key...), or a failure
    there turned it off for the rest of this run. Once per process: a batch's threads share one count of requests."""
    with _SPILL_LOCK:
        if SPILL["server"] is None and SPILL["off"] is None:
            r = {**REMOTE_DEFAULTS, **(CFG.get("remote") or {})}
            if not r["spill"] or not r["roots"]:
                SPILL["off"] = "remote.spill is off"
            else:
                try:
                    SPILL["server"] = remote_server()
                    SPILL["slots"] = threading.BoundedSemaphore(max(1, int(SPILL["server"]["parallel"])))
                except WorkerError as e:
                    SPILL["off"] = str(e)
        return None if SPILL["off"] else SPILL["server"]


def spill_for(files):
    """The server a run's requests may spill to: spill is on, and the caller's folder and every file read into
    them (the brief, the prompt and task files too) are under remote.roots. None: the run stays here, as ever."""
    r = spill_server()
    if r is None:
        return None
    try:
        may_leave([os.getcwd(), *(f for f in files if f)], r["roots"])
    except WorkerError:
        return None
    return r


def _spill_off(why):
    with _SPILL_LOCK:
        if SPILL["off"] is None:
            SPILL["off"] = why
            print(f"remote.spill off for the rest of this run, the requests wait for the local worker: {why}",
                  file=sys.stderr, flush=True)


def _chat_remote(body, r=None):
    """The same request, in OpenAI's own fields only: a hosted server refuses the ones llama.cpp adds. There is no
    shared context to count there, only its limits: a 429 is waited for (Retry-After) up to remote.wait. r: the
    server, --remote's (REMOTE) by default"""
    r = r or REMOTE
    asked = body.get("chat_template_kwargs") or {}
    out = {"model": r["model"], "messages": body["messages"], "max_tokens": body["max_tokens"],
           **{k: body[k] for k in ("temperature", "top_p") if k in body}, **r["extra"]}
    if asked.get("enable_thinking"):
        effort = EFFORT_ELSEWHERE[asked.get("reasoning_effort") or "xhigh"]
        out.update(r["think"] if r["think"] is not None else {"reasoning_effort": effort})
    else:
        out.update(r["plain"])
    if r.get("stream", True):  # OpenAI's own fields: the usage comes in the stream's last event
        out.update(stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(r["url"].rstrip("/") + "/chat/completions", data=json.dumps(out).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "User-Agent": "cranker-llm",
                                          **({"Authorization": "Bearer " + r["key"]} if r["key"] else {})})
    t0 = time.time()
    noted = False
    while True:
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                if "text/event-stream" in (resp.headers.get("Content-Type") or ""):
                    d = read_stream(resp, body["messages"])
                else:
                    d = json.loads(resp.read().decode("utf-8"))
            break
        except StreamError as e:
            raise WorkerError(f"{r['host']} failed the request while answering: {e.text[:400]}")
        except urllib.error.HTTPError as e:
            try:
                text = " ".join(e.read().decode("utf-8", "replace").split())[:400]
            except Exception:
                text = str(e)
            if e.code in (401, 403):
                raise WorkerError(f"{r['host']} refused the key in {r['key_env'] or 'no key'} (HTTP {e.code}: {text})")
            try:
                after = float(e.headers.get("Retry-After") or 10)
            except ValueError:
                after = 10.0
            spent = any(w in text.lower() for w in QUOTA_WORDS)  # a quota that is used up is not waited for
            if e.code not in (429, 502, 503, 504) or after > 120 or spent:
                raise WorkerError(f"HTTP {e.code} from {r['host']}: {text}"
                                  + (". That is a quota or the account, not a busy moment: waiting does not help"
                                     if spent else ""))
            if time.time() - t0 + after > r["wait"]:
                raise WorkerBusy(f"not run: {r['host']} kept limiting the request for {_span(r['wait'])} (HTTP "
                                 f"{e.code}: {text[:200]}). Nothing is wrong with the task: run it again later, or "
                                 "with fewer --jobs")
            if not noted:  # with what the server says: its reason is what tells a busy moment from a plan's limit
                print(f"waiting, up to {_span(r['wait'])}: {r['host']} limits the requests (HTTP {e.code}: "
                      f"{text[:200]})", file=sys.stderr)
                noted = True
            time.sleep(after)
        except (OSError, http.client.HTTPException, ValueError) as e:
            raise WorkerError(f"{r['host']} did not answer ({type(e).__name__}: {e})")
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = d.get("usage") or {}
    content = msg.get("content") if isinstance(msg.get("content"), str) else ""
    thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
    thinking = thinking if isinstance(thinking, str) else ""
    if not usage and d.get("pieces"):  # a stream stopped before its end has no counts, and no /tokenize: by length
        usage = {"prompt_tokens": sum(len(m.get("content") or "") for m in body["messages"]) // 4,
                 "completion_tokens": max(d["pieces"], (len(content) + len(thinking)) // 4)}
    written = usage.get("completion_tokens") or 0
    thought = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    seconds = time.time() - t0
    return {
        "where": r, "repeat": d.get("repeat"),
        "content": content, "thinking": thinking, "finish": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens") or 0, "completion_tokens": written, "seconds": seconds,
        "prompt_tps": None, "gen_tps": written / seconds if seconds else None,
        # the answer without its thinking, as the server counts it, or else by its length: there is no /tokenize
        "answer_tokens": (max(0, written - thought) if thought is not None
                          else written if not thinking else len(content) // 4),
    }


_REMOTE_SLOTS = {}  # the server elsewhere's requests at once (remote.parallel), for all the roles: one account
_REMOTE_SLOTS_LOCK = threading.Lock()


def _remote_slots(r):
    with _REMOTE_SLOTS_LOCK:
        if r["url"] not in _REMOTE_SLOTS:
            _REMOTE_SLOTS[r["url"]] = threading.BoundedSemaphore(max(1, int(r["parallel"])))
        return _REMOTE_SLOTS[r["url"]]


def _chat(body, server=None):
    """One request, to the local worker, or elsewhere: to a role's server (server: LOCAL, or the remote one with the
    role's model), with --remote, or with remote.spill when the local worker is full and this thread's task may leave
    (_ON.spill). The result's "where" is the server elsewhere, or None."""
    if server == LOCAL:
        return _chat_local(body)
    if server or REMOTE:
        with _remote_slots(server or REMOTE):
            return _chat_remote(body, server)
    there = getattr(_ON, "spill", None)
    if there is None:
        return _chat_local(body)
    t0 = time.time()
    noted = False
    while True:
        if SPILL["off"]:
            return _chat_local(body)  # a failure there: here, waiting as ever
        if getattr(_ON, "down", False):
            full = "the local server is not running"
        else:
            # a batch (_ON.share) runs on both at once: with a request of its own already running here, the next
            # one goes there while it has a free slot. The local GPU shares its speed among its requests (a map
            # alone at 29 tok/s, five at once at 8 to 15 each), the remote does not. One decision at a time
            # (_SHARE_LOCK): the batch's threads all start together.
            with _SHARE_LOCK:
                here_first = not getattr(_ON, "share", False) or _SHARE["here"] == 0
                if here_first:
                    _SHARE["here"] += 1
            if not here_first:
                res = _there(body, there)
                if res is not None:
                    return res
                if SPILL["off"]:
                    continue
                with _SHARE_LOCK:
                    _SHARE["here"] += 1
            try:
                return _chat_local(body, spill=True)
            except LocalFull as e:
                full = str(e)
            finally:
                with _SHARE_LOCK:
                    _SHARE["here"] -= 1
        res = _there(body, there)
        if res is not None:
            return res
        if SPILL["off"]:
            continue
        if time.time() - t0 > CFG["context_wait"]:
            raise WorkerBusy(f"not run: {full}, and {there['host']} had its {there['parallel']} requests (remote."
                             f"parallel) running, still after {_span(CFG['context_wait'])}. Nothing is wrong with the "
                             "task: run it again later")
        if not noted:
            print(f"waiting: {full}, and {there['host']} has its {there['parallel']} requests running",
                  file=sys.stderr, flush=True)
            noted = True
        time.sleep(1)


_SHARE = {"here": 0}  # this process's spill requests running on the local worker
_SHARE_LOCK = threading.Lock()


def _there(body, there):
    """The request on the server elsewhere if it has a free slot (remote.parallel): its result, or None when it has
    none, or when it failed, which turns spill off for the rest of the run (a quota, its limits, the network)"""
    if not SPILL["slots"].acquire(blocking=False):
        return None
    try:
        return _chat_remote(body, there)
    except (WorkerError, OSError, http.client.HTTPException, ValueError) as e:
        _spill_off(f"{there['host']}: {e}")
        return None
    finally:
        SPILL["slots"].release()


def _chat_local(body, spill=False):
    t0 = time.time()
    r = post_with_room("/v1/chat/completions", body, spill)
    choice = r["choices"][0]
    msg = choice["message"]
    t = r.get("timings") or {}
    usage = r.get("usage") or {}
    content, thinking = msg.get("content") or "", msg.get("reasoning_content") or ""
    if not usage and r.get("pieces"):  # a stream stopped before its end has no counts: the server's tokenizer
        usage = {"prompt_tokens": count_tokens("\n".join(m["content"] for m in body["messages"])) or 0,
                 "completion_tokens": count_tokens(thinking + content) or r["pieces"]}
    seconds = time.time() - t0
    written = usage.get("completion_tokens") or 0
    return {
        "where": None,
        "repeat": r.get("repeat"),
        "content": content,
        "thinking": thinking,
        "finish": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": written,
        "seconds": seconds,
        "prompt_tps": t.get("prompt_per_second"),
        "gen_tps": t.get("predicted_per_second") or (written / seconds if seconds else None),
    }


# ----- streaming, and an answer that goes round in a loop, stopped while it is written

class StreamError(WorkerError):
    """A streamed request that failed once its answer had started: the server says so in the stream (its status was
    200 by then). .text is what it said"""

    def __init__(self, text):
        super().__init__(text)
        self.text = text


def read_stream(resp, messages):
    """A streamed answer (server-sent events), put together in the shape of a whole one. LoopWatch reads the answer
    as it comes, and once it goes round in a loop the reading stops: the caller then leaves the response, which
    closes the connection, and that is how the server learns to drop the request. The finish_reason is then
    "repeat" and "repeat" says what repeated; "pieces" counts the events that brought text, for a stream stopped
    before the counts of its last event."""
    watch = LoopWatch(messages)
    content, thinking = [], []
    finish = repeat = None
    usage, timings, pieces = {}, {}, 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("error:"):  # llama-server's way to fail a stream that has started
            raise StreamError(line[6:].strip())
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            d = json.loads(data)
        except ValueError:
            continue
        if d.get("error"):
            raise StreamError(json.dumps(d["error"]))
        usage = d.get("usage") or usage
        timings = d.get("timings") or timings
        for c in d.get("choices") or []:
            delta = c.get("delta") or {}
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(thought, str) and thought:
                thinking.append(thought)
                pieces += 1
            text = delta.get("content")
            if isinstance(text, str) and text:
                content.append(text)
                pieces += 1
                repeat = repeat or watch.feed(text)
            finish = c.get("finish_reason") or finish
        if repeat:
            finish = "repeat"
            break
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": "".join(content),
                                                              "reasoning_content": "".join(thinking)}}],
            "usage": usage, "timings": timings, "repeat": repeat, "pieces": pieces}


# On 20-11-2026 a review (ask --remote, max_tokens 6000) wrote a few findings, then `sdk/tests/subdir_project.py:N:
# fine.` for every N from 136 down to 1, until max_tokens: the room went to the loop, and its line said "cut by
# max_tokens: raise --max-tokens", which would only have bought more of it. Measured on that review and on 151
# outputs of the worker (maps, reviews, searches, notes, C, C++ and Python drafts, test logs):
# - lines: among the last 30 lines with a letter in them, with their numbers aside, the loop had 1 different line,
#   and the fewest in any of the 151 was 10. LOOP_FORMS (3) also catches a loop of two or three lines in turn;
# - words: no output of the 151 repeated, word for word, passages of 32 words it had already written in more than a
#   quarter of any 800 words (0.25); one that writes the same paragraphs over and over repeats them all (1.0). A
#   passage of numbers and signs only (a table of zeros) does not count, there as in the measure.
# What the request's own messages hold, as many times as they hold it, does not count: a rewrite copies its file,
# the repetitions in it too.
LOOP_LINES, LOOP_FORMS = 30, 3
LOOP_WORDS, LOOP_SPAN, LOOP_SHARE = 32, 800, 0.9
_NUMBER = re.compile(r"0[xX][0-9a-fA-F]+|\d+")
_WORD = re.compile(r"\w+|[^\w\s]")
_LETTER = re.compile(r"[^\W\d_]")


class LoopWatch:
    """Reads an answer as it streams: feed() says what repeats once it goes round in a loop, else None"""

    def __init__(self, messages):
        given = "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))
        self.lines_given = collections.Counter(self.form(line) for line in given.splitlines())
        words = _WORD.findall(given)
        self.runs_given = collections.Counter(hash(tuple(words[i:i + LOOP_WORDS]))
                                              for i in range(len(words) - LOOP_WORDS + 1))
        self.line = self.pending = ""
        self.forms = collections.deque(maxlen=LOOP_LINES)
        self.last = collections.deque(maxlen=LOOP_WORDS)
        self.lettered = 0  # words in self.last that start with a letter
        self.written = set()
        self.span = collections.deque()
        self.repeated = 0

    @staticmethod
    def form(line):
        """A line as a loop is told by: with its numbers aside, as `file:N: fine.` for every N"""
        return _NUMBER.sub("#", line.strip())

    def feed(self, text):
        self.line += text
        *ended, self.line = self.line.split("\n")
        for line in ended:
            said = self._line(line)
            if said:
                return said
        self.pending += text
        cut = max(self.pending.rfind(c) for c in " \t\r\n")  # whole words only: a word may go on in the next piece
        if cut < 0:
            return None
        words, self.pending = _WORD.findall(self.pending[:cut]), self.pending[cut + 1:]
        for word in words:
            said = self._word(word)
            if said:
                return said
        return None

    def _line(self, line):
        form = self.form(line)
        if not _LETTER.search(form):  # blank, or numbers and signs only: a table, a brace
            return None
        if self.lines_given[form] > 0:
            self.lines_given[form] -= 1
            return None
        self.forms.append(form)
        if len(self.forms) < LOOP_LINES:
            return None
        counts = collections.Counter(self.forms)
        if len(counts) > LOOP_FORMS:
            return None
        top, times = counts.most_common(1)[0]
        return (f"{len(counts)} different line{'s' if len(counts) > 1 else ''} in its last {LOOP_LINES}, numbers "
                f"aside: \"{top[:80]}\" {times} times")

    def _word(self, word):
        if len(self.last) == LOOP_WORDS:  # the word that leaves the run
            self.lettered -= self.last[0][0].isalpha() or self.last[0][0] == "_"
        self.last.append(word)
        self.lettered += word[0].isalpha() or word[0] == "_"
        if len(self.last) < LOOP_WORDS:
            return None
        run = hash(tuple(self.last))
        if self.runs_given[run] > 0:
            self.runs_given[run] -= 1
            again = False
        else:  # a run of numbers and signs only (a table of zeros) repeats as tables do, as in the measure
            again = self.lettered > 0 and run in self.written
        self.written.add(run)
        self.span.append(again)
        self.repeated += again
        if len(self.span) > LOOP_SPAN:
            self.repeated -= self.span.popleft()
        if len(self.span) < LOOP_SPAN or self.repeated < LOOP_SHARE * LOOP_SPAN:
            return None
        return (f"{self.repeated / LOOP_SPAN:.0%} of its last {LOOP_SPAN} words repeat, word for word, passages it had "
                "already written")


# ----- the project's brief: how the user works there (style, rules), so it is not repeated in every prompt

BRIEF_NAME = Path(".cranker") / "brief.md"
BRIEF_WARN = 8000  # chars: the brief goes into every request and the context is one pool for all the slots
_BRIEF = {}


def find_brief(explicit=None):
    """The brief that goes with every request: --brief FILE, else CRANKER_BRIEF, else .cranker/brief.md in the
    caller's folder or the nearest one above it up to the repository's root (outside a repository, the caller's
    folder only). '' / 'none' turns it off. Returns (path, text) or (None, '')."""
    spec = explicit if explicit is not None else os.environ.get("CRANKER_BRIEF")
    if spec is not None:
        if spec.strip().lower() in ("", "none"):
            return None, ""
        path = Path(spec)
    else:
        # up to the repository's root and no further: a brief in a folder above it (the user's home, a drive's root)
        # is not that project's, and would steer every repo below it
        here = Path.cwd()
        dirs = [here]
        for d in [here, *here.parents]:
            if (d / ".git").exists():
                dirs = [here, *here.parents][:[here, *here.parents].index(d) + 1]
                break
        path = next((d / BRIEF_NAME for d in dirs if (d / BRIEF_NAME).is_file()), None)
        if path is None:
            return None, ""
    key = str(path.resolve())
    if key not in _BRIEF:
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except OSError as e:  # one that was asked for and is not there is an error, not a run without the rules
            raise WorkerError(f"the brief {path} cannot be read ({e.strerror}): fix --brief or CRANKER_BRIEF, or "
                              "pass --brief \"\" for none")
        if len(text) > BRIEF_WARN:
            print(f"the brief {path} is {len(text):,} chars: it goes into every request; keep it to the rules the "
                  f"code must follow", file=sys.stderr)
        _BRIEF[key] = text
    return path, _BRIEF[key]


def with_brief(system, brief):
    if not brief:
        return system
    return (f"{system}\n\n=== HOW THIS PROJECT WORKS (follow it in everything you write) ===\n{brief}\n"
            f"=== END ===")


# ----- the check: a command that says whether an output is right; the worker fixes its own output until it passes

CHECK_TAIL = 6000  # chars of the check's output that go back to the model
FIX_PROMPT = ("Your output above was saved to {out} and checked with `{cmd}`. The check is right and your output is "
              "wrong:\n\n```\n{log}\n```\n\n{again}Change your output so that it passes: work out from each error "
              "what the check expects, and fix the cause, not just the line it names. Leave alone what the errors do "
              "not touch. {form}")
FIX_FORM = ("Output the WHOLE result again, in the same form the instructions asked for, with nothing before or after "
            "it.")
# A code file's fix: on 20-11-2026 a Python draft came back twice as its block and a paragraph after it, and the
# paragraph went into the file; the second one reported a real fault of the machine the check ran on (a stale
# toolchain path). The block goes to the file and any text around it beside it (split_answer), so the paragraph
# costs the file nothing; it is asked for only when the cause is outside the file.
FIX_FORM_CODE = ("Output the WHOLE file again as one fenced code block and nothing before or after it: the block is "
                 "what is written to the file. Only if the check fails for a cause outside the file (a path, a tool or "
                 "a setting of the machine it runs on), say so in a short paragraph after the block: it is kept apart "
                 "from the file, for the one who reads the result.")
SAME_AGAIN = "Your last fix came back unchanged, so it failed the same way: this time change it. "
QUALITY_LOG = LOGS / "quality.jsonl"


def _decode(raw):
    """A check's output: UTF-8, else the console's code page (Windows tools such as MSVC print in it), so the
    model sees the real characters of the errors."""
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in (("oem", "mbcs") if WINDOWS else ()):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def run_check(cmd, out):
    """Runs the check with {out} replaced by the output's path, in the caller's folder. (passed, its output)."""
    line = cmd.replace("{out}", str(out))
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # a Python check prints in UTF-8
    try:
        r = subprocess.run(line, shell=True, capture_output=True, env=env, timeout=CFG["check_timeout"])
    except subprocess.TimeoutExpired:
        return False, f"the check took more than {CFG['check_timeout']} s"
    text = (_decode(r.stdout) + _decode(r.stderr)).strip()
    if len(text) > CHECK_TAIL:
        text = "[...]\n" + text[-CHECK_TAIL:]
    return r.returncode == 0, text or f"exit code {r.returncode}, no output"


def log_quality(task_id, out, check, result, tries, **extra):
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cwd": os.getcwd(), "id": task_id, "out": str(out),
           "check": check, "result": result, "tries": tries, **extra}
    session = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if session:
        rec["session"] = session
    if os.environ.get("CRANKER_USAGE_TAG"):
        rec["tag"] = os.environ["CRANKER_USAGE_TAG"]
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with _USAGE_WRITE, open(QUALITY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# A cut answer (finish 'length') is unfinished whatever its check says, and the check's errors on it are only about
# the missing end. Sent back as a fix with the same limit, it is cut again: on 17-11-2026 a CMake file of two parts
# came back "CHECK FAILED: the reply must hold the second file" four times, and nothing said it had been cut. So it
# is asked for again as it was, with RAISE times the room, once; cut again, the task needs splitting.
RAISE = 1.5


def answer_room(prompt, step, history):
    """The most max_tokens a request can have: what the context holds beside its prompt, with post_with_room's
    accounting (a request over it alone is refused there). None if the server cannot count it."""
    server = step.get("server")
    if server != LOCAL and (server or REMOTE):  # a server elsewhere: its context is not this machine's to count
        return None
    need = needed_tokens({"messages": build_messages(prompt, step["files"], step["system"], history,
                                                     step["number"])})
    return None if need is None else CFG["ctx"] - ADMIT_MARGIN - need


def run_checked(t, cmd, keep_thinking, brief):
    """One task (a batch line, or ask's arguments in the same shape): runs it, writes 'out', and with a 'check'
    runs the check and hands the model its own output and the check's errors until it passes or the retries run
    out. The result says how it ended: check = pass (first time) | fixed (by its retries) | fail (left for Claude,
    with the check's last output in out.check.txt). A cut answer counts as failing: its retry asks for it again as
    it was, with more max_tokens (RAISE, as far as the context holds it), once; 'raised' says from what to what, and
    'split' why it stops when even that is cut. t["_server"]: where its role sends it (server_for); t["_escalate"]:
    where its fixes go once it failed remote.escalate_after answers ('escalated' says to which role)."""
    out = Path(t["out"])
    args = dict(files=t.get("files", []), system=with_brief(t.get("system") or CFG["system"], brief),
                think=t.get("think", False), max_tokens=t.get("max_tokens"), temperature=t.get("temperature"),
                cmd=cmd, effort=t.get("effort"), number=bool(t.get("number")), log=False, server=t.get("_server"))
    work = work_id(out, t.get("files"))

    def logged(res, **how):
        """The request's line in usage.jsonl, written once its check has run: it says which piece of work it was
        and how its output came out, which is what the quality yardstick is counted from (mark_tries)"""
        log_usage(**res.pop("usage"), dest=str(out.resolve()), work=work, **how)  # "out" is the tokens written

    res = run_task(t["prompt"], **args)
    write_out(out, res, keep_thinking)
    # the task's own role and model, said in its line (after an escalation too); a local role's is this machine's
    own = ({"role": t["role"], "model": LOCAL} if t.get("_server") == LOCAL
           else {"role": res.get("role"), "model": res.get("model")})
    check = t.get("check")
    if not check:
        logged(res)
        res.update(own)
        return res
    retries = t.get("retries", CFG["check_retries"])
    note = out.with_name(out.name + ".check.txt")
    for old in round_logs(out):
        old.unlink(missing_ok=True)
    tries = fixes = 0
    totals = {k: res[k] for k in ("prompt_tokens", "completion_tokens", "seconds")}
    previous = None
    step, history = dict(args), None  # how the last answer was asked for: a cut one is asked for again the same way
    raised = split = escalated = None
    escalate, after = t.get("_escalate"), int((CFG.get("remote") or {}).get("escalate_after") or 1)
    while True:
        passed, log = run_check(check, out)
        cut = res["finish"] == "length"
        looped = res["finish"] == "repeat"
        logged(res, check="pass" if passed else "fail")  # the check's own word: a cut answer's line says "cut" too
        if not passed or cut or looped:  # every round that failed keeps its output, not only the last one
            round_log(out, tries + 1).write_text(f"$ {check}\n{log}\n", encoding="utf-8")
        # a loop is not fixed by handing it back (its tokens as the fix's history), nor by more room
        if looped or (passed and not cut) or tries >= retries:
            break
        if cut:
            if raised:
                split = f"after raising it from {raised[0]} for a first cut"
                break
            room = answer_room(t["prompt"], step, history)
            limit = int(res["max_tokens"] * RAISE) if room is None else min(int(res["max_tokens"] * RAISE), room)
            if limit <= res["max_tokens"]:
                split = f"and the context has no room to raise it beside the prompt ({room} tokens at most)"
                break
            raised = (res["max_tokens"], limit)
            # the same request with more room; with thinking, its budget stays and the room goes to the answer
            step = {**step, "max_tokens": limit, "think_tokens": res["think_tokens"] or None}
        else:
            fixes += 1
            same = res["content"].strip() == previous
            previous = res["content"].strip()
            form = FIX_FORM if is_document(out) else FIX_FORM_CODE
            history = [{"role": "assistant", "content": res["content"]},
                       {"role": "user", "content": FIX_PROMPT.format(out=out, cmd=check, log=log, form=form,
                                                                     again=SAME_AGAIN if same else "")}]
            if escalate and not escalated and fixes >= after and not same_server(escalate, step["server"]):
                # the fixes go to the role above: a model that errs in other things, and whose credits go only to
                # what the first one could not get right (from its second try by default, which is still effective)
                step = {**step, "server": escalate}
                escalated = (t["_escalate_role"], escalate["model"] if isinstance(escalate, dict) else LOCAL, fixes)
            if not step["think"] and (fixes >= 2 or same):
                # a plain fix that did not take: the next one thinks first (medium effort, so it does not spend
                # the whole budget thinking). The answer keeps the room it had and the thinking comes on top: with
                # the thinking default instead (10,000, 7,952 of it thinking), a 12,000-token task had 2,048 left
                # to write its whole file again in, and was cut at line 206 (17-11-2026)
                budget = think_budget(CFG["max_tokens_thinking"])
                limit = res["max_tokens"] + budget
                room = answer_room(t["prompt"], step, history)
                if room is not None and limit > room:
                    limit, budget = room, max(256, room - res["max_tokens"])
                step = {**step, "think": True, "effort": step["effort"] or "medium", "max_tokens": limit,
                        "think_tokens": budget}
        tries += 1
        res = run_task(t["prompt"], **step, history=history, attempt=tries)
        for k in totals:
            totals[k] += res[k]
        write_out(out, res, keep_thinking, answer=tries + 1)
    res.update(totals)
    cut = res["finish"] == "length"
    res["check"] = ("pass" if tries == 0 else "fixed") if passed and not cut and not looped else "fail"
    res.update(tries=tries, raised=raised, split=split, escalated=escalated, **own)
    rounds = round_logs(out)
    if res["check"] != "fail":
        note.unlink(missing_ok=True)
        for old in rounds:  # what passed in the end leaves only itself, as before
            old.unlink(missing_ok=True)
    else:
        head = (repeat_line(res) if looped else cut_line(res) if cut else
                f"max_tokens was raised {raised[0]} -> {raised[1]} after a cut; the check still fails:" if raised
                else "")
        more = (f"\nthe check's output of every round: {', '.join(p.name for p in rounds)}\n" if len(rounds) > 1
                else "")
        note.write_text((head + "\n" if head else "") + f"$ {check}\n{log}\n" + more, encoding="utf-8")
    log_quality(t.get("id"), out, check, res["check"], tries,
                **({"cut": res["limit"]} if cut else {}), **({"raised": list(raised)} if raised else {}),
                **({"split": True} if split else {}), **({"repeat": True} if looped else {}),
                **({"escalated": escalated[0]} if escalated else {}))
    return res


def same_server(a, b):
    """Whether two of run_task's servers are one model: LOCAL, a remote server, or None (the run's own)"""
    def model(s):
        return s["model"] if isinstance(s, dict) else REMOTE["model"] if s is None and REMOTE else LOCAL
    return model(a) == model(b)


def round_log(out, n):
    """The check's output for the n-th answer of a task that failed it (n from 1): only the last one was kept once,
    and the first failure of a run that ended failing (an MSYS2 cmake on the PATH, 20-11-2026) was lost"""
    return out.with_name(f"{out.name}.check.{n}.txt")


def round_logs(out):
    """The round_log files beside an output, in their order"""
    pattern = re.compile(re.escape(out.name) + r"\.check\.(\d+)\.txt")
    try:
        found = [(int(m.group(1)), p) for p in out.parent.iterdir() if (m := pattern.fullmatch(p.name))]
    except OSError:
        return []
    return [p for _, p in sorted(found)]


STREAM = {"stream": True, "stream_options": {"include_usage": True}}  # read as it is written: LoopWatch


def build_messages(prompt, files=(), system=None, history=None, number=False):
    parts = [read_file_spec(f, number) for f in files]
    parts.append(prompt)
    return ([{"role": "system", "content": system or CFG["system"]}, {"role": "user", "content": "\n".join(parts)}]
            + (history or []))


def run_task(prompt, files=(), system=None, think=False, max_tokens=None, temperature=None, cmd="ask", effort=None,
             history=None, attempt=0, number=False, log=True, think_tokens=None, server=None):
    """One request (server: where it goes, a role's: see _chat). history: more turns after the first (the model's
    last answer and what to fix in it). log=False:
    its line for usage.jsonl comes back in res["usage"] for the caller to write (run_checked, after the check).
    think_tokens: its thinking budget, when not think_budget(max_tokens): the room a cut answer gains goes to the
    answer, not to more thinking. The result carries the limits it ran with (max_tokens, think_tokens) and 'limit',
    the max_tokens of the request its answer came from (the recovery's, once recovered)."""
    if max_tokens is None:
        max_tokens = CFG["max_tokens_thinking"] if think else CFG["max_tokens"]
    sampling = dict(CFG["sampling"]["thinking" if think else "plain"])
    if temperature is not None:
        sampling["temperature"] = temperature
    kwargs = {"enable_thinking": bool(think)}
    if think and effort:
        kwargs["reasoning_effort"] = effort
    messages = build_messages(prompt, files, system, history, number)
    body = {"model": CFG["alias"], "messages": messages, "max_tokens": max_tokens, "chat_template_kwargs": kwargs,
            **STREAM, **sampling}
    if think:
        think_tokens = think_tokens or think_budget(max_tokens)
        body["reasoning_budget_tokens"] = think_tokens
        body["reasoning_budget_message"] = BUDGET_MESSAGE
    res = _chat(body, server)
    where = res.pop("where", None)
    res.update(max_tokens=max_tokens, think_tokens=think_tokens if think else 0, limit=max_tokens,
               remote=where["host"] if where else None, model=where["model"] if where else None,
               role=where.get("role") if where else None)
    if think and not res["content"].strip() and res["thinking"]:
        # still no answer (a server that ignores the budget): the thinking goes back as text and the model concludes
        # from it without thinking; the summary says so, to be checked
        again = _chat({"model": CFG["alias"], "max_tokens": CFG["max_tokens"],
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [*messages[:-1], {"role": "user", "content": messages[-1]["content"]
                                                     + RECOVER.format(thinking=res["thinking"])}],
                       **STREAM, **CFG["sampling"]["plain"]}, server)
        again.pop("where", None)
        res.update(content=again["content"], finish=again["finish"], repeat=again.get("repeat"), recovered=True,
                   limit=CFG["max_tokens"],
                   prompt_tokens=res["prompt_tokens"] + again["prompt_tokens"],
                   completion_tokens=res["completion_tokens"] + again["completion_tokens"],
                   seconds=res["seconds"] + again["seconds"], gen_tps=again["gen_tps"],
                   **({"answer_tokens": again["answer_tokens"]} if "answer_tokens" in again else {}))
    if "answer_tokens" in res:  # a server elsewhere: counted by _chat_remote
        answer = res.pop("answer_tokens")
    else:
        answer = count_tokens(res["content"]) if res["thinking"] else res["completion_tokens"]
    # "cut": the limit an answer cut by max_tokens ran into; True for any other finish that is not a whole answer,
    # with "repeat" for one stopped going round in a loop
    res["usage"] = dict(cmd=cmd, tokens_in=res["prompt_tokens"], tokens_out=res["completion_tokens"], answer=answer,
                        seconds=res["seconds"], files=len(files), where=where,
                        **({"recovered": True} if res.get("recovered") else {}),
                        **({"attempt": attempt} if attempt else {}),
                        **({"cut": res["limit"]} if res["finish"] == "length" else
                           {"cut": True, "repeat": True} if res["finish"] == "repeat" else
                           {"cut": True} if res["finish"] != "stop" else {}))
    if log:
        log_usage(**res.pop("usage"))
    return res


# What a failed task's output ends with. Claude's reflex with a wrong output is to fix it itself, which is the work
# the worker was for, and then to write off the worker for that kind of task: the line says what to do instead.
RESEND = ("next: send it again with something changed, not a fix by hand (unless it is a slip of a line or two): "
          "--think, more --max-tokens, a narrower task (line ranges, one function), a --check, or the rule or example "
          "the prompt missed. If the limit is the worker's own, tell the user what in cranker-llm would fix it. One "
          "failure is that task with those parameters, not what the worker can't do.")
# ... and a loop's, where more room is the wrong change
REPEAT_NEXT = ("next: don't raise --max-tokens: the room went to the loop, and more would only buy more of it. Send it "
               "again with something changed: a narrower task (fewer files, line ranges, one question), what to leave "
               "out said outright (\"only the defects; nothing for what is fine\"), or --think. What came before the "
               "loop is in the output, a draft like any other: check it before using any of it. One loop is that "
               "task with those parameters, not what the worker can't do.")


def repeat_line(res):
    """What an answer stopped going round in a loop says first (the task's line, and its .check.txt)"""
    return (f"REPEATING: the answer went round in a loop ({res['repeat']}) and was stopped there, "
            f"{res['completion_tokens']} tokens out of max_tokens {res['limit']}")


def cut_line(res):
    """What a cut answer says first (the task's line, and its .check.txt), before any check: a check's errors on an
    unfinished answer are about its missing end"""
    s = f"cut by max_tokens {res['limit']}"
    if res.get("think_tokens") and not res.get("recovered"):
        s += f" ({res['think_tokens']} of it a thinking budget)"
    if res.get("split"):
        return (f"{s}, {res['split']}: the task needs splitting, its answer does not fit in one reply (fewer files "
                "or functions per task, line ranges)")
    return s + ": the answer is unfinished, raise --max-tokens" + (" or think less" if res.get("think_tokens") else "")


def summary(res, out):
    s = (f"in {res['prompt_tokens']} tok ({res['prompt_tps'] or 0:.0f} tok/s), "
         f"out {res['completion_tokens']} tok ({res['gen_tps'] or 0:.1f} tok/s), {res['seconds']:.1f} s")
    if res.get("role"):
        s += f", {res['model']} as {res['role']}"
    elif res.get("remote") and not REMOTE:
        s += f", on {res['remote']} (remote.spill)"
    if res.get("recovered"):
        s += ", RECOVERED: its thinking ran out, the answer was concluded from it (check it)"
    lead = ""
    if res["finish"] == "length":
        lead = cut_line(res) + " "
    elif res["finish"] == "repeat":
        lead = repeat_line(res) + " "
    elif res["finish"] != "stop":
        s += f", FINISH={res['finish']}"
    raised = res.get("raised")
    raised = f" (max_tokens raised {raised[0]} -> {raised[1]} after a cut)" if raised else ""
    fixes = f"{res.get('tries')} fix{'' if res.get('tries') == 1 else 'es'}"
    if res.get("check") == "pass":
        s += ", check passed"
    elif res.get("check") == "fixed":
        s += f", check passed after {fixes} by the worker{raised}"
    elif res.get("check") == "fail":
        s += (f", {'left for Claude' if lead else 'CHECK FAILED'} after {fixes}{raised}"
              + (f": {out}.check.txt" if out else ""))
    if res.get("escalated"):
        role, model, n = res["escalated"]
        s += f", fixes from fix {n} on by {model} as {role} (escalated)"
    if res.get("notes"):
        s += f", the text around the code kept apart in {res['notes']}"
    if res.get("blocks"):
        s += f", {res['blocks']} fenced blocks: written as they came"
    return lead + (f"-> {out}: " if out else "") + s


# Asked for a file, the model often wraps the whole of it in one fenced block (```c ... ```). Written as it came, the
# fence lines went into janus.c, its check failed on line 1, and the fixes went to the fence, not to the code
# (18-11-2026). So an answer that is one fenced block and nothing else is written without the fence (unfence). And
# on 20-11-2026 a Python draft came back twice as its block and a paragraph after it ("What is still uncertain:
# ..."): the file kept the fence and the paragraph, and its check failed on line 1 and then on line 141, the
# paragraph. So when the output is a code file (not a document: is_document), an answer with exactly one fenced
# block and text before or after it has the block written to the file and the text beside it, in <out>.notes.md
# (split_answer): the text was worth reading, the second one reported a fault of the machine the check ran on.
# Two blocks or more are written as they came: which one is the file can't be told (two files asked for in one
# reply, a before and an after, the file and an example of its use), so choosing one would drop code without a
# word, and joining them would make a file nobody asked for. Their check fails on the first fence, and the fix
# asks for the whole file in one block.
FENCE_OPEN = re.compile(r"[ \t]*```[^`]*")
FENCE_CLOSE = re.compile(r"[ \t]*```[ \t]*")
FENCE_LINE = re.compile(r"^[ \t]*```", re.M)
DOCUMENTS = {"", ".md", ".markdown", ".mdx", ".rst", ".adoc", ".asciidoc", ".org", ".txt", ".text", ".log"}
CODE_NAMES = {"cmakelists.txt", "makefile", "gnumakefile", "dockerfile", "requirements.txt"}


def is_document(out):
    """An output that is prose (a map, a review, notes): the text around a code block is the answer there, so only an
    answer that is the block alone loses its fence. Any other output is a code file."""
    out = Path(out)
    return out.name.lower() not in CODE_NAMES and out.suffix.lower() in DOCUMENTS


def fenced(text):
    """An answer's fenced code blocks: (its lines, [(opening line, closing line), ...]), or None when a fence is left
    open or one stands inside a block (```c inside ```markdown): such an answer is not taken apart."""
    lines = text.splitlines(keepends=True)
    blocks, start = [], None
    for i, line in enumerate(lines):
        bare = line.rstrip("\r\n")
        if start is None:
            if FENCE_OPEN.fullmatch(bare):
                start = i
        elif FENCE_CLOSE.fullmatch(bare):
            blocks.append((start, i))
            start = None
        elif FENCE_LINE.match(bare):
            return None
    return None if start is not None else (lines, blocks)


def split_answer(text):
    """(what goes to a code file, the text around it, the answer's fenced blocks): with exactly one block, its inside
    (line ends as they came) and the text before and after it; otherwise the answer as it came and no text."""
    found = fenced(text)
    if not found:
        return text, "", 0
    lines, blocks = found
    if len(blocks) != 1:
        return text, "", len(blocks)
    (a, b), = blocks
    around = ("".join(lines[:a]).strip() + "\n\n" + "".join(lines[b + 1:]).strip()).strip()
    return "".join(lines[a + 1:b]), around, 1


def unfence(text):
    """The inside of an answer that is exactly one fenced code block (whitespace around it aside, with or without a
    language tag, no other fence inside); any other answer unchanged."""
    inside, around, blocks = split_answer(text)
    return inside if blocks == 1 and not around else text


def notes_file(out):
    return Path(out).with_name(Path(out).name + ".notes.md")


TASK_KEYS = ("prompt", "files", "system", "think", "effort", "max_tokens", "temperature", "check", "retries", "number",
             "role", "escalate")


def task_sig(t):
    """What a task asks for, as a short hash: a batch runs a failed task again only once this has changed"""
    asked = json.dumps({k: t.get(k) for k in TASK_KEYS}, sort_keys=True)
    return hashlib.sha256(asked.encode("utf-8")).hexdigest()[:16]


def failed_mark(out):
    """Next to an output a batch left cut or failing its check: {"why": "cut" | "check", "task": task_sig}.
    Without it a failed output counted as done, and the batch skipped the task even after its line was changed."""
    return Path(out).with_name(Path(out).name + ".failed.json")


def failed_before(out):
    try:
        mark = json.loads(failed_mark(out).read_text(encoding="utf-8"))
        return mark if isinstance(mark, dict) else None
    except (OSError, ValueError):
        return None


def write_out(out, res, keep_thinking, answer=1):
    """Writes an answer to its output: a code file gets the code and its notes file the text around it, each
    answer's under its own heading (answer: which of the task's answers it is; the first starts the notes again)"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failed_mark(out).unlink(missing_ok=True)  # a new output: the batch marks it again if this one fails too
    notes = notes_file(out)
    if answer == 1:
        notes.unlink(missing_ok=True)
    if is_document(out):
        text, around, blocks = unfence(res["content"]), "", 0
    else:
        text, around, blocks = split_answer(res["content"])
    out.write_text(text, encoding="utf-8")
    if around:
        with open(notes, "a", encoding="utf-8") as f:
            f.write(f"## answer {answer}\n\n{around}\n\n")
    res["notes"] = str(notes) if notes.exists() else None
    res["blocks"] = blocks if blocks > 1 else None
    if keep_thinking and res["thinking"]:
        out.with_name(out.name + ".thinking.md").write_text(res["thinking"], encoding="utf-8")


def _http_error(e):
    try:
        return f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}"
    except Exception:
        return str(e)


def _not_running():
    print(f"server not running at {URL}: py {Path(__file__)} start")
    return 1


def _same_file(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def check_out(out, files, what="-o"):
    """An output over one of the task's own files is refused: a cut or wrong answer would replace the source (a cut
    one left 'int ma' where a C file was), and a fix after a failed check would read it back as the file."""
    for spec in files or []:
        m = FILE_RANGE.match(spec)
        if _same_file(out, m.group(1) if m else spec):
            raise WorkerError(f"{what} is one of the task's own files ({out}): a cut or wrong answer would replace "
                              "the source. Write to another file and move it over once it is checked")


def load_tasks(path):
    """A batch's tasks, checked before any of them runs: a bad line is named by its number in the file."""
    tasks = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except ValueError as e:
            raise WorkerError(f"{path}, line {n}: not JSON ({e.msg}, column {e.colno})")
        if not isinstance(t, dict) or not isinstance(t.get("prompt"), str) or not isinstance(t.get("out"), str):
            raise WorkerError(f"{path}, line {n}: a task is an object with at least \"prompt\" and \"out\"")
        t["_line"] = n
        tasks.append(t)
    outs = {}
    for t in tasks:
        name = t.get("id", f"line {t['_line']}")
        key = os.path.normcase(os.path.abspath(t["out"]))
        if key in outs:
            raise WorkerError(f"{path}: tasks {outs[key]} and {name} write the same \"out\" ({t['out']}): one would "
                              "replace the other, and the next run would skip both")
        outs[key] = name
        check_out(t["out"], t.get("files"), f"{path}, task {name}: its \"out\"")
    return tasks


def escalation_of(t):
    """The role a checked task's fixes go to once it failed: its own "escalate", else remote.escalate; "" none"""
    name = t.get("escalate", (CFG.get("remote") or {}).get("escalate"))
    return name if name and t.get("check") else None


def place(t, files):
    """Resolves where a task's role sends it, and its escalation's, before anything runs: a role that is not
    listed, or files that may not leave for it (remote.roots), refuse the task with why. remote.escalate is for
    every checked task, and a task whose files may not leave keeps its fixes where it runs: an escalation sends its
    files, and the check's output with its errors, to the role's model. One the task names (--escalate, its
    "escalate") is refused instead, as a role is."""
    t["_server"] = server_for(t.get("role"), files)
    t["_escalate_role"] = escalation_of(t)  # the task's own, or remote.escalate's
    t["_escalate"] = None
    if t["_escalate_role"]:
        try:
            t["_escalate"] = server_for(t["_escalate_role"], files)
        except NotUnderRoots:
            if "escalate" in t:
                raise
            t["_escalate_role"] = None
    return t


def cmd_ask(args):
    try:
        far = bool(args.role) and role_of(args.role) != LOCAL
    except WorkerError as e:
        print(e)
        return 1
    if args.local and far:
        print(f"--local and --role {args.role}: that role is a model elsewhere ({role_of(args.role)['model']})")
        return 1
    could_spill = not args.remote and not args.local and not args.role and spill_server() is not None
    if args.remote:
        use_remote([*(args.file or []), args.prompt_file])
    elif not far and not could_spill and not healthy():
        return _not_running()
    prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig") if args.prompt_file else args.prompt
    if not prompt:
        print("no prompt (-p or --prompt-file)")
        return 1
    if args.check and not args.out:
        print("--check needs -o: the check reads the output from its file ({out})")
        return 1
    if args.out:
        check_out(args.out, args.file)
    brief_path, brief = find_brief(args.brief)
    if REMOTE and brief_path:
        may_leave([brief_path], REMOTE["roots"])
    if could_spill:
        _ON.spill = spill_for([*(args.file or []), args.prompt_file, brief_path])
        _ON.down = not healthy()
        if _ON.down and not _ON.spill:
            return _not_running()
    t = {"prompt": prompt, "files": args.file or [], "system": args.system, "think": args.think,
         "max_tokens": args.max_tokens, "temperature": args.temperature, "effort": args.effort,
         "out": args.out, "check": args.check, "number": args.number, "role": args.role}
    if args.retries is not None:
        t["retries"] = args.retries
    if args.escalate is not None:
        t["escalate"] = args.escalate
    try:
        place(t, [*(args.file or []), args.prompt_file, brief_path])
        if args.out:
            res = run_checked(t, "ask", args.keep_thinking, brief)
        else:
            res = run_task(prompt, t["files"], with_brief(args.system or CFG["system"], brief), args.think,
                           args.max_tokens, args.temperature, effort=args.effort, number=args.number,
                           server=t["_server"])
    except WorkerBusy as e:  # not run, as in batch and agent: nothing wrong with the task, the same command again
        print(e)
        return 2
    except WorkerError as e:
        print(e)
        return 1
    except urllib.error.HTTPError as e:
        print(_http_error(e))
        return 1
    if args.print or not args.out:
        print(res["content"])
    print(summary(res, args.out), file=sys.stderr if not args.out else sys.stdout)
    if res["finish"] == "stop" and res.get("check") != "fail":
        return 0
    print(REPEAT_NEXT if res["finish"] == "repeat" else RESEND, file=sys.stderr if not args.out else sys.stdout)
    return 2


def _as_they_end(futs):
    """as_completed, waking twice a second: on Windows a wait with no timeout does not see Ctrl+C until a task ends"""
    pending = set(futs)
    while pending:
        done, pending = concurrent.futures.wait(pending, timeout=0.5,
                                                return_when=concurrent.futures.FIRST_COMPLETED)
        yield from done


def cmd_batch(args):
    could_spill = not args.remote and not args.local and spill_server() is not None
    up = args.remote or healthy()
    roles = bool((CFG.get("remote") or {}).get("roles"))  # a task's role may send it elsewhere: seen below
    if not up and not could_spill and not roles:
        return _not_running()
    tasks = load_tasks(args.tasks)
    if args.remote:
        use_remote([args.tasks, args.system_file, args.prompt_prefix,
                    *(f for t in tasks for f in t.get("files") or [])])
    # what every task shares goes once on the command line, not in every line: a task's own value wins
    shared = {}
    if args.system_file:
        shared["system"] = Path(args.system_file).read_text(encoding="utf-8-sig").strip()
    if args.check:
        shared["check"] = args.check
    if args.retries is not None:
        shared["retries"] = args.retries
    if args.number:
        shared["number"] = True
    if args.role:
        shared["role"] = args.role
    if args.escalate is not None:
        shared["escalate"] = args.escalate
    tasks = [{**shared, **{k: v for k, v in t.items() if v is not None}} for t in tasks]
    if args.prompt_prefix:
        prefix = Path(args.prompt_prefix).read_text(encoding="utf-8-sig").strip()
        tasks = [{**t, "prompt": prefix + "\n\n" + t["prompt"]} for t in tasks]
    brief_path, brief = find_brief(args.brief)
    if REMOTE and brief_path:
        may_leave([brief_path], REMOTE["roots"])
    # an output that exists is done, unless the task left it cut or failing its check (failed_mark): that one runs
    # again once its task has changed, and is left alone, and said, while it has not
    todo, same = [], []
    for t in tasks:
        failed = None if args.force or not Path(t["out"]).exists() else failed_before(t["out"])
        if args.force or not Path(t["out"]).exists() or (failed and failed.get("task") != task_sig(t)):
            todo.append(t)
        elif failed:
            same.append(t)
    common = [args.tasks, args.system_file, args.prompt_prefix, brief_path]
    for t in todo:  # where each role sends its task, refused before anything runs
        name = t.get("id", t["out"])
        try:
            if args.local and t.get("role") and role_of(t["role"]) != LOCAL:
                raise WorkerError(f"--local, and its role {t['role']} is a model elsewhere")
            place(t, [*common, *(t.get("files") or [])])
        except WorkerError as e:
            raise WorkerError(f"task {name}: {e}")
    far = next((t["_server"] for t in todo if isinstance(t["_server"], dict)), None)
    # remote.spill: the tasks with no role whose files may all leave can go there when the local worker is full
    spills = {}
    if could_spill:
        spills = {id(t): spill_for([*common, *(t.get("files") or [])]) for t in todo if t["_server"] is None}
    there = next((r for r in spills.values() if r), None)
    if not up and not there and not far:
        return _not_running()
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo) - len(same)} already done, {len(todo)} to run"
          + (f"; brief {brief_path}" if brief else "")
          + (f"; {sum(1 for r in spills.values() if r)} of them may go to {there['host']} as well (remote.spill)"
             if there else ""))
    named = {}
    for t in todo:
        if t.get("role"):
            model = t["_server"]["model"] if isinstance(t["_server"], dict) else LOCAL
            named[(t["role"], model)] = named.get((t["role"], model), 0) + 1
    if named:
        print("roles: " + ", ".join(f"{n} as {role} ({model})" for (role, model), n in sorted(named.items())))
    if not up and (there or far):
        print("the local worker is not running: only the tasks that go elsewhere run")
    if same:
        print(f"{len(same)} failed before (cut, or failing their check) and their tasks have not changed, so they "
              f"are not run again: {', '.join(str(t.get('id', t['out'])) for t in same)}. Change them, or --force")
    t0 = time.time()
    failed = tokens = not_run = kept = ended = raised = split = loops = 0
    checks = {"pass": 0, "fixed": 0, "fail": 0}
    waits = []

    away = 0

    def one(t):
        _ON.spill, _ON.down, _ON.share = spills.get(id(t)), not up, True
        try:
            elsewhere = isinstance(t["_server"], dict)
            if not up and not _ON.spill and not elsewhere:
                raise WorkerBusy("not run: the local worker is not running, and this task's files may not leave "
                                 "(remote.roots). Start it (worker.py start) and run the batch again")
            # this machine's GPU is not the one working
            waited = 0 if (REMOTE and t["_server"] != LOCAL) or elsewhere or not up else cool_down()
            if waited >= 1:
                waits.append(waited)
            return run_checked(t, "batch", args.keep_thinking, brief)
        finally:
            _ON.spill = _ON.down = _ON.share = None

    # here and there at once: the local slots and the remote's requests (remote.parallel, one count for its roles)
    away_width = int((there or far or {}).get("parallel") or 0)
    width = (int(REMOTE["parallel"]) + (CFG["parallel"] if any(t["_server"] == LOCAL for t in todo) else 0)
             if REMOTE else CFG["parallel"] + away_width)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs or width)
    try:
        futs = {ex.submit(one, t): t for t in todo}
        for f in _as_they_end(futs):
            t = futs[f]
            ended += 1
            try:
                res = f.result()
                tokens += res["completion_tokens"] or 0
                looped = res["finish"] == "repeat"
                cut = res["finish"] != "stop"
                bad = res.get("check") == "fail"
                failed += cut or bad
                kept += cut or bad
                loops += looped
                if cut or bad:
                    why = "repeat" if looped else "cut" if cut else "check"
                    failed_mark(t["out"]).write_text(json.dumps({"why": why, "task": task_sig(t)}), encoding="utf-8")
                if res.get("check"):
                    checks[res["check"]] += 1
                raised += bool(res.get("raised"))
                split += bool(res.get("split"))
                away += bool(res.get("remote") and not REMOTE)
                tag = "LOOP " if looped else "CUT " if cut else "BAD " if bad else "ok  "
                print(f"{tag}{t.get('id', t['out'])}: {summary(res, t['out'] if bad or cut else None)}")
            except WorkerBusy as e:  # never sent: not a failure of the task, and nothing to change in it
                not_run += 1
                print(f"WAIT {t.get('id', t['out'])}: {e}")
            except urllib.error.HTTPError as e:
                failed += 1
                print(f"FAIL {t.get('id', t['out'])}: {_http_error(e)}")
            except Exception as e:
                failed += 1
                print(f"FAIL {t.get('id', t['out'])}: {e}")
    except KeyboardInterrupt:
        # the pool's threads are not daemons: left to end by themselves, the tasks still waiting would all run and
        # the process would end with the last of them. Leaving now closes the running requests' connections, which
        # is how the server learns to drop them.
        ex.shutdown(wait=False, cancel_futures=True)
        print(f"interrupted: {ended} of {len(todo)} tasks had ended. Run the same batch again for the rest: the "
              "finished ones are skipped", flush=True)
        os._exit(130)
    ex.shutdown()
    print(f"done: {len(todo) - failed - not_run} ok, {failed} failed or cut, "
          + (f"{not_run} not run, " if not_run else "") + f"{tokens} tokens out in {time.time() - t0:.0f} s"
          + (f"; paused {len(waits)} times for GPU heat ({sum(waits):.0f} s)" if waits else "")
          + (f"; {away} of them on {there['host']} (remote.spill)" if there else ""))
    if any(checks.values()):
        print(f"checks: {checks['pass']} passed the first time, {checks['fixed']} fixed by the worker, "
              f"{checks['fail']} still failing (their errors in <out>.check.txt, every round's in "
              "<out>.check.N.txt)")
    if raised or split:
        print(f"cuts: {raised} answers cut by max_tokens asked for again with more (x{RAISE:g}, as far as the "
              "context holds it)" + (f"; {split} need splitting, cut even so or with no room to raise "
                                     "(their CUT lines)" if split else ""))
    if loops:
        print(f"loops: {loops} answers went round in a loop and were stopped there (their LOOP lines)")
        print(REPEAT_NEXT)
    if failed > loops:
        print(RESEND)
    if kept:
        print(f"{kept} of them left an output (cut, looping, or failing its check). The same batch runs a task like "
              "that again once its line has changed (or send the changed ones in a file of their own); unchanged, it "
              "is skipped")
    if not_run:
        print(f"{not_run} tasks were not run: the worker was taken for the whole wait (see their WAIT lines for "
              "who had it). They did not fail: run the same batch again, the finished tasks are skipped")
    return 0 if failed == 0 and not_run == 0 and not same else 2


# ------------------------------------------------------------------------------------------------------- agent

def agent_env(url):
    # CLAUDE_CODE_USE_*: the caller's Bedrock or Vertex setting would send the agent there instead of to the relay
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_OAUTH", "CLAUDE_CODE_USE_", "AWS_BEARER"))}
    alias = CFG["alias"]
    env.update({
        "ANTHROPIC_BASE_URL": url,  # the relay to the local server (AgentRelay): nothing goes to Anthropic
        "ANTHROPIC_API_KEY": "local",
        "ANTHROPIC_MODEL": alias,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": alias,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": alias,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": alias,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # the local model's real window, which Claude Code does not know: it assumes a larger one, never compacts
        # and ends in an API error past the context; it compacts with room left for the compaction's own request
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(CFG["ctx"]),
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(CFG["ctx"] - 10000),
        # a turn's output, its thinking included: the relay caps the thinking at agent_think_budget
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(AGENT_MAX_OUTPUT),
    })
    return env


AGENT_MAX_OUTPUT = 8000
AGENT_STALL = 2  # turns in a row cut at AGENT_MAX_OUTPUT: the agent is stopped instead of thinking again from scratch


def kill_tree(p):
    """Ends a process and the ones it started (Claude Code runs rg and shells). Elsewhere than Windows it must have
    been started in a session of its own (start_new_session), so that its group is only its own tree."""
    if WINDOWS:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        return
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        p.kill()


class AgentRelay:
    """Between the agent's Claude Code and llama-server. Claude Code asks for adaptive thinking, which llama-server
    reads as no limit: before writing a file the model thought past its 8,000-token turn, was cut, lost that
    thinking and thought again from scratch, turn after turn (three agents, 25 to 45 minutes each, nothing
    written). Claude Code has no setting for it, so the relay gives every request a thinking budget, which the
    server honours, drops the thinking of earlier turns from the context, and counts the turns cut in a row."""
    TURN = 1024  # a request with fewer max_tokens is not a turn of the agent (a title, a probe): its cut is no stall

    def __init__(self, upstream, budget, stall_limit=AGENT_STALL):
        parts = urlsplit(upstream)
        self._host, self._port = parts.hostname, parts.port or 80
        self.budget = budget
        self.stall_limit = stall_limit
        self.cuts = 0          # turns in a row that ended at max_tokens
        self.stalled = False   # stall_limit of them: the caller stops the agent
        self.requests = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.url = None
        self._lock = threading.Lock()
        self._server = None

    def _rewrite(self, body):
        """(the request's body with the budget and without old thinking, its max_tokens); (body, 0) if not JSON"""
        try:
            obj = json.loads(body)
            if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
                return body, 0
        except ValueError:
            return body, 0
        max_tokens = obj.get("max_tokens") or AGENT_MAX_OUTPUT
        obj["thinking"] = {"type": "enabled", "budget_tokens": min(self.budget, think_budget(max_tokens))}
        turns = [m for m in obj["messages"] if isinstance(m, dict) and m.get("role") == "assistant"]
        for m in turns[:-1]:  # the last turn keeps its thinking: the model is still in it
            if isinstance(m.get("content"), list):
                kept = [b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))]
                if kept:
                    m["content"] = kept
        return json.dumps(obj, ensure_ascii=False).encode("utf-8"), max_tokens

    def _count(self, max_tokens, stop_reason, tok_in, tok_out):
        with self._lock:
            self.requests += 1
            self.tokens_in += tok_in
            self.tokens_out += tok_out
            if max_tokens < self.TURN:
                return
            if stop_reason == "max_tokens":
                self.cuts += 1
                self.stalled = self.stalled or self.cuts >= self.stall_limit
            else:
                self.cuts = 0

    def start(self):
        relay = self

        def usage_in(u):
            return sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                               "cache_creation_input_tokens"))

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # nothing printed: the caller reads the agent's JSON
                pass

            def _handle(self):
                length = self.headers.get("Content-Length")
                body = self.rfile.read(int(length)) if length else None
                messages = self.command == "POST" and self.path.split("?")[0] == "/v1/messages"
                max_tokens = 0
                if messages and body:
                    body, max_tokens = relay._rewrite(body)
                skip = ("host", "content-length", "connection", "accept-encoding", "transfer-encoding")
                headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
                if body is not None:
                    headers["Content-Length"] = str(len(body))
                conn = http.client.HTTPConnection(relay._host, relay._port, timeout=None)  # a turn takes minutes
                try:
                    conn.request(self.command, self.path, body=body, headers=headers)
                    resp = conn.getresponse()
                except (OSError, http.client.HTTPException) as e:
                    err = json.dumps({"type": "error", "error": {"type": "api_error",
                                                                 "message": f"cranker relay: {e}"}}).encode("utf-8")
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(err)
                    return
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")  # HTTP/1.0, ended by the close: a stream goes line by line
                self.end_headers()
                stream = "text/event-stream" in (resp.getheader("Content-Type") or "")
                stop_reason, tok_in, tok_out, whole = None, 0, 0, b""
                try:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                        if not stream:
                            whole += line
                        elif messages and line.startswith(b"data:"):
                            try:
                                ev = json.loads(line[5:])
                            except ValueError:
                                continue
                            if ev.get("type") == "message_start":
                                tok_in = usage_in((ev.get("message") or {}).get("usage") or {})
                            elif ev.get("type") == "message_delta":
                                stop_reason = (ev.get("delta") or {}).get("stop_reason") or stop_reason
                                tok_out = (ev.get("usage") or {}).get("output_tokens") or tok_out
                except (OSError, http.client.HTTPException):
                    pass  # the client or the server went away: what was copied is what there is
                finally:
                    conn.close()
                if not messages:
                    return
                if not stream:
                    try:
                        obj = json.loads(whole)
                        stop_reason = obj.get("stop_reason")
                        tok_in, tok_out = usage_in(obj.get("usage") or {}), (obj.get("usage") or {}).get(
                            "output_tokens") or 0
                    except (ValueError, AttributeError):
                        pass
                relay._count(max_tokens, stop_reason, tok_in, tok_out)

            do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = _handle

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._server.handle_error = lambda *a: None  # a client that hangs up mid-answer is not worth a traceback
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()


AGENT_APPEND = ("You are a worker another agent handed a task to. Be brief and concrete; cite files with line "
                "numbers. Only write files inside the current working directory.")


def agent_turn(wait):
    """One agent at a time: an agent's request alone (Claude Code's prompt, its tools and the files it read) is
    32-40K tokens; two do not fit in a 64K pool, fail each other's requests and retry over and over. The lock is
    held while the agent runs; the system lets it go when the process ends, however it ends. None: still taken
    after 'wait' s."""
    LOGS.mkdir(parents=True, exist_ok=True)
    f = open(AGENT_LOCK, "a+")
    t0 = time.time()
    noted = False
    while True:
        try:
            _try_lock(f)
            return f
        except OSError:
            if time.time() - t0 > wait:
                f.close()
                return None
            if not noted:
                print(f"waiting, up to {_span(wait)}: {agent_holder()} has the worker (two agents do not fit in "
                      "the context)", file=sys.stderr)
                noted = True
            time.sleep(5)


def cmd_agent(args):
    if not healthy():
        return _not_running()
    prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig") if args.prompt_file else args.prompt
    if not prompt:
        print("no prompt (-p or --prompt-file)")
        return 1
    exe = shutil.which("claude")  # before the waits below: not after half an hour of them
    if not exe:
        print("claude not found on PATH")
        return 1
    # it loads no CLAUDE.md (no setting sources): the project's brief is how it learns how the user works there
    _, brief = find_brief(args.brief)
    turn = agent_turn(args.timeout)  # held until this process ends
    if turn is None:
        print(f"not run: {agent_holder()} still has the worker after {_span(args.timeout)} of waiting. Nothing is "
              "wrong with the task: run it again when that agent ends, or ask its session")
        return 2
    agent_note("waiting", args.timeout, prompt)
    atexit.register(lambda: AGENT_INFO.unlink(missing_ok=True))  # ours: nobody else writes it while the lock is held
    # the agent takes the context alone: its requests grow with every file it reads, so it starts with the server
    # idle, and asks and batches wait while it runs (post_with_room, agent_running)
    t0 = time.time()
    noted = False
    while True:
        used, busy = used_tokens()
        if busy == 0:
            break
        if time.time() - t0 > args.timeout:
            print(f"not run: the server still busy after {_span(args.timeout)} of waiting ({used} tokens held by "
                  f"{busy} requests of other tasks). Nothing is wrong with the task: run it again when they end")
            return 2
        if not noted:
            print(f"{used} tokens held by {busy} requests: the agent waits for the server to be idle", file=sys.stderr)
            noted = True
        time.sleep(15)
    tools = ["Read", "Grep", "Glob"]
    allowed = ["Read", "Grep", "Glob"]
    cwd = Path(args.write).resolve() if args.write else (LOGS / "agent")
    cwd.mkdir(parents=True, exist_ok=True)
    if args.write:
        tools += ["Write", "Edit"]
        allowed += ["Edit(./**)", "Write(./**)"]
    # not --bare: it leaves only Read and Edit. No setting sources instead (no hooks, plugins or CLAUDE.md), no MCP
    # servers; the API key and base URL in agent_env keep every request on this machine.
    cmd = [exe, "-p", prompt, "--setting-sources", "", "--strict-mcp-config",
           "--model", CFG["alias"], "--output-format", "json",
           "--tools", ",".join(tools), "--allowedTools", " ".join(allowed),
           "--append-system-prompt", with_brief(AGENT_APPEND, brief)]
    for d in args.dir or []:
        cmd += ["--add-dir", d]
    # which piece of work this run is, for usage.jsonl: the same search sent again is one more try of it (mark_tries)
    piece = {"dest": str(Path(args.out).resolve()), "work": work_id(args.out, args.dir)} if args.out else {}
    relay = AgentRelay(URL, CFG["agent_think_budget"], stall_limit=AGENT_STALL)
    relay.start()
    agent_note("running", args.timeout, prompt)
    t0 = time.time()
    stopped = None
    try:
        p = subprocess.Popen(cmd, cwd=cwd, env=agent_env(relay.url), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", start_new_session=not WINDOWS)  # see kill_tree
        while True:
            try:
                stdout, stderr = p.communicate(timeout=2)
                break
            except subprocess.TimeoutExpired:
                if relay.stalled:
                    stopped = (f"stopped: {relay.cuts} turns in a row ended at the output cap ({AGENT_MAX_OUTPUT} "
                               f"tokens, thinking included) before it acted. What it has to write does not fit in "
                               f"one turn: one file per run, or write it with ask/batch and --check")
                elif time.time() - t0 > args.timeout:
                    stopped = f"timed out after {args.timeout} s"
                if stopped:
                    kill_tree(p)
                    stdout, stderr = p.communicate()
                    break
    finally:
        relay.stop()
    if stopped:
        log_usage("agent", relay.tokens_in, relay.tokens_out, 0, time.time() - t0, turns=relay.requests,
                  stopped="stall" if relay.stalled else "timeout", **piece)
        print(RESEND)
        print(f"agent {stopped} ({relay.requests} requests, {time.time() - t0:.0f} s, in {relay.tokens_in} tok, "
              f"out {relay.tokens_out} tok)")
        return 2
    try:
        d = json.loads(stdout)
    except Exception:
        print(f"agent failed (exit {p.returncode}): {(stderr or stdout)[:800]}")
        return 1
    result = d.get("result") or ""
    u = d.get("usage", {})
    tokens_in = ((u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                 + (u.get("cache_creation_input_tokens") or 0))
    error = (d.get("terminal_reason") or "error") if d.get("is_error") else None
    log_usage("agent", tokens_in, u.get("output_tokens"), count_tokens(result), time.time() - t0,
              turns=d.get("num_turns"), **({"error": error} if error else {}), **piece)
    # with an error the result is the error's text: it goes on the last line, where a caller that reads only the
    # tail of the output still sees why
    line = (f"turns {d.get('num_turns')}, {time.time() - t0:.0f} s, in {tokens_in} tok, "
            f"out {u.get('output_tokens')} tok, denials {len(d.get('permission_denials') or [])}"
            + (f", ERROR {error}: {' '.join(result.split())[:300]}" if error else ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(unfence(result), encoding="utf-8")
    if args.print or not args.out:
        print(result)
    if error:
        print(RESEND)
    print((f"-> {args.out}: " if args.out else "") + line)
    return 1 if error else 0


# -------------------------------------------------------------------------------------------------------- main

def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="cranker-llm: a local LLM as Claude Code's worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--ignore-power-limit", action="store_true", help="start even above max_power_w")
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("config", help="print the settings in effect")
    p = sub.add_parser("ask")
    p.add_argument("-p", "--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("-f", "--file", action="append", help="file, or file:first-last for a line range")
    p.add_argument("--number", action="store_true",
                   help="give the model every line with its number: for answers that cite lines, not for rewrites")
    p.add_argument("--remote", action="store_true", help=REMOTE_HELP)
    p.add_argument("--local", action="store_true", help=LOCAL_HELP)
    p.add_argument("-o", "--out")
    p.add_argument("--system")
    p.add_argument("--think", action="store_true", help="let the model think first (slower, better on hard tasks)")
    p.add_argument("--effort", choices=EFFORTS, help="with --think: how much it thinks (Qwen3.8's default: xhigh)")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--print", action="store_true", help="also print the answer when writing -o")
    p.add_argument("--keep-thinking", action="store_true", help="save the thinking next to -o")
    p.add_argument("--check", help="a command that passes (exit 0) when the output is right; {out} is its path")
    p.add_argument("--retries", type=int, help="how many times the worker fixes its output for the check")
    p.add_argument("--role", help=ROLE_HELP)
    p.add_argument("--escalate", help=ESCALATE_HELP)
    p.add_argument("--brief", help="the project's brief (default: .cranker/brief.md here or above; '' turns it off)")
    p = sub.add_parser("batch")
    p.add_argument("tasks")
    p.add_argument("--jobs", type=int, help="default: the server's slots")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-thinking", action="store_true")
    p.add_argument("--check", help="the check for every task without its own; {out} is the task's output")
    p.add_argument("--retries", type=int, help="how many times the worker fixes an output for the check")
    p.add_argument("--role", help=ROLE_HELP + " (for every task without its own \"role\")")
    p.add_argument("--escalate", help=ESCALATE_HELP)
    p.add_argument("--system-file", help="the system prompt for every task without its own")
    p.add_argument("--number", action="store_true", help="numbered lines for every task without its own \"number\"")
    p.add_argument("--remote", action="store_true", help=REMOTE_HELP)
    p.add_argument("--local", action="store_true", help=LOCAL_HELP)
    p.add_argument("--prompt-prefix", help="a file with what every task's prompt starts with (the shared rules)")
    p.add_argument("--brief", help="the project's brief (default: .cranker/brief.md here or above; '' turns it off)")
    p = sub.add_parser("agent", help="Claude Code on the local model, with tools")
    p.add_argument("--brief", help="the project's brief (default: .cranker/brief.md here or above; '' turns it off)")
    p.add_argument("-p", "--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--dir", action="append", help="a folder it may read (a repo root brings its CLAUDE.md)")
    p.add_argument("--write", help="the folder it works in and may write to (only there)")
    p.add_argument("-o", "--out")
    p.add_argument("--print", action="store_true")
    p.add_argument("--timeout", type=int, default=1200)
    p = sub.add_parser("usage", help="tokens through the local model, by command, project and day")
    p.add_argument("--since", help="YYYY-MM-DD")
    p.add_argument("--all", action="store_true", help="also the runs tagged with CRANKER_USAGE_TAG")
    p.add_argument("--session", nargs="?", const=os.environ.get("CLAUDE_CODE_SESSION_ID") or "",
                   help="only one Claude Code session's requests: the calling one, or the id given (its start)")
    args = ap.parse_args()
    try:
        return {"start": cmd_start, "stop": cmd_stop, "status": cmd_status, "config": cmd_config, "ask": cmd_ask,
                "batch": cmd_batch, "agent": cmd_agent, "usage": cmd_usage}[args.cmd](args)
    except WorkerError as e:  # what the caller got wrong (a task file, a brief, an output's path), in one line
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
