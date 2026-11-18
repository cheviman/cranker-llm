"""How much of the work the local worker does next to Claude, against a scale: `ratio` local tokens for each token
of Claude (5 by default: a million local for 200K of Claude). A target, not a rule: it never blocks anything.

Claude's tokens are what each turn adds, from Claude Code's transcripts (CLAUDE_CONFIG_DIR or ~/.claude, projects/):
input + cache writes + output, one count per message id, subagents included. Cache reads are left out: they are the
same context sent again on every turn, and they run to dozens of times the rest. The worker's are in + out from
usage.jsonl (its thinking too), benchmarks left out. That is what it consumed; the effective part leaves out the runs
that read and gave nothing back (an agent stopped or in an error, an empty answer), and the scale is measured on it:
both are shown. Each request carries the calling session's id (CLAUDE_CODE_SESSION_ID), so a session's own share is
exact.

    local_share.py [--day YYYY-MM-DD]   the day: in all, and by session
    local_share.py --hook               the hook (hooks/hooks.json: PostToolUse and UserPromptSubmit)

The hook stays quiet until the session has used the worker (or works in one of the folders under `roots`), has spent
`floor` tokens of Claude and is below the scale; then, at most every `every_min` minutes, it adds a note to Claude's
context with the session's numbers and what to hand over. Settings, in CRANKER_HOME/config.json:

    "scale": {"ratio": 5, "roots": [], "floor": 150000, "every_min": 15}

`ratio` 0 turns the hook off; CRANKER_RATIO overrides it. Any error is silent. Standard library only.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HOME = Path(os.environ.get("CRANKER_HOME") or Path.home() / ".cranker-llm")
USAGE_LOG = HOME / "logs" / "usage.jsonl"
STATE = HOME / "logs" / "local_share.json"
PROJECTS = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
WORKER = Path(__file__).resolve().parent / "worker.py"
DEFAULTS = {"ratio": 5.0, "roots": [], "floor": 150_000, "every_min": 15}
CHECK = 300  # s between two measurements in one session (each reads the session's transcript)


def settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads((HOME / "config.json").read_text(encoding="utf-8")).get("scale") or {})
    except (OSError, ValueError, AttributeError):
        pass
    if os.environ.get("CRANKER_RATIO"):
        s["ratio"] = os.environ["CRANKER_RATIO"]
    s["ratio"] = float(s["ratio"])
    return s


def norm(p):
    p = str(p)
    if os.name == "nt" and len(p) > 2 and p[0] == "/" and p[2] == "/" and p[1].isalpha():  # /d/... (Git Bash)
        p = p[1] + ":" + p[2:]
    return os.path.normcase(os.path.abspath(os.path.expanduser(p)))


def _day_of(ts):
    """A transcript's UTC timestamp -> the local day"""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d")


def claude_tokens(files, day=None):
    """{session: [tokens, first cwd]} from transcripts; a subagent's file counts for its session"""
    seen = set()
    out = {}
    for f in files:
        f = Path(f)
        sid = f.parent.parent.name if f.parent.name == "subagents" else f.stem
        try:
            fh = open(f, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"usage"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                m = r.get("message") or {}
                u = m.get("usage") if isinstance(m, dict) else None
                if not isinstance(u, dict) or not r.get("timestamp"):
                    continue
                if day and _day_of(r["timestamp"]) != day:
                    continue
                key = m.get("id") or r.get("requestId")
                if key in seen:
                    continue
                seen.add(key)
                a = out.setdefault(sid, [0, r.get("cwd", "?")])
                a[0] += (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) \
                    + (u.get("output_tokens") or 0)
    return out


def local_rows(day=None):
    rows = []
    try:
        with open(USAGE_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("tag") or (day and not r.get("ts", "").startswith(day)):
                    continue
                rows.append(r)
    except OSError:
        pass
    return rows


def spent(r):
    """What a request consumed: everything the model read and wrote, whether or not anything came back"""
    return r["in"] + r["out"]


def work(r):
    """A request's effective tokens: its consumption, or 0 for a run that read and gave nothing back (an agent the
    worker stopped or that ended in an error, an empty answer). The scale is measured on these: it is about work
    done, and tokens that only burnt must not move it."""
    return 0 if r.get("stopped") or r.get("error") or not r.get("answer") else spent(r)


def ratio_text(local, claude):
    return f"{local / claude:.2f} to 1" if claude else "no Claude tokens yet"


def report(day, s):
    start = datetime.strptime(day, "%Y-%m-%d").timestamp()
    files = [f for f in PROJECTS.rglob("*.jsonl") if f.stat().st_mtime >= start] if PROJECTS.exists() else []
    claude = claude_tokens(files, day)
    rows = local_rows(day)
    by_session = {}  # session: [consumed, effective]
    for r in rows:
        if r.get("session"):
            a = by_session.setdefault(r["session"], [0, 0])
            a[0] += spent(r)
            a[1] += work(r)
    c_total = sum(v[0] for v in claude.values())
    l_spent = sum(spent(r) for r in rows)
    l_total = sum(work(r) for r in rows)
    print(f"{day}: Claude {c_total:,} new tokens; the local worker consumed {l_spent:,}, {l_total:,} of them "
          f"effective: {ratio_text(l_total, c_total)} (the scale: {s['ratio']:g} to 1, "
          f"{int(c_total * s['ratio']):,} local)")
    shown = [(sid, c, cwd) for sid, (c, cwd) in claude.items() if c >= 20_000 or by_session.get(sid)]
    if shown:
        print(f"\n{'session':<9} {'project (where it started)':<40} {'Claude':>11} {'consumed':>11} "
              f"{'effective':>11} {'ratio':>7}")
        for sid, c, cwd in sorted(shown, key=lambda t: -t[1]):
            used, eff = by_session.get(sid, (0, 0))
            print(f"{sid[:8]:<9} {cwd[-40:]:<40} {c:>11,} {used:>11,} {eff:>11,} {eff / c if c else 0:>7.2f}")
    untagged = [r for r in rows if not r.get("session")]
    if untagged:
        print(f"\n{len(untagged)} of the worker's requests that day carry no session (not from Claude Code, or from "
              f"before cranker-llm 1.1.0): {sum(spent(r) for r in untagged):,} tokens, in the day's total only")
    print("\nClaude: input + cache writes + output (cache reads left out: the same context again on every turn). "
          "The worker: consumed is in + out of every request; effective leaves out the runs that read and gave "
          "nothing back (an agent stopped or in an error, an empty answer), and the ratio is on it. A session's row "
          "is that session's alone: its numbers are not added to another's.")


def note(claude, local, ratio, used=None):
    """local: the session's effective tokens; used: all it consumed, said when some of it gave nothing back"""
    wasted = ""
    if used and used > local:
        wasted = (f" The worker consumed {used:,} in all: {used - local:,} went on runs that read and gave nothing "
                  "back; they are consumption, not work, and the scale is on the effective ones.")
    return (
        f"cranker-llm scale: this session has spent {claude:,} tokens of Claude and {local:,} effective tokens of the "
        f"local worker ({ratio_text(local, claude)}).{wasted} The scale set for it is {ratio:g} local tokens for each "
        "one of Claude: give "
        "the worker real load, in the background, and keep it busy while you do the rest (skill cranker-llm; "
        f"python {WORKER} status). Real load: the first draft of code you are about to write (a function, a script, "
        "a test, a mechanical patch), from an exact spec, for you to review and fix; a map before reading any long "
        "file; searches across many files (agent); reviewing your diff before you do; long logs and outputs; names "
        "and comments in batch; first drafts of notes and READMEs. Not load: accents, spelling, one-line rewrites, "
        "anything quicker to do than to hand over. What it answers is a draft: check it before it is used. When a "
        "task comes back wrong or fails, send it again with something changed (--think, more --max-tokens, a "
        "narrower task, a --check) instead of fixing it yourself, and tell the user when the limit is the worker's "
        "own; one failure is not what the worker can't do. When "
        "you report the worker's work, give this session's own numbers (worker.py usage --session), never another "
        "session's or project's added in. Not a hard rule: the scale is what to aim at."
    )


def hook():
    s = settings()
    if s["ratio"] <= 0:
        return
    data = json.load(sys.stdin)
    sid = data.get("session_id")
    tp = data.get("transcript_path")
    if not sid or not tp:
        return
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    st = state.get(sid, {})
    now = time.time()
    if now - st.get("checked", 0) < CHECK or now - st.get("noted", 0) < float(s["every_min"]) * 60:
        return
    st["checked"] = now
    files = [Path(tp)] + sorted(Path(tp).with_suffix("").glob("subagents/*.jsonl"))
    claude = claude_tokens(files).get(sid, [0, ""])[0]
    mine = [r for r in local_rows() if r.get("session") == sid]
    used, local = sum(spent(r) for r in mine), sum(work(r) for r in mine)
    cwd = norm(data.get("cwd") or ".")
    listed = any(cwd == norm(r) or cwd.startswith(norm(r) + os.sep) for r in s["roots"])
    if (listed or used) and claude >= int(s["floor"]) and local < s["ratio"] * claude:
        st["noted"] = now
        print(json.dumps({"hookSpecificOutput": {"hookEventName": data.get("hook_event_name", "PostToolUse"),
                                                 "additionalContext": note(claude, local, s["ratio"], used)}}))
    state = {k: v for k, v in state.items() if now - v.get("checked", 0) < 7 * 86400}
    state[sid] = st
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, STATE)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--day", default=time.strftime("%Y-%m-%d"))
    p.add_argument("--hook", action="store_true")
    args = p.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    if args.hook:
        try:
            hook()
        except Exception:
            pass
        return 0
    report(args.day, settings())
    return 0


if __name__ == "__main__":
    sys.exit(main())
