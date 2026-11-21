"""How much of the work the local worker does next to Claude, against a scale: `ratio` local tokens for each token
of Claude (5 by default: a million local for 200K of Claude). A target, not a rule: it never blocks anything.

Claude's tokens are what each turn adds, from Claude Code's transcripts (CLAUDE_CONFIG_DIR or ~/.claude, projects/):
input + cache writes + output, one count per message id, subagents included. Cache reads are left out: they are the
same context sent again on every turn, and they run to dozens of times the rest. The worker's are in + out from
usage.jsonl (its thinking too), benchmarks left out, by two yardsticks that are kept apart. Consumed: every request,
whatever came of it; the tokens are spent all the same, and a task run four times is four runs. Effective: the work
that was right by the first or second try (`tries`); work that took more, or was left cut, failing its check or with
nothing back, is not effective. The scale is measured on the effective, and both are shown. Each request carries the
calling session's id (CLAUDE_CODE_SESSION_ID), so a session's own share is exact.

    local_share.py [--day YYYY-MM-DD]   the day: in all, and by session
    local_share.py --hook               the hooks (hooks/hooks.json: SessionStart, UserPromptSubmit, PostToolUse)

The hooks say what the user would otherwise keep repeating, to a session that has used the worker or works in one of
the folders under `roots`: the rule when the session starts and again after a compaction; a short reminder at every
prompt while Claude is doing more than the worker's effective work (below `least`); and, below the scale, a longer
note at most every `every_min` minutes. The last two wait for the session to have spent `floor` tokens of Claude.
Each tells volume from quality: more load when the worker has not consumed its share, better tasks, and not more,
when it has and little of it was effective. Settings, in CRANKER_HOME/config.json:

    "scale": {"ratio": 5, "least": 1, "tries": 2, "roots": [], "floor": 150000, "every_min": 15, "remind": true}

`ratio` 0 turns the hooks off; CRANKER_RATIO overrides it. `remind` false leaves only the longer note. Nothing is
ever blocked. Any error is silent. Standard library only.
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
DEFAULTS = {"ratio": 5.0, "roots": [], "floor": 150_000, "every_min": 15, "remind": True, "least": 1.0, "tries": 2}
CHECK = 300  # s between two measurements in one session (each reads the session's transcript)


def settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads((HOME / "config.json").read_text(encoding="utf-8-sig")).get("scale") or {})
    except (OSError, ValueError, AttributeError):
        pass
    if os.environ.get("CRANKER_RATIO"):
        s["ratio"] = os.environ["CRANKER_RATIO"]
    s["ratio"] = float(s["ratio"])
    s["tries"] = max(1, int(s["tries"]))
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


def local_rows(day=None, most=2):
    rows = []
    try:
        with open(USAGE_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not r.get("tag"):
                    rows.append(r)
    except OSError:
        pass
    # the quality yardstick, as in worker.py (mark_tries): a piece of work ("work": an output and its files) is
    # effective when its output was right by try `most`. Over the whole log, before the day is picked: a task
    # sent again today is one more try of yesterday's
    pieces = {}
    for r in rows:
        if r.get("work"):
            pieces.setdefault(r["work"], []).append(r)
    for runs in pieces.values():
        tries, beside = piece_runs(runs)
        right = bool(tries) and not wrong(tries[-1])
        for r in tries:
            r["_tries"], r["_right"] = len(tries), right
            r["_effective"] = right and len(tries) <= most
        for r in beside:  # the variants of a task that were not kept: consumed, not effective
            r["_tries"], r["_right"], r["_effective"], r["_beside"] = len(tries), False, False, True
    # the line that says which variant was kept is no request
    return [r for r in rows if r.get("cmd") != "variants" and (not day or r.get("ts", "").startswith(day))]


def piece_runs(runs):
    """As worker.py's: a piece of work's runs as (its tries, the runs spent beside them, on its variants not kept)"""
    tries, beside, pending = [], [], {}
    for r in runs:
        if r.get("cmd") == "variants":
            for k, rs in sorted(pending.items()):
                (tries if r.get("chosen") in (None, k) else beside).extend(rs)
            pending = {}
        elif r.get("variant"):
            pending.setdefault(r["variant"], []).append(r)
        else:
            tries.append(r)
    for k, rs in sorted(pending.items()):
        tries.extend(rs)
    return tries, beside


def wrong(r):
    """A run that gave nothing back (an agent the worker stopped or that ended in an error, an empty answer), a cut
    answer, or an output that still failed its check"""
    return bool(r.get("stopped") or r.get("error") or not r.get("answer") or r.get("cut") or r.get("check") == "fail")


def spent(r):
    """What a request consumed: everything the model read and wrote. The tokens are spent whatever came of them:
    a task run four times is four runs here."""
    return r["in"] + r["out"]


def work(r):
    """A request's effective tokens, the other yardstick: its consumption when its piece of work was right by the
    first or second try (scale.tries), else 0. A run with no piece of work to its name is judged by itself."""
    return spent(r) if r.get("_effective", not wrong(r)) else 0


def quality(rows, most):
    """The pieces of work among these runs, by the try they were right at: (pieces, first, by `most`, more, wrong)"""
    pieces = {r["work"]: r for r in rows if r.get("work") and not r.get("_beside")}.values()
    first = sum(1 for r in pieces if r["_effective"] and r["_tries"] == 1)
    by = sum(1 for r in pieces if r["_effective"] and r["_tries"] > 1)
    more = sum(1 for r in pieces if r["_right"] and r["_tries"] > most)
    return len(pieces), first, by, more, len(pieces) - first - by - more


def quality_text(rows, most):
    n, first, by, more, bad = quality(rows, most)
    if not n:
        return ""
    return (f"Quality, the other yardstick: of its {n} pieces of work, {first} were right at the first try, {by} by "
            f"try {most}, {more} took more tries and {bad} were left wrong.")


def ratio_text(local, claude):
    return f"{local / claude:.2f} to 1" if claude else "no Claude tokens yet"


def report(day, s):
    start = datetime.strptime(day, "%Y-%m-%d").timestamp()
    files = [f for f in PROJECTS.rglob("*.jsonl") if f.stat().st_mtime >= start] if PROJECTS.exists() else []
    claude = claude_tokens(files, day)
    rows = local_rows(day, s["tries"])
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
    n, first, by, more, bad = quality(rows, s["tries"])
    if n:
        print(f"\nQuality, the other yardstick: {n} pieces of work (an output and the files it is about), {first} "
              f"right at the first try, {by} by try {s['tries']}, {more} took more tries, {bad} left wrong.")
    print("\nClaude: input + cache writes + output (cache reads left out: the same context again on every turn). "
          "The worker, by two yardsticks. Consumed: in + out of every request, whatever came of it: the tokens are "
          f"spent all the same, and a task run four times is four runs. Effective: the work that was right by try "
          f"{s['tries']}, with every run it took; work that took more tries, or was left cut, failing its check or "
          "with nothing back, is not effective. The ratio is on the effective. A session's row is that session's "
          "alone: its numbers are not added to another's.")


PY = "py" if os.name == "nt" else "python3"
# The user's words: the scale is "not strict"; "ten to one makes no sense if it runs it four times"; "a task done
# four times is four times, but it is not effective: that is another yardstick"; "the idea is work of quality at the
# first or the second"; "the tokens are spent all the same".
HOW = ("What makes work effective is a task that is right at the first or second try: an exact spec (what goes in, "
       "what comes out, a neighbouring file as the example) and a --check that proves it. The scale is where to aim, "
       "not a quota: never send work just to move a number.")
LOAD = ("the first draft of code you are about to write (a function, a script, a test, a mechanical patch), for you "
        "to review; a map before reading any long file; searches across many files (agent); a review of your diff "
        "before your own; long logs and outputs; names and comments in batch; first drafts of notes and READMEs. Not "
        "load: accents, spelling, one-line rewrites, anything quicker to do than to hand over")


def numbers(claude, used, local, tries):
    """The session by both yardsticks: what its worker consumed, and how much of that was effective"""
    text = f"{claude:,} tokens of Claude and {local:,} effective tokens of the local worker ({ratio_text(local, claude)})"
    if used > local:
        text += (f". The worker consumed {used:,} in all ({ratio_text(used, claude)}): {used - local:,} went on work "
                 f"that was not right by try {tries} (it took more tries, or was left cut, failing its check or with "
                 "nothing back). Those tokens were spent all the same; they are consumption, and not effective")
    return text


def note(claude, used, local, s, quality_line=""):
    """The long note, at most every `every_min`. Which of the two yardsticks is short decides what it asks for:
    more load when the worker has not consumed the scale, better tasks when it has and little of it was effective."""
    ratio = s["ratio"]
    if used >= ratio * claude:
        ask = ("The worker has had the volume: what is short is quality. Do not send more: send it so that it is "
               f"right by try {s['tries']}. {HOW} When a task comes back wrong, work out what its prompt missed "
               "before sending it again, instead of sending it again as it is.")
    else:
        ask = (f"Give the worker real load, in the background, and keep it busy while you do the rest (skill "
               f"cranker-llm; {PY} \"{WORKER}\" status). Real load: {LOAD}. {HOW} When a task comes back wrong, send "
               "it again with something changed (the rule or the example it missed, --think, a narrower task, a "
               "--check) rather than fix it yourself, and tell the user when the limit is the worker's own.")
    return (
        f"cranker-llm scale: this session has spent {numbers(claude, used, local, s['tries'])}. The scale the user "
        f"set is {ratio:g} effective local tokens for each one of Claude. {quality_line}{' ' if quality_line else ''}"
        f"{ask} What the worker answers is a draft: check it before it is used. When you report its work, give this "
        "session's own numbers, consumed and effective (worker.py usage --session), never another session's or "
        "project's added in."
    )


def standing(s):
    """What a session is told when it starts, and again after a compaction dropped it: the rule itself, so that the
    user does not have to say it."""
    return (
        f"cranker-llm: this machine has a local worker (a model on its own GPU), and the user wants it to do more of "
        f"the volume than Claude: their scale is {s['ratio']:g} local tokens for each token of Claude, as a "
        "direction, not a quota. They should not have to say it again in this session. On every task the worker's "
        "part goes out first, in the background, before you read, search or draft that part yourself. Do not prepare "
        "a task for the worker by reading the files it is about: its map of them is the preparation "
        f"({PY} \"{WORKER}\" ask -f FILE --number -o OUT -p \"For each function, in order, one line: `name` (line N): "
        "what it does.\"; the skill cranker-llm has the rest). Its load: maps of long files, searches across many "
        "files, the first draft of code you are about to write, a review of your diff, long logs. Keep for yourself "
        "what needs judgement. Its work is measured by two yardsticks: what it consumed, where every run counts, and "
        f"what was effective, which is the work that was right by try {s['tries']}. {HOW}"
    )


def reminder(claude, used, local, s, idle, quality_line=""):
    """What a prompt starts with while Claude is doing more than the worker's effective work: what the user was
    repeating by hand"""
    least = float(s["least"])
    if used >= least * claude:  # it has been sent the volume: what is short is quality
        ask = (f"The worker has had the volume: what is short is quality. Do not send more for the sake of it. What "
               f"this request needs from the worker, send so that it is right by try {s['tries']}: an exact spec "
               "(what goes in, what comes out, a neighbouring file as the example) and a --check that proves it.")
    else:
        ask = ("Before you start on this request yourself, send the worker its part of it, in the background: the "
               "map or the search of what it touches, the first draft of what has to be written, from an exact spec "
               "and with a --check, so that it is right at the first or second try. Do not read first to prepare "
               "that: what you would read is what it maps. A request with no volume in it needs nothing sent, and "
               "nothing is sent just to move a number.")
    return (f"cranker-llm scale: this session has spent {numbers(claude, used, local, s['tries'])}{idle}. That is "
            f"below the {least:g} to 1 the user wants at the least. {quality_line}{' ' if quality_line else ''}{ask}")


def under(path, root):
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def say(event, text):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))


def hook():
    s = settings()
    if s["ratio"] <= 0:
        return
    data = json.load(sys.stdin)
    sid = data.get("session_id")
    tp = data.get("transcript_path")
    if not sid or not tp:
        return
    event = data.get("hook_event_name") or "PostToolUse"
    cwd = norm(data.get("cwd") or ".")
    listed = any(under(cwd, norm(r)) for r in s["roots"])
    most = s["tries"]
    if event == "SessionStart":
        # the rule itself, when the session starts and again after a compaction dropped it
        if s["remind"] and (listed or any(r.get("session") == sid for r in local_rows(most=most))):
            say(event, standing(s))
        return
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    st = state.get(sid, {})
    now = time.time()
    # the long note comes at most every `every_min`; the short reminder at a prompt is not paced, and each prompt
    # measures: that one is what the user was saying by hand
    at_prompt = event == "UserPromptSubmit" and s["remind"]
    if not at_prompt and (now - st.get("checked", 0) < CHECK or now - st.get("noted", 0) < float(s["every_min"]) * 60):
        return
    st["checked"] = now
    files = [Path(tp)] + sorted(Path(tp).with_suffix("").glob("subagents/*.jsonl"))
    claude = claude_tokens(files).get(sid, [0, ""])[0]
    mine = [r for r in local_rows(most=most) if r.get("session") == sid]
    used, local = sum(spent(r) for r in mine), sum(work(r) for r in mine)  # the two yardsticks
    applies = (listed or used) and claude >= int(s["floor"])
    if at_prompt:
        # only while Claude is doing more than the worker's effective work (`least`, 1 to 1): from there up to the
        # scale, the long note at its own pace is enough. The scale is where to aim, and `least` is what is asked
        if applies and local < float(s["least"]) * claude:
            last = max((r.get("ts", "") for r in mine), default="")
            idle = "; it has sent the worker nothing yet"
            if last:
                minutes = (now - datetime.strptime(last, "%Y-%m-%d %H:%M:%S").timestamp()) / 60
                idle = f"; nothing has gone to the worker for {minutes:.0f} minutes" if minutes >= 10 else ""
            say(event, reminder(claude, used, local, s, idle, quality_text(mine, most)))
    elif applies and local < s["ratio"] * claude:
        st["noted"] = now
        say(event, note(claude, used, local, s, quality_text(mine, most)))
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
