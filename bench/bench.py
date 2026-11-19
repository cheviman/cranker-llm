"""cranker-llm's benchmarks, against the server running now (it does not start or stop it).

    py bench/bench.py [TAG] [--only map,rename,agent,batch] [--runs 3]

  map     list the 9 functions of bench/sample.c, one line each: how many it names
  rename  rename the local 'state' (not the field ->state) in sample.c, --runs times: how many come out exactly right
  agent   the agent: which functions call elevator_step, at which lines
  batch   16 explanations at once (--jobs = the server's slots): tokens out per second in all
  lines   map scripts/worker.py (some 1,800 lines, ~35K tokens with the answer) with the line of each function,
          without and with --number: how many lines come out exact (two long requests: a few minutes)
Answers and the report go to bench/runs/<TAG>/. The runs are tagged "bench" in usage.jsonl, so `worker.py usage`
leaves them out. Send the report with your GPU and profile: the profiles' "measured" lines come from it.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
WORKER = [sys.executable, str(BENCH.parent / "scripts" / "worker.py")]
SAMPLE = BENCH / "sample.c"
FUNCTIONS = ["queue_empty", "queue_push", "queue_pop", "elevator_init", "elevator_call", "next_state",
             "elevator_step", "elevator_run", "elevator_fault"]
STEPS = ["map", "rename", "agent", "batch", "lines"]
LONG = BENCH.parent / "scripts" / "worker.py"  # the long file of the lines step
OUT_TPS = re.compile(r"out (\d+) tok \(([\d.]+) tok/s\)")


def worker(*args):
    env = dict(os.environ, CRANKER_USAGE_TAG="bench")
    p = subprocess.run([*WORKER, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return p.returncode, (p.stdout + p.stderr).strip()


def code_of(text):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    return "\n".join(l.rstrip() for l in body.strip().splitlines())


def expected_rename():
    """'state' renamed to 'step' where it is a local: inside next_state and elevator_step, never after -> or ."""
    lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("static uint8_t next_state("))
    end = next(i for i, l in enumerate(lines) if l.startswith("void elevator_run("))
    for i in range(start, end):
        lines[i] = re.sub(r"(?<![>.])\bstate\b", "step", lines[i])
    return "\n".join(l.rstrip() for l in "\n".join(lines).strip().splitlines())


def step_map(out, rep):
    rc, line = worker("ask", "-f", str(SAMPLE), "-o", str(out / "map.md"), "--max-tokens", "1500", "-p",
                      "For each function in this file, in order, write one line: `name`: what it does, concretely. "
                      "Nothing else.")
    text = (out / "map.md").read_text(encoding="utf-8") if (out / "map.md").exists() else ""
    named = [f for f in FUNCTIONS if re.search(rf"\b{f}\b", text)]
    rep(f"map: {len(named)} of {len(FUNCTIONS)} functions named; {line.splitlines()[-1] if line else ''}")


def step_rename(out, rep, runs):
    want = expected_rename()
    right = 0
    tps = []
    for i in range(runs):
        o = out / f"rename_{i + 1}.c"
        rc, line = worker("ask", "-f", str(SAMPLE), "-o", str(o), "--max-tokens", "3000", "-p",
                          "Rewrite this C file exactly as it is, but rename the local variable 'state' to 'step' "
                          "everywhere it is a local variable (not the struct field 'state'). Output only the code.")
        got = code_of(o.read_text(encoding="utf-8")) if o.exists() else ""
        ok = got == want
        right += ok
        m = OUT_TPS.search(line)
        if m:
            tps.append(float(m.group(2)))
        rep(f"  rename {i + 1}: {'right' if ok else 'wrong'}" + (f", {m.group(2)} tok/s" if m else f" ({line[-200:]})"))
    rep(f"rename: {right} of {runs} right" + (f"; {min(tps):.0f}-{max(tps):.0f} tok/s" if tps else ""))


def step_agent(out, rep):
    lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    calls = [i + 1 for i, l in enumerate(lines) if "elevator_step(e);" in l]
    t0 = time.time()
    rc, line = worker("agent", "--dir", str(BENCH), "-o", str(out / "agent.md"), "-p",
                      f"In {BENCH}, which functions call elevator_step, and at which lines of sample.c? "
                      f"Answer with one line per call: the calling function and sample.c:line.")
    text = (out / "agent.md").read_text(encoding="utf-8") if (out / "agent.md").exists() else ""
    found = [n for n in calls if re.search(rf"\b{n}\b", text)]
    callers = [f for f in ("elevator_run", "elevator_fault") if f in text]
    rep(f"agent: {len(found)} of {len(calls)} call lines, {len(callers)} of 2 callers, {time.time() - t0:.0f} s; "
        f"{line.splitlines()[-1] if line else ''}")


def step_batch(out, rep):
    tasks = []
    for i in range(16):
        f = FUNCTIONS[i % len(FUNCTIONS)]
        tasks.append({"id": f"{i:02}", "prompt": f"Explain the function {f} step by step in about 150 words.",
                      "files": [str(SAMPLE)], "out": str(out / "batch" / f"{i:02}.md"), "max_tokens": 300})
    jl = out / "batch.jsonl"
    jl.write_text("\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8")
    rc, text = worker("batch", str(jl), "--force")
    last = text.splitlines()[-1] if text else ""
    m = re.search(r"(\d+) tokens out in (\d+) s", last)
    rep(f"batch: {last}" + (f" -> {int(m.group(1)) / max(int(m.group(2)), 1):.0f} tok/s in all" if m else ""))


def score_lines(text, want):
    """(functions named with a line, lines exact) in a map of `name` (line N) lines"""
    said = {m.group(1): int(m.group(2)) for m in re.finditer(r"`?(\w+)`?\s*\(lines? (\d+)", text)}
    return sum(f in said for f in want), sum(said.get(f) == n for f, n in want.items())


def step_lines(out, rep):
    """What --number is for: asked for each function's line, a model counts the lines by eye, and over a long file
    it drifts; with the number in front of each line it has it to copy."""
    source = LONG.read_text(encoding="utf-8").splitlines()
    want = {m.group(1): i for i, l in enumerate(source, 1) if (m := re.match(r"def (\w+)\(", l))}
    got = []
    for name, flags in (("plain", []), ("numbered", ["--number"])):
        o = out / f"lines_{name}.md"
        rc, line = worker("ask", "-f", str(LONG), "-o", str(o), "--max-tokens", "6000", *flags, "-p",
                          "For each top-level function in this file (not the methods), in order, write one line: "
                          "`name` (line N): what it does, in a few words. N is the line its `def` is on. Nothing "
                          "else.")
        named, exact = score_lines(o.read_text(encoding="utf-8") if o.exists() else "", want)
        got.append(f"{name} {exact} exact of {named} named")
        rep(f"  lines {name}: {line.splitlines()[-1] if line else ''}")
    rep(f"lines: {len(want)} functions in {len(source)} lines; " + "; ".join(got))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--only", default=",".join(STEPS))
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    unknown = [s for s in a.only.split(",") if s not in STEPS]
    if unknown:
        ap.error(f"--only: no step {', '.join(unknown)} (the steps: {', '.join(STEPS)})")
    rc, status = worker("status")
    if rc != 0:
        print(status)
        return 1
    out = BENCH / "runs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    report = open(out / "report.txt", "w", encoding="utf-8")

    def rep(text):
        print(text, flush=True)
        report.write(text + "\n")
        report.flush()

    rep(f"cranker-llm bench {a.tag}, {time.strftime('%Y-%m-%d %H:%M')}\n{status}")
    for s in a.only.split(","):
        if s == "rename":
            step_rename(out, rep, a.runs)
        else:
            globals()["step_" + s](out, rep)
    rep(f"report: {out / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
