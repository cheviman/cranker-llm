"""cranker-llm's PreToolUse hook on Read: in the folders you list, a whole-file Read of a text file over max_lines
lines is turned away with how to map it with the local worker first, or read a range. A Read with offset or limit
always goes through (that is also the way to read one whole when it is needed). Off until you list folders:

    CRANKER_HOME/config.json:  "read_guard": {"roots": ["D:/projects/personal"], "max_lines": 300}

so work you do not want slowed down is never touched. Any error lets the Read through. Each refusal is logged in
logs/read_guard.jsonl, which `worker.py usage` counts. Standard library only.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("CRANKER_HOME") or Path.home() / ".cranker-llm")
WORKER = Path(__file__).resolve().parent / "worker.py"
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf", ".ipynb", ".zip", ".7z", ".gz", ".bin",
          ".rom", ".gguf", ".exe", ".dll", ".so", ".o", ".obj", ".lib", ".a", ".wav", ".ogg", ".mp3", ".mp4", ".ttf"}


def norm(p):
    p = str(p)
    if os.name == "nt" and len(p) > 2 and p[0] == "/" and p[2] == "/" and p[1].isalpha():  # /d/... (Git Bash)
        p = p[1] + ":" + p[2:]
    return os.path.normcase(os.path.abspath(os.path.expanduser(p)))


def main():
    # utf-8-sig: with a BOM in the config (Windows PowerShell 5.1 writes one) the guard went quiet without a word
    cfg = json.loads((HOME / "config.json").read_text(encoding="utf-8-sig")) if (HOME / "config.json").exists() else {}
    guard = cfg.get("read_guard") or {}
    roots = [norm(r) for r in guard.get("roots", [])]
    max_lines = int(guard.get("max_lines", 300))
    if not roots:
        return
    data = json.load(sys.stdin)
    if data.get("tool_name") != "Read":
        return
    ti = data.get("tool_input") or {}
    if ti.get("offset") is not None or ti.get("limit") is not None or not ti.get("file_path"):
        return
    path = norm(ti["file_path"])
    # rstrip: a drive's root already ends in the separator ("d:\\"), and with one more nothing was under it
    if not any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots):
        return
    if Path(path).suffix.lower() in BINARY or not os.path.isfile(path):
        return
    with open(path, "rb") as f:
        raw = f.read(64 << 20)
    if b"\0" in raw[:8192]:
        return
    lines = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
    if lines <= max_lines:
        return
    url = f"http://{cfg.get('host', '127.0.0.1')}:{cfg.get('port', 8090)}/health"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as r:
            running = json.loads(r.read()).get("status") == "ok"
    except Exception:
        running = False
    py = "py" if os.name == "nt" else "python3"
    start = "" if running else f"The worker is not running: start it first ({py} \"{WORKER}\" start). "
    reason = (
        f"{ti['file_path']} has {lines} lines: in these folders a file over {max_lines} lines is not read whole "
        f"(cranker-llm's read guard). {start}Map it with the local worker first, e.g. {py} \"{WORKER}\" ask -f "
        f"\"{ti['file_path']}\" --number -o <scratch>/map.md -p \"For each function, in order, one line: `name` (line N): "
        f"what it does, concretely. Nothing else.\" -- then Read only the range you need (offset and limit). If the "
        f"whole file is really needed (editing most of it), Read it with offset 1 and limit {lines}."
    )
    try:
        (HOME / "logs").mkdir(parents=True, exist_ok=True)
        with open(HOME / "logs" / "read_guard.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "path": ti["file_path"], "lines": lines,
                                "cwd": data.get("cwd")}) + "\n")
    except OSError:
        pass
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # never block a Read over a bug here
    sys.exit(0)
