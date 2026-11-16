---
description: The tokens the local worker read instead of Claude, by command, project and day, and the day's scale
argument-hint: "[YYYY-MM-DD]"
allowed-tools: Bash
---

Run the local worker's `usage` command, with `--since $ARGUMENTS` if a date was given:
`py "${CLAUDE_PLUGIN_ROOT}/scripts/worker.py" usage` on Windows, `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/worker.py"
usage` elsewhere. Show its tables as they are, then one line with the total "kept off Claude" and the share of it
that comes from `agent` (an upper bound: the agent's input counts Claude Code's own prompt on every turn).

Then run `scripts/local_share.py` the same way, with `--day $ARGUMENTS` if a date was given, and show its first line
(Claude's new tokens against the worker's, and the scale) and its sessions table as they are.
