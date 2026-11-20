---
description: The tokens the local worker read instead of Claude: this session's first, then by command, project and day, and the day's scale
argument-hint: "[YYYY-MM-DD]"
allowed-tools: Bash
---

Run the local worker's `usage --session`, with `--since $ARGUMENTS` if a date was given:
`py "${CLAUDE_PLUGIN_ROOT}/scripts/worker.py" usage --session` on Windows, `python3
"${CLAUDE_PLUGIN_ROOT}/scripts/worker.py" usage --session` elsewhere. That is this session's own work: show its
tables as they are, then one line with this session's "kept off Claude", the share of it that comes from `agent` (an
upper bound: the agent's input counts Claude Code's own prompt on every turn), and the runs that ended in nothing
if it lists any. This line is the answer to "how much did the worker do here". When the quality line says that some
of the right pieces passed a check never tried or had no check, say how many in that line too: they count as
effective, but nothing proved them right.

Then run `usage` again without `--session` and show its tables under their own heading, "all sessions and
projects". Never add another session's or project's numbers to this session's, and never report the total as this
session's work.

Then run `scripts/local_share.py` the same way, with `--day $ARGUMENTS` if a date was given, and show its first line
(Claude's new tokens against the worker's, and the scale) and its sessions table as they are, saying which row is
this session.
