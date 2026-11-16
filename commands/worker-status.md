---
description: Is the local worker running, which model, how much GPU memory, and what it has saved today
allowed-tools: Bash
---

Run `status` and then `usage --since` today's date (YYYY-MM-DD) with the local worker:
`py "${CLAUDE_PLUGIN_ROOT}/scripts/worker.py"` on Windows, `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/worker.py"`
elsewhere. Report in two or three lines: running or not (model, slots, GPU memory), and today's requests and the
tokens kept off Claude. If it is not configured, say so and offer `scripts/setup.py` (it lists the models that fit;
downloading needs the user's yes). Do not start or stop the server for this.
