---
description: What the local worker's runs read and wrote, and what the same tokens would cost on Claude, by day: the last 7 days, or since a date
argument-hint: "[YYYY-MM-DD]"
allowed-tools: Bash
---

Run the local worker's `runs`, with `--since $ARGUMENTS` if a date was given: `py
"${CLAUDE_PLUGIN_ROOT}/scripts/worker.py" runs` on Windows, `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/worker.py" runs`
elsewhere. Show its first line and its table as they are, then one line with the period's totals: the runs, the
tokens read and written, and what they would cost at each Claude price in the table (and the remote's own cost if
it has that column).

Say in that line that this is what the worker consumed, whatever came of each run, and not what it saved: the
savings are in `/worker-usage` ("USD saved"). Add nothing from another command, and do not round the table's
numbers.
