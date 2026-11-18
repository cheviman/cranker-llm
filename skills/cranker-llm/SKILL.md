---
name: cranker-llm
description: "A local LLM (llama.cpp on this machine's GPU) that Claude Code hands volume work to, so it does not spend Claude's tokens: map or summarise a long file before reading it, find who calls what across many files, propose names and comments in bulk, summarise long logs and command output, write first drafts. Use before reading a file of more than ~300 lines whole, before launching a Claude subagent to search or read a lot, for naming or commenting rounds, with long logs, or when the user asks to save tokens or to use the local worker."
allowed-tools: Read, Grep, Glob, Bash
---

# cranker-llm

A llama.cpp server on this machine runs a local model; `scripts/worker.py` hands it work. Claude orchestrates and
checks; the worker reads and writes the volume, and what it produces **does not go through Claude's context**: it
goes to a file, and Claude sees a one-line summary. The server listens only on 127.0.0.1: nothing leaves the machine.

`W` below is `${CLAUDE_PLUGIN_ROOT}/scripts/worker.py`, run with `py` on Windows and `python3` elsewhere.

## Before the first use

```
py W status        # running ... model ... slots ... VRAM
py W config        # where the settings and the model are
```

- "model missing" or no config: run `py ${CLAUDE_PLUGIN_ROOT}/scripts/setup.py`, which lists the models that fit
  this GPU. **Downloading is the user's call**: tell them the profile, the files, their size and source (the
  `--dry-run` output) and wait for a yes before `setup.py --profile NAME --yes`.
- "not running": `py W start` (10-90 s; the first start may patch the model's chat template and restart once).
  Starting needs no permission when it is configured; stopping does: **never `stop` the server unless the user asks**:
  another session of theirs may be using it.
- `start` refusing for the GPU power limit (`max_power_w` in the config): tell the user; never pass
  `--ignore-power-limit` on your own.

**The read guard** (a hook): in the folders the user listed under `read_guard.roots` in the worker's config, a
whole-file Read of a text file over `max_lines` (300) is refused with how to map it with the worker. A Read with
`offset`/`limit` always goes through; when a file is really needed whole (to edit most of it), Read it with
offset 1 and limit its line count. Folders outside those roots are never touched.

**The scale** (a hook): the target is `ratio` local tokens for each token of Claude (`scale` in the config, 5 by
default: a million local for 200K of Claude). Claude's tokens are the new ones of each turn (input, cache writes,
output; not cache reads), and the worker's are its `in` + `out`. When a session that used the worker (or works under
`scale.roots`) is below it, a note with the session's numbers comes into the context, at most every 15 minutes: hand
over real load then, not more accents. `py scripts/local_share.py [--day YYYY-MM-DD]` (beside W) shows the day by
session. It is a target to aim at, not a rule, and it blocks nothing.

## The three ways to use it

**1. `ask`: a question about files, no tools.** The files go to the model, the answer to `-o`.
```
py W ask -f src/big.c -o /tmp/map.md -p "For each function, in order, one line: `name` (line N): what it does, concretely. Nothing else."
py W ask -f src/big.c:120-260 -o out.md -p "..."     # a line range
py W ask ... --think                                  # thinks first: slower, better on hard questions
```
It prints `-> out.md: in N tok, out M tok, s`. Read the output only if needed, and only the part needed.

With `--think` the thinking counts inside `--max-tokens`, and Qwen thinks a lot: a 260-line diff's review and a
647-line file's map spent 10,000 tokens thinking and came out empty. So every `--think` request carries a thinking
budget (`--max-tokens` less up to 2,048 for the answer): past it, the server closes the thinking and the model
answers. If one still comes back empty, the worker hands the cut thinking back and asks for the conclusion without
thinking; the summary then says **`RECOVERED ... (check it)`**, and that answer is checked before anything else
(it comes from half a reasoning). `--effort medium` or `low` (`"effort"` in a batch task) makes Qwen3.8 think less
than its default `xhigh`. Maps, summaries and searches: no `--think`. If it must think: `--max-tokens 20000
--keep-thinking`, and expect 10-15 minutes.

**2. `batch`: many tasks in parallel** (8 slots: ~118 tok/s in all on a 3090, against ~30 for one request). One JSON
object per line: `{"id", "prompt", "files": [...], "out", "max_tokens"?, "think"?, "system"?, "check"?,
"retries"?}`. Tasks whose `out`
exists are skipped, so a batch can be stopped and resumed; `--force` redoes them.
```
py W batch tasks.jsonl
```

**3. `agent`: Claude Code itself on the local model**, with Read, Grep and Glob (and Write/Edit only inside its own
folder). For searching many files and putting an answer together. It loads none of the user's settings (no hooks,
plugins, CLAUDE.md or MCP servers).
```
py W agent --dir src -o /tmp/answer.md -p "Find ... Answer in N lines with file:line."
py W agent --dir src --write /tmp/job1 -p "... write the results to results.md"
```
`--dir` with source folders, not a repo root (it would bring that repo's CLAUDE.md); it can be repeated. A write
outside `--write` is denied by Claude Code's permission rules. **Never give `--write` a repository**: the agent writes
in its folder and Claude applies what it found.

The agent is for **finding and putting together**, not for writing code. A turn has 8,000 tokens, its thinking
included (capped at 4,096), so a file it writes must fit in the rest (~300 lines). It can write (measured: a
100-line test and its 100-line export script from two 700-line sources, 23 turns, 34 minutes alone on the server),
but code to write, tests included, goes first through `ask` or `batch` with `--check`, which compile or run it
and leave the server to the other sessions. When a turn is cut twice in a row before it
acts, the worker stops the agent at once and says so (`agent stopped: ...`): split the task, don't run it again as is.
Run one agent at a time and read its result line before starting the next one: never chain several long agents in
one background command and leave them unattended.

Prompts to the worker in English. A prompt with double quotes or a long one goes in a file (`--prompt-file`):
Windows PowerShell 5.1 splits `-p "..."` with `\"` inside into several arguments.

## The brief and the check: so Claude neither prepares nor redoes the work

The worker sees no CLAUDE.md, no environment and no compiler: only the files and the prompt. Two things make up for
that, and both go with every task that writes something:

- **The brief**: `.cranker/brief.md` in the project (the caller's folder or above it, up to the repository's root;
  outside a repo, only the caller's folder: then pass `--brief`) goes into the system prompt of every `ask`, `batch`
  and `agent`. In a repo the user doesn't own, `--brief ""`. The first time a project gets code, names or comments from the worker and has
  no brief, Claude writes one from that project's CLAUDE.md: only the rules the code must follow (language,
  indentation, headers and markers, what is never done), 5 to 20 lines, not Claude's own workflow. It is a file in
  the user's repo: say so when it is created. `--brief FILE` uses another one, `--brief ""` none.
- **The check**: `--check "CMD {out}"` (or `"check"` in a batch line), a command that exits 0 when the output is
  right: a compile, a test, a style script. The worker runs it and hands the model its errors until it passes
  (`check_retries`, 2; from the second fix on it thinks first). What comes back to Claude is only what still fails:
  `CHECK FAILED` and the errors in `<out>.check.txt`. A cut answer is asked for again with 1.5 times the room, once,
  and its line starts `cut by max_tokens N`. **Don't run fix rounds by hand** (compile, read errors, write a
  prompt for each one, send again): that's the check's job. A check runs in the caller's folder, several at once in
  a batch, so it writes only files of its own (one object file per task, say). A check that runs the draft (a test
  script, a program) builds the folder it runs in again on every run, never reusing one a previous round's draft
  could have changed: a draft test's cleanup once deleted the project's `main.c`, and every later round failed on the
  missing file instead of on its own errors. An answer that is one fenced block and nothing else (`` ```c ... ``` ``)
  is written without its fence, so a check doesn't need to strip it.
- **What all the tasks share is written once**: `batch --prompt-prefix rules.txt` or `--system-file`, and
  `--check` for all of them; a line's own value wins. The line then holds only what is its own.

The check proves what it checks, and nothing more. What passes still gets read in the diff like any draft, and an
output without a check is checked by Claude as before. `usage` shows the rework by project: first time / by the
worker / for Claude. When "for Claude" stays high, the prompt, the brief or the check is what needs work, not
more rounds.

## The shared context

The context (64K by default) is **one pool for all the slots** (`-kvu`), not 8K each. An agent's request alone is
32-40K tokens (Claude Code's system prompt, its tools and what it has read) and grows with every file it reads. If
the pool overflows, the server does not refuse the new request: it fails **every** running request (HTTP 500,
"Context size has been exceeded"). So:
- **An agent runs alone**: it starts with the server idle, asks and batches wait while it runs, and two agents take
  turns (a lock). For a search in a few files, Claude's own grep or an `ask` with the files is faster. An agent
  takes the worker from **every session on the machine**, for up to its `--timeout`: keep it short, and don't
  start one while other sessions' batches are running (`status` shows the requests and who has the worker).
- **A task that waits says who it waits for**: `waiting, up to 30 min: an agent of session 601d636d
  (vs-cheviman\saturn-fullset; 12 min in, 33 min left at most; "Write a test for...") has the worker`. When the
  wait (`context_wait`, 30 min) runs out it ends as `not run` (`WAIT` in a batch), which is not a failure of the
  task: tell the user which session has the worker, and run the same command again when it is free (a batch skips
  what is done). Don't rewrite the task or do it by hand because of it.
- **`ask` and `batch` send a request only when it fits** (counted with the server's tokenizer, answer included,
  against what the running requests hold); otherwise they wait. One that does not fit even alone fails at once:
  split it, use line ranges or lower `--max-tokens`.
- Two big asks started in the same second may both see room: start them ~30 s apart.

## How Claude works with it

- Worker tasks run in the background (`run_in_background`); the notice comes when they end. Meanwhile Claude does
  what needs judgement. Never poll the output.
- What is delegated is not also done in parallel.
- Small things Claude does itself: a file under ~300 lines, a grep, a short diff cost less to read than to send.
- **What the worker answers is a draft, not a finding**: every file:line and every fact used is checked in the code
  before telling the user or touching the repo.

## When a task fails or comes back wrong

Claude's reflex is to fix the output itself, or to do the task itself. That is the work the worker was for, and it
is how a session ends with Claude doing everything and the scale at zero. In this order:

1. **Send it again with something changed**, not the same thing and not a fix by hand: `--think` (or `--effort`),
   more `--max-tokens` for a cut answer, a narrower task (line ranges, one function, one file), a `--check` that
   proves it (the test or the compile, not only "it parses"), or the rule or example the prompt missed. The draft
   and its error go back as files of the new task. A failed output ends with a `next:` line saying this.
2. **Fix by hand only a slip of a line or two**: what is quicker to fix than to describe.
3. **When the limit is the worker's own** (a cap, a timeout, a missing option, an error of the server), tell the
   user in the report: what failed, with its numbers, and what in cranker-llm would fix it. Don't work around it in
   silence, and don't wait an hour for it: a run that makes no progress is stopped and looked at.
4. **One failure is that task with those parameters.** Don't conclude, tell the user or write to memory that "the
   worker can't do X" from one run: say what was tried, with what, and what was not tried.

**Consumed is not effective.** Everything the worker reads and writes is consumed and counts as that. A run that
read and gave nothing back (an agent stopped or in an error, an empty answer) is not effective: it is not kept off
Claude, not saved, and the scale is measured without it. `usage` and `local_share.py` show both numbers. And **a
report of the worker's work is this session's own** (`usage --session`, its row in `local_share.py`), with both
numbers: never another session's or project's added in, never consumption passed off as work done.

## What to delegate (measured with Qwen3.8-27B Q4_K_XL on an RTX 3090)

| Task | Measured |
|---|---|
| Map a file (each function, one line) | 666 lines of C: 25 of 25 functions, 22 accurate, 40 s; 6,772 tokens read, 1,160 back |
| Who defines and who calls X (`agent`) | exact file:line, 2-4 turns, 13-25 s |
| Names for unnamed functions | with the file and two neighbours as examples: 4 of 12 exact, 4 near |
| Mechanical rewrites of code | 180 tok/s with ngram speculation, output identical to the plain run |
| Explain or summarise text | ~30 tok/s a request, ~118 tok/s in all in a batch of 8 |
| An agent over 2 folders, ~20 files | 20 turns, 11 min, exact; but alone on the server |

Delegate: maps of long files before reading them, searches across many files, volume naming and commenting (with
mechanical checks after), long logs ("what failed and why, with the lines"), first drafts. Also **first drafts of
code** Claude is about to write (a function, a script, a test, a mechanical patch), from an exact spec (what goes in,
what comes out, a neighbouring file as an example), for Claude to review, fix and test; and **a review of the diff**
before Claude's own, marking what looks wrong with file:line. These move the scale most. Not load: accents, spelling,
a single line, anything quicker to do than to hand over.
Do not delegate: decisions, stateful debugging, anything that needs several things weighed at once, anything that
goes to the repo without a mechanical check, text the user will read as is.

## Its mistakes (always check)

- **It copies details from the examples it is given**: with neighbour functions as examples it put the neighbour's
  numbers and names into its comments. Numbers, types and names it proposes are checked against the code.
- **It mixes up repetitive code**: three similar dispatchers got the same list of cases.
- **What it says about its own tools is not reliable.**
- Generic comments without style examples: give it two or three real comments from the same file.
- Bonsai 2 (the ternary profile): as good at reading, but on an exact edit it returned the code unchanged half the
  time. Edits go to the Q4 Qwen.

## Measuring the savings

Every `ask`, `batch` task and `agent` run adds a line to `usage.jsonl` (in the config's home): the command, the
caller's folder, the model, the tokens read/written/given back and the seconds. `py W usage [--since YYYY-MM-DD]`
adds them up by command, project and day; "kept off Claude" = what the model read less what it gave back (for the
agent an upper bound: it counts Claude Code's prompt on every turn). It ends with what that would have cost on
Claude at API list prices ("read": the tokens kept off Claude at the input price; "+write": the answers too, at the
output price, as if Claude had written them; the truth lies between). Call the worker from the project's folder so it
counts by project. Tests and benchmarks go with `CRANKER_USAGE_TAG=bench` so they stay out of the totals.

## If something fails

- `server not running`: `py W start`. If it does not start: `logs/server.stdout.log` in the config's home.
- `cut by max_tokens N`: raise `--max-tokens`, or ask it to think less. With a check, the worker already asked for
  it again with 1.5 times the room; `needs splitting` means even that was cut: fewer files or functions per task,
  or line ranges, not a bigger limit.
- `agent stopped: N turns in a row ended at the output cap`: what it had to write does not fit in a turn. One file
  per run, or have the code written by `ask`/`batch` with `--check`.
- "Context size has been exceeded": see "The shared context". `curl -s http://127.0.0.1:8090/slots` shows the slots.
- The agent's first request fails with HTTP 500 "System message must be at the beginning": the chat template was not
  patched. `start` patches it when `template` is `auto`; with another model whose template raises differently, see
  the README.
- Everything slow (a few tok/s): the GPU's memory is full and the driver is moving it to system RAM (on Windows).
  Fewer slots, less context, or close what else uses the GPU.
