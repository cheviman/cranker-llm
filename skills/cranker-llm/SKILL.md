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
- "local worker off" (`status`; the user ran `local off` to free the GPU's memory): **never start it or turn it
  back on yourself**. `ask` and `batch` go to the remote server with no flag, from `remote.roots` only; a task whose
  files may not leave is refused: do that part yourself. `agent` and `--local` are refused.
  Starting needs no permission when it is configured; stopping does: **never `stop` the server unless the user asks**:
  another session of theirs may be using it.
- `start` refusing for the GPU power limit (`max_power_w` in the config): tell the user; never pass
  `--ignore-power-limit` on your own.

**The read guard** (a hook): in the folders the user listed under `read_guard.roots` in the worker's config, a
whole-file Read of a text file over `max_lines` (300) is refused with how to map it with the worker. A Read with
`offset`/`limit` always goes through; when a file is really needed whole (to edit most of it), Read it with
offset 1 and limit its line count. Folders outside those roots are never touched.

**The scale** (hooks): the user wants the worker to do more of the volume than Claude. `ratio` local tokens for
each token of Claude (`scale` in the config, 5 by default: a million local for 200K of Claude) is where to aim, not
a quota, and nothing is blocked. Claude's tokens are the new ones of each turn (input, cache writes, output; not
cache reads). The worker's are measured by two yardsticks, kept apart:
- **consumed**: `in` + `out` of every request. The tokens are spent whatever came of them, and a task run four
  times is four runs;
- **effective**: the work that was right by the first or second try (`scale.tries`). A piece of work is an output
  and the files it is about, so sending it again, and each fix after a failed check, is one more try. Work that
  took more tries, or was left cut, failing its check or with nothing back, is not effective.

The scale is on the effective, so it is not reached by sending things again: it is reached by tasks that come out
right, from an exact spec (what goes in, what comes out, a neighbouring file as the example) and with a `--check`.
The hooks say so without the user having to: the rule when the session starts and after a compaction, a short
reminder at every prompt while Claude is doing more than the worker's effective work (below `scale.least`, 1 to 1),
and a longer note while working below the scale, at most every 15 minutes. When a note says the volume is there
and the quality is not, don't send more: send better. **The worker's part of a task goes out first**, in the
background, before reading or drafting that part: don't prepare a task for it by reading the files it is about.
`py scripts/local_share.py [--day YYYY-MM-DD]` (beside W) shows the day by session, by both yardsticks.

## The three ways to use it

**1. `ask`: a question about files, no tools.** The files go to the model, the answer to `-o`.
```
py W ask -f src/big.c --number -o /tmp/map.md -p "For each function, in order, one line: `name` (line N): what it does, concretely. Nothing else."
py W ask -f src/big.c:120-260 -o out.md -p "..."     # a line range
py W ask ... --think                                  # thinks first: slower, better on hard questions
```
It prints `-> out.md: in N tok, out M tok, s`. Read the output only if needed, and only the part needed.

`--number` (`"number": true` in a batch line) gives the model every line with its number in the file in front of
it: use it whenever the answer cites lines (a map, a search, a review), because without it the model counts the
lines by eye, and a line it gives is then a guess. It costs a couple of tokens a line. Never for a rewrite: the
numbers would come back in the code. A line the worker cites is still checked before it is used.

Exit codes: 0 done and right; 2 the output is not right (cut, looping, its check still failing, an agent stopped)
or the task was not run because the worker was taken, and the last lines say which; 1 an error.

With `--think` the thinking counts inside `--max-tokens`, and Qwen thinks a lot: a 260-line diff's review and a
647-line file's map spent 10,000 tokens thinking and came out empty. So every `--think` request carries a thinking
budget (`--max-tokens` less up to 2,048 for the answer): past it, the server closes the thinking and the model
answers. If one still comes back empty, the worker hands the cut thinking back and asks for the conclusion without
thinking; the summary then says **`RECOVERED ... (check it)`**, and that answer is checked before anything else
(it comes from half a reasoning). `--effort medium` or `low` (`"effort"` in a batch task) makes Qwen3.8 think less
than its default `xhigh`. Maps, summaries and searches: no `--think`. If it must think: `--max-tokens 20000
--keep-thinking`, and expect 10-15 minutes.

A **server elsewhere** takes no thinking budget, so nothing closes its thinking before `--max-tokens`, and its
`plain` fields may think too (`reasoning_effort: medium`). On 20-11-2026 three OutRunners tasks came back empty
after 64,000 to 200,000 tokens of it. Now: an empty answer with thinking is recovered whether or not `--think` was
asked, with the room of the request (not 4,096) and the end of the reasoning; and an answer that was all thinking
with none of it sent (deepseek at high effort sends it in one block at the end) is asked again once thinking one
effort less (`effort lowered xhigh -> medium` in its line), never with more `--max-tokens`, which buys more thinking.
`cut by max_tokens N with nothing written: all of it went to thinking` means even that failed: a narrower task, or
`--effort low`, not a bigger limit.

**2. `batch`: many tasks in parallel** (8 slots: ~118 tok/s in all on a 3090, against ~30 for one request). One JSON
object per line: `{"id", "prompt", "files": [...], "out", "max_tokens"?, "think"?, "system"?, "check"?,
"check_null"?, "check_good"?, "check_bad"?, "retries"?, "number"?, "context"?: [...], "variants"?: [...]}`. Tasks
whose `out` exists are skipped, so a batch can be stopped and resumed; `--force` redoes them. A task that left its
`out` cut or failing its check is not done: change its line (more `max_tokens`, `think`, a narrower prompt) and run
the same batch again, and only the changed ones run; unchanged, they are skipped and listed, so running the batch
again after a `not run` burns nothing. **A stopped batch is safe to run again**: every file is written whole or not
at all, and each task's `out` carries `<out>.running.json` until the task has ended and been recorded, so the next
run redoes (and lists) what a stopped one left half done, and an empty `out` with no failure mark; one that another
batch still running holds is left to it. The file is checked before any task runs, and an `out` is never one of the
task's own `files`: write beside the source and move it over once it is checked.

**`"variants"`**, instead of sending a task three times and picking the best by hand: `"variants": [{"role":
"draft"}, {"role": "review", "effort": "medium"}]` runs the task once per variant, side by side (each changes only
`prompt`, `system`, `role`, `effort`, `think`, `max_tokens`, `temperature`, `escalate`, `retries`), each into
`<out stem>.vN<suffix>` with its own fixes, and writes to `out` the one that passed its check with the best score:
the check's last line `score: N` (higher is better; print the measure negated when lower is), or the first that
passed when none prints one. It needs a check. The line names the one kept and how each ended. In `usage` the task
is one piece of work: the kept variant's runs are its tries, and the others are consumed, not effective (`were
variants not kept`): variants are for a task worth paying for several times, not a way to raise the scale.
```
py W batch tasks.jsonl
```

**`--remote`** (`ask` and `batch`) sends that run to the server under `remote` in the config instead of the local
one, so the files leave the machine. **Only when the user asks for it**, never because the local server is down,
busy or slow: say so and let them decide. The worker itself refuses it outside the folders the user listed in
`remote.roots`, for every file it reads into the request: with `--remote`, the prompt file, the task file and the
rest go in one of those folders too, not in the scratchpad. It reads the key from an environment variable the user
sets; never ask for the key or write it anywhere. `status` shows whether a remote server is configured.

The user asking in their own words ("hacelo en la nube", "mandalo remoto", "send it to the cloud") is asking for
`--remote`: that run goes there whatever the local worker's load. A server there that refuses a streamed request
(`stream`, `stream_options`) needs `"stream": false` under `remote`; that is the user's config, and without the
stream a loop there runs to `--max-tokens`.

**`remote.spill`** is the user's standing yes for the rest: with `"spill": true` in the config, an `ask` that finds
the local worker full (an agent has it, no room in the context, or the server is down) goes to the remote by itself
when all its files are under `remote.roots`, and a batch runs on both at once (while the local worker runs one of
its tasks, the next go there as far as `remote.parallel` allows). Nothing to pass for it: send work as always, and
do not hold back a batch because the local worker is busy. `--local` keeps a run here (a client's code that sits
under a listed folder). Turning spill on, or listing a folder, is the user's call, never yours.

**`remote.roles`** is the user's choice of a model for each kind of task (`status` lists them):
`--role NAME` on `ask`, `"role"` in a batch line (`batch --role NAME` for the lines without one) sends that task to
the role's model, with no `--remote`, and only from the folders under `remote.roots`; a role may also be `local`.
Claude orchestrates: it writes the spec and the check, sends the draft to the role for drafts and its review to a
role of **another family** (a model reviewing its own draft tends to agree with it), then reads the diff and the
findings, drops the false ones and decides; the commit stays Claude's, with the user's yes. With a check,
`--escalate NAME` (`"escalate"` in a line, `remote.escalate` for all) sends the fixes to that role once the first
answer failed (`remote.escalate_after`, 1: its second try, which still counts as effective), so a bigger model is
paid only for what the first one could not get right. A fix that goes elsewhere carries the check's output too:
`remote.escalate` skips a task whose files may not leave, and a check that prints files from outside `remote.roots`
is not one to send there. The line says `MODEL as ROLE`, or `fixes from fix N on by MODEL as ROLE (escalated)`;
`usage` prices each model apart. Use the roles that are listed and never invent one;
adding a role or changing its model is the user's call: a role's model may bill their credits.

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
  the user's repo: say so when it is created. `--brief FILE` uses another one, `--brief ""` none. A run without one
  whose tasks write code (an `out` that is not a document) or ask for changes (patch, diff, change, fix, apply...)
  prints `no brief: ...` on stderr: on 20-11-2026 OutRunners had none, nothing said so, and a batch of designs
  came back with formats made up. Write the brief then, or say why not, before sending the batch again.
- **The check**: `--check "CMD {out}"` (or `"check"` in a batch line), a command that exits 0 when the output is
  right: a compile, a test, a style script. The worker runs it and hands the model its errors until it passes
  (`check_retries`, 2; from the second fix on it thinks first). What comes back to Claude is only what still fails:
  `CHECK FAILED` and the errors in `<out>.check.txt`. A cut answer is asked for again with 1.5 times the room, once,
  and its line starts `cut by max_tokens N`. **Don't run fix rounds by hand** (compile, read errors, write a
  prompt for each one, send again): that's the check's job. A check runs in the caller's folder, several at once in
  a batch, so it writes only files of its own (one object file per task, say). A check that runs the draft (a test
  script, a program) builds the folder it runs in again on every run, never reusing one a previous round's draft
  could have changed: a draft test's cleanup once deleted the project's `main.c`, and every later round failed on the
  missing file instead of on its own errors. Every round that fails keeps its check's output in
  `<out>.check.N.txt` (N from 1), not only the last: an early round may show a fault of the machine the later ones
  hide. An answer that is one fenced block and nothing else (`` ```c ... ``` ``) is written without its fence, so a
  check doesn't need to strip it. For a code file (any `out` but a document: `.md`, `.txt`, `.rst`, `.log`, none;
  `CMakeLists.txt` and `Makefile` are code), an answer with exactly one fenced block and text before or after it has
  the block written to `out` and the text to `<out>.notes.md`; its line ends `the text around the code kept apart
  in ...`. **Read the notes when the line says so**: on 20-11-2026 they reported a stale toolchain path on the machine
  the check ran on, which no fix of the file could get past. Two blocks or more are written as they came, and the
  line says how many: which one is the file can't be told, so narrow the task to one file.
- **The check is tried before it is trusted.** Before any request, the worker runs it on an output that changes
  nothing, which it must fail (an empty one by default; `--check-null FILE`, `"check_null"`, gives the real null:
  the file as it is now, an empty patch), and with `--check-good FILE` (`"check_good"`) on a known-good output,
  which it must pass. A check that passes the null or fails the good stops the run with exit 1 and its output,
  nothing sent: fix the check, not the task. So **a check that only compiles is refused** (an empty file
  compiles): make it also run the code, or look for what the change must produce. `--check-null none` only when an
  empty answer can be right. The probes run on `<stem>.cranker-probe<suffix>` beside the `out` (deleted after), so
  a check that counts its runs or logs what it saw sees them too. A check with no `{out}` in it can't be tried on a
  copy and is refused too: put `{out}` in it, or run it untried with `--check-null none`.
- **A known-bad output is tried too**: `--check-bad FILE` (repeatable; `"check_bad"`, a file or a list), which the
  check must fail. Failing the null is not enough: on 20/21-11-2026 three OutRunners checks failed it and still
  passed wrong work (one compared folders that differed, the output lacked `src/sound`; one compiled with `gcc -w`,
  which hides a pointer put in an integer; one measured and compiled a patch that then broke the game). Keep the
  output that got through, and pass it as a bad one from then on. **A check of changes runs the result**: a test or
  a scenario of the program, on the patched copy (its own `sys.path`, the same folders on both sides, warnings that
  matter as errors), not only "it applies, compiles and measures". OutRunners' `build/p2r4/check_group.py` is the
  model: it applies the blocks to a copy, decompiles, measures, compiles with pointer/integer errors on, and runs a
  scenario under a lock.
- **What all the tasks share is written once**: `batch --prompt-prefix rules.txt` or `--system-file`,
  `--context` (the files, next section), and `--check` (with `--check-null`/`--check-good`) for all of them; a
  line's own value wins. The line then holds only what is its own.

The check proves what it checks, and nothing more. What passes still gets read in the diff like any draft, and an
output without a check is checked by Claude as before. `usage` shows the rework by project: first time / by the
worker / for Claude. When "for Claude" stays high, the prompt, the brief or the check is what needs work, not
more rounds.

## Before a wide batch: the context pack

A worker sees only its files and its prompt. On 20-11-2026 a batch of 14 designs for OutRunners went out with 4
sites of C each, and came back with `symbols.txt` lines in formats nobody uses, data addresses where instruction
addresses go, a structure that does not exist, and "no change in the core" said without seeing the core. The
model was not the cause: the format was in the parser, the instruction addresses in the listing, the solved cases
in earlier rounds, and none of it was in the request. So before a batch of design or change tasks (more than a
handful, or any that writes code or a plan to apply), build a **context pack**, a folder that goes to every task
with `--context DIR` (or `"context": [...]` in a line; files, `file:a-b` ranges and folders, which give their files
in name order). **Ranges of code by name, not by line**: `file.py::name`, `file.py::Class.method`, `file.c::name`
(C/C++ functions, `struct`/`enum` blocks, `Foo::bar`) or `file.py::first..last` (from one to the end of the other)
are found when each request is put together, with the comments right above them; a `file:a-b` written at the start
was stale by the next step on 21-11-2026 (the step before had added lines) and left out the very function that step
was about. A name that is not in the file stops the run before anything is sent, with the names close to it. The
same works in `-f` and a line's `"files"`:

1. **The formats, from the code that reads them**: the parser by name (in OutRunners,
   `tools/core/disassemble.py::load_names`), never a format retyped from memory in the prompt. What the reader
   accepts is the format.
2. **The mechanisms the change goes through, with their code ranges**: how the core types a pointer, where a
   local is declared, what decides a register. The task may then say what the core lacks, with the line.
3. **2 to 4 solved cases, each with its before and after** (a diff of an earlier round, or the two versions):
   what a right answer looks like in this project, and the size of one.
4. **The data the task needs**: the listing ranges with the instruction addresses, the tables, the current
   `symbols.txt` lines of the names involved. A worker that has to quote a value must have it in front of it.

Then the prompt: what it may not invent ("if what you need is not in the files, name what is missing instead of
guessing"), what it cannot know ("say `unknown` for the core unless its code is in the files"), and the answer's
form. And **the check that measures the real result**: the metric or the test the change must move, run on the
patched copy, not "it applies and compiles". Try it by hand before the batch with a change that does nothing (it
must fail; that is `--check-null`) and with a known good one (it must pass; `--check-good`): on 20-11-2026 a check
that only applied and compiled passed a patch that changed nothing, and one that decompiled "with the patched
copy" ran the real code (the script put `tools/core` first in `sys.path`), so the worker failed three times
against code that never had its patch. The worker tries both again before the first request.

The pack goes first in every request (after the system prompt and the brief), so it is the same prefix in every
task: a server that caches prompts reads it once. The batch's first line says its size (`context N files (X
chars) first in every request`); it counts in every request, so keep it to what the tasks need, in ranges, and
split a task whose answer will not fit (the biggest OutRunners design was cut thinking at 48,000 tokens: it held
too many uses for one answer). The pack's files are not what the work is about: a task sent again with a pack is
the same piece of work, its next try.

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
- **One session per round.** A long session costs more on every turn (all it holds is read again), and by the third
  hour of the OutRunners session of 20/21-11-2026 each turn carried hours of logs. When a round is done (its results
  checked, its notes written to a file), the next round starts in a new session from those notes.

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
5. **When a careful try changes nothing, debug: that is Claude's.** A rule Claude gives the worker ("type it as a
   pointer and the use goes away") is a hypothesis until it is seen working. If a well-specified task, or a variant
   with more thinking, moves nothing, the cause is not in the prompt: find it by running the code on one case,
   printing what it does, reading where it decides. Sending more tasks on the same hypothesis spends the batch
   again; in OutRunners three batches went that way before a five-minute script showed what could not move.

**Before a batch of changes, measure the reach.** On 20/21-11-2026 three batches of designs and patches (25, 14 and
14 tasks) went out against a measure that was mostly out of reach; a script over the artifact that counted what each
change could move said so afterwards, in five minutes. Before a wide batch of changes, write that script (or ask the
worker for its draft): how much of the measure each kind of change can reach, and what none can. Send tasks only for
what is reachable, narrowest first. A batch that runs 8 or more tasks that write changes for the first time says
`wide batch of changes: ...` on stderr; `--reach FILE` (that measure) says it was done, and the batch names it in its
first line.

**Consumed is not effective.** Everything the worker reads and writes is consumed and counts as that, every run of
it. Effective is another yardstick: the work that was right by the first or second try. A run that gave nothing
back (an agent stopped or in an error, an empty answer), a cut answer, an output that still fails its check, and
work that took three tries or more are not effective: not kept off Claude, not saved, and the scale is measured
without them. `usage` and `local_share.py` show both numbers, and the pieces of work by the try they were right at.
A right piece is only as right as its check: `usage` splits the right ones into those that passed a check tried
before it ran, those whose check was never tried (`--check-null none`, or older runs) and those with no check
(right only because nothing sent them again), and its table of checked tasks has a `tried` column. Many right
"at the first try" on checks never tried, or with no check, is not quality yet: say so in a report.
And **a report of the worker's work is this session's own** (`usage --session`, its row in `local_share.py`), with
both numbers: never another session's or project's added in, never consumption passed off as work done.

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
  or line ranges, not a bigger limit. `... with nothing written: all of it went to thinking` (a server elsewhere):
  not more room, less thinking (`--effort low`, no `--think`) or a narrower task.
- `the check took more than N s, and was stopped`: the check (and every process it started) ran past
  `check_timeout`; a check that hangs is a fault of the check, or of the output it runs (an endless loop).
- `REPEATING: the answer went round in a loop (...)` (`LOOP` in a batch): the worker reads every answer as it is
  written (local and `--remote`), and one that writes the same lines over and over, numbers aside (`file:N: fine.`
  for every N), or the same passages word for word, is stopped there, as not effective, and a check does not hand it
  back as a fix. **Don't raise `--max-tokens`**: the room went to the loop. Send it narrower, and say what to leave
  out ("only the defects; nothing for the lines that are fine"), or `--think`. A task that asks for more than 30
  lines of one form it was not given (the same line for every N) is stopped too: ask for a range or a script that
  prints them. Code that needs a repeated text (a test's input) builds it in code (`text * 20`): a test draft asked
  for "a paragraph repeated 20 times" wrote "and again and again..." for 22,500 tokens (20-11-2026).
- `agent stopped: N turns in a row ended at the output cap`: what it had to write does not fit in a turn. One file
  per run, or have the code written by `ask`/`batch` with `--check`.
- "Context size has been exceeded": see "The shared context". `curl -s http://127.0.0.1:8090/slots` shows the slots.
- The agent's first request fails with HTTP 500 "System message must be at the beginning": the chat template was not
  patched. `start` patches it when `template` is `auto`; with another model whose template raises differently, see
  the README.
- Everything slow (a few tok/s): the GPU's memory is full and the driver is moving it to system RAM (on Windows).
  Fewer slots, less context, or close what else uses the GPU.
