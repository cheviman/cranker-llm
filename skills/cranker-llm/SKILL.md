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
object per line: `{"id", "prompt", "files": [...], "out", "max_tokens"?, "think"?, "system"?}`. Tasks whose `out`
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

Prompts to the worker in English. A prompt with double quotes or a long one goes in a file (`--prompt-file`):
Windows PowerShell 5.1 splits `-p "..."` with `\"` inside into several arguments.

## The shared context

The context (64K by default) is **one pool for all the slots** (`-kvu`), not 8K each. An agent's request alone is
32-40K tokens (Claude Code's system prompt, its tools and what it has read) and grows with every file it reads. If
the pool overflows, the server does not refuse the new request: it fails **every** running request (HTTP 500,
"Context size has been exceeded"). So:
- **An agent runs alone**: it starts with the server idle, asks and batches wait while it runs, and two agents take
  turns (a lock). For a search in a few files, Claude's own grep or an `ask` with the files is faster.
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
- `FINISH=length`: raise `--max-tokens`, or ask it to think less.
- "Context size has been exceeded": see "The shared context". `curl -s http://127.0.0.1:8090/slots` shows the slots.
- The agent's first request fails with HTTP 500 "System message must be at the beginning": the chat template was not
  patched. `start` patches it when `template` is `auto`; with another model whose template raises differently, see
  the README.
- Everything slow (a few tok/s): the GPU's memory is full and the driver is moving it to system RAM (on Windows).
  Fewer slots, less context, or close what else uses the GPU.
