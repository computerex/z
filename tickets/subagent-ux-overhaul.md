# Subagent UX Overhaul — Design & Implementation Plan (v3)

## 0. Problem statement

The subagent subsystem is functionally sound (isolation, notification-loop guards, batch
spawns are all regression-tested) but the **interactive UX is broken**. Concrete failures
reported by a real user, with root causes verified in code:

| # | Symptom | Root cause |
|---|---------|-----------|
| 1 | To watch a running agent you must `/agents`, copy the name, then `/agent <name>`; `/agent-back` to return. | No keybinding, no index selection, no tab-completion of agent names. The Ctrl+E cycling in `docs/sub_agent.md` §7 was never implemented. `HarnessCompleter` completes commands/paths, not agent names. |
| 2 | Focused on a subagent running a cmd; **Ctrl+B does nothing**; Esc does nothing. | (a) `KeyboardMonitor` (`interrupt.py`) is only enabled during parent turns — while a subagent runs, no thread is even consuming keys. (b) Even enabled, `tools/shell.py:31 _owner_interrupts()` blocks subagents because they run with `enable_interrupt=False`. The design conflated *unobserved background* with *user-is-watching*. |
| 3 | Typed `hi` + Enter while the focused agent was busy → **terminal appeared frozen**. | `main.py:2977` blocks in `loop.run_until_complete(run(...))`, and `SubAgentManager.run()` (sub_agent_manager.py:322) **silently awaits the in-flight task first**. During the await: no prompt, no status line (subagents run `enable_status_line=False`), no message, no cancel. |
| 4 | Ctrl+C (twice) **terminated the session**; after restart **the subagent was gone**. | 1st Ctrl+C sets the global interrupt flag — the subagent ignores it (same gate as #2), so *nothing visibly happens*; 2nd Ctrl+C hard-exits. Sessions **are** written to `.sessions/_sub_<name>.json` — there is simply no restore path. |

### Design principles for the fix

1. **Interrupt what I'm watching.** Keyboard signals (Esc/Ctrl+B/Ctrl+C) route to the
   agent the user is focused on; unobserved background agents stay isolated.
2. **Never block silently.** Any wait must either stream visible output or show state.
3. **Nothing is lost on exit or crash.** Subagent state survives restarts.
4. **No copy-paste in the terminal.** Every agent is reachable by one chord or a number.

A key simplification discovered while refining: once input is queued instead of blocking
(v1's Phase 1.3), a focused agent's turn always runs *while the prompt is up* — so there
is **no new "monitor while prompt dismissed" state to build**. The keyboard-ownership
invariant becomes the one that already holds for parent turns:

> **The `KeyboardMonitor` runs only while the prompt is dismissed (parent turns). While
> the prompt is up, prompt_toolkit owns the keyboard; all agent control flows through
> prompt bindings that set per-agent signals.**

This invariant is platform-critical on **both** console families: on Windows the
monitor's `ReadConsoleInputW` and prompt_toolkit read the same console input buffer; on
Unix the monitor's cbreak raw reads and prompt_toolkit race for the same `stdin` fd.
Following it removes v1's riskiest component (input contention) on every OS.

---

## 1. Phase 1 — Fix the interaction model

### 1.1 Per-agent signal events (replaces global-flag surgery)
**Files:** `sub_agent_manager.py` (new `AgentSignals` on `SubAgentInstance`), `cline_agent.py` (`agent_signals` attr + loop-level check), `tools/shell.py` (signal consumption). `interrupt.py` itself is untouched by this item — its only change is the parent-turn press-2 path in 1.5.

- Each `SubAgentInstance` gains `signals = AgentSignals(interrupt: bool, background: bool)`
  with consume-on-read semantics.
- **Plumbing:** `ClineAgent` gains an optional `agent_signals` attribute injected by
  `SubAgentManager.create()` (None for the parent). The shell tool consults
  `owner_agent.agent_signals` first (focused subagents), then falls back to the global
  flags when `owner.interrupts_enabled()` (parent) — so the global `InterruptState`
  mechanism stays untouched.
- **Two interrupt surfaces, both wired to the same signal:**
  1. *Within a running command* — the shell tool's tail loop checks the agent's own
     signals (kill / promote to background), exactly like the parent's path.
  2. *At turn level* — the agent loop's existing per-iteration interrupt checks
     (`_run_loop`) additionally consult `agent_signals.interrupt`, so a signal stops
     the whole turn (not just the current cmd) between tools, mirroring parent Esc.
- **Signal latency expectations** (documented, not engineered around): signals take
  effect at boundaries — the shell poll (~0.15s), loop iteration boundaries, and API
  call boundaries. A mid-stream HTTP response is *not* aborted (same as parent Esc
  today); the turn stops after the current response completes. This is deliberate —
  aborting in-flight API streams is a different, riskier feature.
- Who sets subagent signals: the prompt bindings (1.2) and Ctrl+C stages (1.5), always
  targeting the *focused* instance. Stale-flag hazard is structurally eliminated in two
  ways: signals are **per-instance** (can never leak to another agent) and **per-turn**
  (the drainer clears them when the turn ends, bounding "background the next command"
  semantics to the current turn).
- Additive rule: during a parent turn, Esc sets the global flag (parent consumes, as
  today) **and** the focused agent's signal if one is focused and running — "Esc stops
  what's happening."

### 1.2 Prompt-side control bindings
**File:** `main.py` (`create_prompt_session`)

New optional callbacks, wired like the existing F2/F3/Ctrl+T ones:

- **Ctrl+B** (`c-b`): if a focused subagent has a live task → set its
  `signals.background = True` (its running cmd promotes via the existing
  `_promote_to_background`); otherwise fall through (binding inert).
  *Platform notes:* prompt_toolkit's emacs default for `c-b` is backward-char —
  overriding it is acceptable at a one-line REPL (arrow keys still work). **Inside
  tmux with the default prefix, Ctrl+B is swallowed by tmux before the app sees it** —
  keep the chord anyway (it matches the existing during-turn `Ctrl+B background` hint
  the harness already advertises, and tmux users who remap their prefix are unaffected),
  and document the caveat; the number-driven `/agents` + `/pause` paths cover affected
  users.
- **Ctrl+E** (see 2.1) — focus cycling; overrides prompt_toolkit's emacs end-of-line
  default, acceptable at a one-line REPL; **F4 alias** for terminals that eat chords
  (F-keys pass through conhost, Windows Terminal, Terminal.app, iTerm2, and tmux
  untouched).
- Bare Esc stays **unbound** at the prompt (needed by `Escape+Enter` newline; escape-
  timeout disambiguation adds latency to the most urgent key). "Stop what I'm watching"
  is Ctrl+C stage 1 instead — safer than Esc.
- Bindings must not print from inside the handler: mutate state, `event.app.invalidate()`,
  defer richer output via the existing `pt_run_in_terminal` helper.

### 1.3 Non-blocking turn queue (kills the freeze)
**File:** `sub_agent_manager.py`, `main.py`

- Each instance gains `pending: deque[QueueEntry]` where
  `QueueEntry = (source: "user"|"model", text, future?)`.
- `queue_input(name, text)` — REPL focused path (`main.py:2977` replaces its blocking
  `run()` call): if idle, start a turn immediately; if busy, append and print
  `⧗ 'x' busy — input queued (N pending)`. The prompt returns instantly; the queued
  turn's output streams live through the already-active TeeWriter.
- **Drain semantics:** when a turn completes, the manager starts a drainer coroutine:
  consecutive `user`-source entries are **coalesced into one turn** (they were all
  authored before the turn ran — conversational fragments; saves API turns); each
  `model`-source entry (`send_agent_input`) gets its **own turn** with its own result.
- **`send_agent_input` de-blocking:** if the target is idle → run the turn, return the
  reply (unchanged synchronous chat). If busy → enqueue, return
  `"Queued (position N). You'll be notified on completion; use get_agent_output(name)."`
  — the parent freezes on a tool call no longer. Optional `wait=true` param restores the
  old blocking behavior explicitly.
- The self-notification-refusal check moves into the queue path: `[SYSTEM: ...]`
  messages naming this agent are dropped, never queued.
- `pause` and `delete` flush the queue (and resolve waiting futures with a cancelled
  marker so `send_agent_input(wait=true)` callers don't hang).
- Focus is irrelevant to draining: unfocus mid-queue → turns continue in background,
  tee buffers, notification fires, replay-on-refocus covers the gap.

### 1.4 Live state in the prompt bar (never look frozen)
**File:** `main.py` (`_build_prompt_text`)

- The focused tag becomes stateful: `[agent:x ⧗42s]` while its task runs, `[agent:x ✓]`
  when done, plus `· N queued` when a queue exists. Colors: running=yellow, done=green.
- Implementation: a 1s ticker coroutine scheduled alongside the existing prompt-race
  (`asyncio.wait` already runs there) calling `app.invalidate()` — the prompt message
  function re-renders the elapsed time. No extra prints, no screen noise.
- Ambient badges (2.4) live in the same function.

### 1.5 Ctrl+C staging — in the REPL handler, not the signal handler
**File:** `main.py` (prompt `KeyboardInterrupt` catch), `interrupt.py` (parent-turn path only)

prompt_toolkit already converts prompt-time Ctrl+C into a caught `KeyboardInterrupt`
(the double-tap-exit flow at `main.py:1731`), so the stage machine slots in with **zero
signal-handler printing** (no garbled-screen risk):

1. **Stage 1** — focused agent has a live task → set its `signals.interrupt`, print
   `[STOP] interrupted 'x'`, continue the loop. (Without a focused live agent this
   stage is skipped.)
2. **Stage 2** — any *active* subagents exist (running or queued) → print
   `N sub-agent(s) active — sessions are saved and resumable. Ctrl+C again to exit.`,
   continue. If the registry holds only completed/errored agents, skip this stage —
   a finished-only registry must not slow the legacy double-tap exit.
3. **Stage 3** — exit (existing `cleanup_and_save()`; with Phase 3 this is fully
   non-destructive).

The staged machine engages **only when subagents exist**; with an empty registry the
legacy double-tap-exit behavior is preserved. Stage counter resets each REPL iteration.

Parent turns (prompt dismissed) keep today's SIGINT behavior for press 1; press 2
currently hard-exits the process, which is hostile while subagents run — change it to
set a flag the agent loop checks at the next iteration boundary → graceful
`cleanup_and_save()` + exit.

### 1.6 Notify on error (existing zombie defect)
**Files:** `sub_agent_manager.py` (`check_completed`/`peek_completed`), all injection sites.

Today `check_completed()` matches only `status == "completed"` — **an errored subagent
never notifies anyone**, and the parent waits forever on a zombie unless it happens to
poll `list_agents`. Fix: match `completed OR error`; the notification text for errors
embeds `last_error` and suggests `list_agents`/`get_agent_output`. This is a bug fix on
its own, not gated behind UX work.

### 1.7 Result log with consume-on-fetch (existing overwrite defect)
**File:** `sub_agent_manager.py`

`_run_agent_task` overwrites `instance.final_result` and resets `completion_notified` on
every turn — if turn 2 completes before turn 1's notification was consumed (parent busy
in a long turn), turn 1's result is silently lost to the ANSI transcript. Replace with a
monotonic per-agent **result log**: `deque[(turn_id, result)]`. Notifications become
"has new output"; `get_agent_output`/`send_agent_input` return-and-clear everything
since the caller's last fetch (or since a cursor the notification carries). Immune to
any turn interleaving.

---

## 2. Phase 2 — Fast focus UX

### 2.1 Ctrl+E / F4 focus cycling
- Cycle `parent → agent1 → … → agentN → parent` (running agents first in listing
  order). Handler: switch focus via the same code path as `/agent` (tee replay +
  activate), then `app.invalidate()`; the prompt-bar tag is the immediate feedback.
- Empty registry → no-op.

### 2.2 `/agents` numbered rows + `/agent <n|name>` + picker
- `/agents` gains a `#` index column; sort order: focused → running → completed/
  interrupted → error. Rows show status icon, elapsed (frozen at completion for done
  agents), **last-output age** (`tee` write-timestamp: distinguishes working from hung),
  and per-agent cost (4.1).
- `/agent` accepts index or name (index wins for numeric-only agent names — discourage
  purely-numeric names in `create_agent`'s result text). No-arg `/agent` prints the
  numbered list + hint (one step from choosing). A `pt_run_in_terminal` radiolist
  picker is a stretch goal — the F2/F3 pickers are currently stubs, so don't build new
  dialog machinery for this.
- `/agent-back` stays; `/back` alias. **`/help` is updated** with the new commands
  (`/pause`, `/kill`, `/back`, `/agents --watch`) and the keybinding cheatsheet
  (`Ctrl+E/F4` cycle, `Ctrl+B` background focused agent, staged Ctrl+C).

### 2.3 Tab-completion of agent names
**File:** `main.py` (`HarnessCompleter`)

- `HarnessCompleter.__init__` gains `agent_names_fn: Callable[[], list[str]]` wired to
  the manager. Input `/agent <partial>` completes live agent names (focused first).
  Kills the copy-paste flow: `/ag` Tab → `/agent ` Tab → pick → Enter.

### 2.4 Ambient badges + completion preview
- `_build_prompt_text` appends, when subagents exist: `⚡2` (running count, cyan) and
  `✓1` (completed-with-unfetched-result, green). Use a cheap `stats()` on the manager
  (dict walk, **no** snippet building — `list()` reads/strips tee buffers and is too
  heavy per keystroke render).
- The `☎ Sub-agent 'x' completed` line gains the first ~100 chars of the ANSI-stripped
  result (single line, `…`) — often enough to skip switching entirely.

### 2.5 Direct user control of background agents
- `/pause <name>` → `pause_agent` semantics (flushes queue); `/kill <name>` →
  `delete_agent` semantics **with confirmation** (`/kill <name> --yes` or y/N prompt).
- No `/resume` — resuming is just focusing and typing; a resume command would imply
  more state than exists.
- `/agents --watch` (opt-in): live-refreshing table every 2s until Ctrl+C (caught
  locally inside the watch loop — it never reaches the staged Ctrl+C machine or the
  REPL exit). No unfocused streaming ticker — that's noise, not visibility; the badges
  + ☎-with-preview are the ambient layer.

---

## 3. Phase 3 — Persistence / crash safety

### 3.1 Registry
**File:** `sub_agent_manager.py`

- `.sessions/_subagents.json`: `{"agents": [{name, safe_name, status, session_path,
  created_at, completed_at, final_result, last_error}]}` — written atomically
  (temp + `os.replace`) on every state transition. `final_result` inline means a
  restored **completed** agent answers `get_agent_output` with **zero hydration**.
  Windows note: `os.replace` fails if the target is open — wrap with a short
  retry loop (writer contention is the registry itself, so this is cheap belt-and-
  braces; session loads elsewhere already tolerate transient locks).

### 3.2 Lazy-hydrated restore
- On startup, the manager loads registry metadata only. `ClineAgent` construction is
  expensive (`WorkspaceIndex.build()`, SmartContext, TodoManager) — restoring N agents
  eagerly could add seconds to boot. Hydrate (construct + `load_session()`) on first
  real interaction: focus, `send_agent_input`, or a turn.
- After hydration: `_initialized = True` (so `run_message` doesn't prepend a second
  system prompt); `load_session` already re-injects/repairs the system prompt.
- **Tool-call stitching (must-have, with its own test):** an agent killed mid-turn has
  history ending in `assistant(tool_calls=[...])` without matching `role="tool"`
  results — providers reject that with 400 (the codebase already synthesizes results
  for ignored calls for exactly this reason). On restore, post-process the loaded
  history: append synthetic `"[Interrupted before execution — this tool call was not
  executed.]"` for each dangling `tool_call_id`; drop orphan tool results.
- Corrupt session file → `load_session` returns False → skip with warning; registry
  entry retained until `/agents --purge`. Name collision with a live agent → live wins,
  restored entry skipped (logged).
- Since the tee transcript is memory-only, a restored agent has no live transcript:
  focus prints `(restored session — transcript not retained)` plus a plain-text tail
  if 3.3 is present. Statuses: `completed` (registry) vs `interrupted` (mid-turn at
  exit — `cleanup()` writes this before cancelling).
- **Not restored (documented):** background-proc bookkeeping
  (`ToolHandlers._background_procs`) — OS-level background processes from the previous
  run are no longer tracked; their `.harness_output` log files remain readable from
  disk, and `check_background_process` reports not-found for stale IDs. The restored
  agent's history may reference them; that's benign.

### 3.3 Transcript tails on disk
- On save points (completion/pause/exit), write the last ~64KB ANSI-stripped transcript
  to `.sessions/_sub_<safe_name>.log`. Cheap; makes restored agents replayable.

### 3.4 Startup integration
- Banner line: `♻ Restored 2 sub-agent session(s) — /agents`.
- One `[SYSTEM: N sub-agent(s) restored from the previous session; statuses: ...]`
  message injected via the queued-prompts mechanism (generalize `_queued_cron_prompts`
  into a system-message queue) so the parent reliably learns it can resume
  orchestration.

### 3.5 Housekeeping
- `delete_agent` removes the registry entry + `_sub_<name>.*` files (destroy means
  destroy). `/agents --purge` clears non-live entries. Cap restore scanning to
  reasonable counts (e.g., warn > 20).

---

## 4. Phase 4 — Polish

- **Per-agent cost** in `/agents`: accumulate per-response usage on the `ClineAgent`
  as the streaming client reports each response's usage (verify where the client
  surfaces it before wiring — the global tracker aggregates too early to partition).
  `get_global_tracker()` stays process-global for the session total.
- **Cost-bomb guard:** `create_agent` result includes a soft warning past ~8 live agents
  ("N agents running; each consumes context independently"). No hard cap (the documented
  no-limit decision stands).
- **Tee buffer cap:** `TeeWriter.buffer` is unbounded — cap at ~2MB, keep tail with a
  head marker, so long-running background agents can't balloon memory.
- **Remote parity:** new messages (queued ack, restore banner, ☎ preview) echo via
  `_echo_remote`.
- **Tool-description updates** (`tool_registry.py`), so the model's mental model matches
  the new mechanics: `send_agent_input` (queue-when-busy ack instead of blocking),
  `get_agent_output` (drains new results since last fetch), `create_agent` (soft
  cost warning), `list_agents` (mentions `interrupted` status).
- **Docs:** rewrite `docs/sub_agent.md` §7 to match reality (it currently describes an
  unimplemented Ctrl+E and pre-`get_agent_output` flows).

---

## 5. Existing defects fixed by this work (regression-tested)

1. **Zombie agents:** errored subagents never notify (1.6).
2. **Silent result overwrite** across overlapping turns (1.7).
3. **Unbounded tee memory** (4).
4. **Dangling tool-call history** breaking resumed conversations (3.2).
5. **Hostile second Ctrl+C** during parent turns while subagents run (1.5).

## 6. Acceptance criteria

**Phase 1**
- Focused agent running a cmd: Ctrl+B backgrounds it; Ctrl+C stops it; typing while busy
  queues (never blocks); prompt shows live `[agent:x ⧗Ns]`; queued turn output streams
  above the prompt. Unfocused agents ignore all of the above.
- Errored agent → parent notified within one loop iteration.
- Two overlapping turn completions → both results retrievable.
- Ctrl+C with a completed-only registry still exits on the second press (legacy
  behavior preserved); with an active registry it takes three deliberate presses,
  each with visible feedback.

**Phase 2**
- Focus any agent: ≤ 2 keystrokes (Ctrl+E cycles; `/agent 2` selects; Tab completes).
- Prompt bar shows running/done counts without switching.
- `/pause`, `/kill <name> --yes` work without asking the model.

**Phase 3**
- Hard-kill the harness mid-subagent-turn (SIGKILL on Unix, `taskkill /F` on Windows);
  restart → agents listed with correct statuses; conversation continues via typing or
  `send_agent_input`; no provider 400s from dangling tool calls; completed agents'
  results available instantly. Verify once per OS family.

**Phase 4**
- `/agents` shows per-agent cost and last-output age; 9th `create_agent` returns a
  warning; memory of a 10MB-output agent stays bounded.

## 7. Cross-platform considerations (Linux / Windows / macOS)

The design deliberately **reuses the existing platform machinery instead of adding new
platform-specific code** — the monitor (`ReadConsoleInputW`/msvcrt on Windows,
cbreak+select on Unix), the SIGINT handler, and prompt_toolkit are all already
cross-platform and stay untouched in their current roles. Residual platform-specific
concerns:

1. **Keyboard chord coverage.** `c-b`/`c-e`/`F4` emit correctly on conhost, Windows
   Terminal, Terminal.app, iTerm2, and common Linux terminals; the only systemic
   swallow is **tmux's default Ctrl+B prefix** (see 1.2) — mitigated by the F4 alias,
   `/agents` numbered selection, and `/pause`. Escape-in-tty latency (VT sequence
   disambiguation) is the reason bare Esc stays unbound — that's terminal-agnostic.
2. **SIGINT delivery.** prompt_toolkit converts prompt-time Ctrl+C to a caught
   `KeyboardInterrupt` on all three OSes — the Ctrl+C stage machine lives entirely in
   the REPL's existing catch and prints **after** the prompt app exits, so no
   signal-handler printing ever garbles the screen (also matters on Windows, where
   `signal.signal(SIGINT)` handlers run between bytecodes on the main thread). The
   parent-turn press-2 graceful path is a plain Python flag checked at an iteration
   boundary — no signals, no re-entrancy.
3. **The monitor invariant is Unix-critical too.** On Unix, the monitor holds stdin in
   cbreak and reads it; prompt_toolkit reads the same fd. "Monitor only while the
   prompt is dismissed" prevents the fd race on all platforms, not just the Windows
   input-buffer contention (§0).
4. **Process-tree kill** on signal-consume (interrupt of a running cmd) goes through
   the existing `_kill_proc`/`kill_process_tree` (psutil), already ported per-OS.
5. **Glyph vocabulary.** The prompt bar already relies on 256-color ANSI + glyphs
   (`❯ ☎ ✓ ✗ ▶ ○`) — a de-facto Windows Terminal/VT-mode prerequisite. New glyphs
   (`⧗ ⚡ ♻`) must be smoke-checked on conhost and macOS Terminal; if any render as
   tofu, fall back to the existing vocabulary (`• …`).
6. **Paths & atomicity.** `pathlib` everywhere; `os.replace` semantics per 3.1.
   Sub-agent name sanitization already strips `/` and `\` for filenames.
7. **Tests must be platform-neutral**: follow the repo's existing pattern
   (`test_shell_wrap_unix.py`) — monkeypatch `platform.system()` for OS-conditional
   code paths, `skipif` markers for POSIX-only end-to-end checks, and exercise the
   signal/queue/restore logic through direct API calls, **never** by synthesizing
   real console key events (impossible to do portably in CI).

## 8. Implementation order & risks

1. **1.6 + 1.7** (defects) — small, independent, immediate value.
2. **1.1–1.4** (signals, bindings, queue, live bar) — the core UX fix.
3. **1.5** (Ctrl+C staging) — after 1.1 lands.
4. **3.x** (persistence) — independent track.
5. **2.x** (focus sugar) — additive, lowest risk, last.
6. **4.x** throughout.

**Risks / gotchas:**
- prompt_toolkit handlers must be non-blocking: mutate → `invalidate()`; prints via
  `pt_run_in_terminal` only.
- The 1s ticker must be cancelled when focus is cleared or the agent finishes (don't
  leave invalidate-loops running).
- Queue drainer must not race the completion-notification cycle: notifications say
  "new output available"; they never carry text that can go stale.
- Ctrl+E overrides emacs end-of-line — verify no other default binding collision; F4
  alias is the escape hatch.
- Windows: monitor/binding split preserves the existing invariant (monitor only while
  prompt dismissed); no new contention class.
- `stats()` for the prompt bar must not read tee buffers (perf).
- Restore: live-created names win; never auto-delete registry entries on failed loads.
- `get_agent_output` on a *running* agent must first drain any unconsumed results from
  prior turns, then append the live tail — otherwise queued-turn results linger
  invisibly while the newest turn runs.
- Ctrl+C stage 1 must clear the staged-press counter on any visible action, so a user
  who interrupt-stops an agent and later wants to exit isn't surprised by a
  carry-over stage.

## 9. Test plan

Extend the existing 6-file subagent suite (naming follows its conventions). All new
tests are platform-neutral per §7.7 — monkeypatched OS detection, direct API calls:

- `test_subagent_signals.py` — per-agent signal consume-on-read; per-turn scoping
  (drainer clears); unfocused agents never observe; additive parent+focused semantics;
  loop-level interrupt stops a turn between tools.
- `test_subagent_input_queue.py` — queue while busy; user-entry coalescing vs
  model-entry-per-turn; self-notification dropped from queue; `pause`/`delete` flush +
  future resolution; unfocus-mid-queue continues in background.
- `test_subagent_result_log.py` — overlapping turn completions both retrievable;
  consume-on-fetch; error results included.
- `test_subagent_error_notification.py` — errored agent notifies within one cycle.
- `test_subagent_restore.py` — registry roundtrip (completed/interrupted/corrupt/
  name-collision); lazy hydration; **dangling tool-call stitching** (assistant
  tool_calls without results → synthetic results injected, no orphan tool messages);
  `_initialized` set post-load (no duplicate system prompt).
- `test_subagent_focus_cycle.py` — pure `next_focus(current, names)` ordering + wrap;
  `/agent <n>` index resolution.
- Update `test_subagent_notification_loop.py` static source assertions for any moved
  injection-site code; keep `test_prompt_completion_watch.py` green (the race now
  shares the loop with the ticker — the watcher must not be starved).
- Keybinding logic (`next_focus`, chord guards) tested as pure functions; actual chord
  capture (c-b/c-e/F4) is a manual smoke test per terminal, per §7.1.

## 10. Out of scope

- Nested subagents (flat by design).
- Per-subagent model override.
- Remote-focus interactions over Telegram (view-only parity).
- Streaming per-agent ticker for unfocused agents (`--watch` covers the need).
