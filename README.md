<img width="1680" height="936" alt="file_000000001418821095b9d1ee79132034" src="https://github.com/user-attachments/assets/5d5f39cd-d7be-41d8-8f4d-cba1ec168b25" />


# NYTHOS PLUS
<div align="center">

![NYTHOS PLUS](assets/00_banner.png)

**Adaptive Effort Runtime for local AI systems**
*Think only as hard as the task deserves.*

`single file` · `stdlib only` · `LM Studio + MCP` · `SQLite learning` · `never touches model lifecycle`

[فارسی](README.fa.md) · [Quick start](#quick-start) · [Architecture](#architecture) · [Benchmarks](#benchmarks) · [Reproduce](#reproduce-with-real-models)

</div>

---

> ### Read this first: what the numbers in this README are
> Everything about the **code** below (self-test, CLI output, the 490-run benchmark pipeline per model, SQLite results) was produced by **actually running `nythosplus.py`**.
> The **models** in the benchmark section, however, are **synthetic profiles** served by a small mock LM Studio server (`bench/mock_lmstudio.py`). Their accuracy, verbosity and speed are *assumed parameters*, **not measurements of Gemma 4, gpt-oss, GLM 4.6 Flash, GPT Sol, GPT Astra, Fable 5 or Sonnet 5.5.**
> The charts therefore show **how the runtime behaves** when it faces models with different characteristics. They do **not** rank real models. Every benchmark image carries a "SIMULATED" watermark. To get real numbers, follow [Reproduce with real models](#reproduce-with-real-models) (it takes one command per loaded model).

---

## Table of contents

1. [What is NYTHOS PLUS?](#what-is-nythos-plus)
2. [What it is not](#what-it-is-not)
3. [The effort ladder](#the-effort-ladder-e0e4)
4. [Architecture](#architecture)
5. [Life of a request](#life-of-a-request)
6. [Decision engine](#decision-engine)
7. [Native vs. emulated effort](#native-vs-emulated-effort)
8. [Safety envelope](#safety-envelope)
9. [Quick start](#quick-start)
10. [CLI reference](#cli-reference)
11. [MCP tools](#mcp-tools)
12. [Objective checks](#objective-checks)
13. [Benchmarks](#benchmarks)
14. [Findings and honest limitations](#findings-and-honest-limitations)
15. [Reproduce with real models](#reproduce-with-real-models)
16. [Repository layout](#repository-layout)
17. [Verification status](#verification-status)

---

### Adaptive Effort Runtime for Local AI

> **Don't make the model think harder. Make it know when it should.**

Nythos Plus is a single-file, standard-library-only Python runtime built around one idea:

**reasoning effort is a resource, not a personality trait.**

Instead of sending every request through the same amount of computation, Nythos Plus estimates how much additional effort is likely to help, what that effort will cost, and whether the expected gain is large enough to justify it.

The goal is not maximum reasoning.

The goal is **minimum sufficient effort**.

---

```text 
python nythosplus.py install --dry-run
python nythosplus.py install
python nythosplus.py status
python nythosplus.py doctor
python nythosplus.py self-test
```

## Why Nythos Plus exists

Most AI runtimes treat computation as a fixed setting:

```text
easy request  ──────────────► same effort
hard request  ──────────────► same effort
```

That is simple, predictable, and sometimes spectacularly wasteful.

Nythos Plus turns effort into a bounded control problem:

```text
Request
   │
   ▼
Fast probe
   │
   ├── task family
   ├── difficulty signals
   ├── model profile
   ├── effort sensitivity
   └── current resource state
   │
   ▼
Expected utility
   │
   ├── expected quality
   ├── token cost
   ├── latency cost
   └── failure / risk cost
   │
   ▼
Effort decision
   │
   ├── E0 DIRECT
   ├── E1 FOCUSED
   ├── E2 DELIBERATE
   ├── E3 VERIFIED
   └── E4 ESCALATED
   │
   ▼
Observable outcome
   │
   ├── quality
   ├── first-pass success
   ├── latency
   ├── token usage
   ├── verification result
   └── resource pressure
   │
   ▼
Learn the response curve
   │
   └──────────────► future decisions
```

The runtime therefore behaves less like a static prompt wrapper and more like a small adaptive controller.

---
<img width="1130" height="405" alt="Screenshot 2026-10-07 100752" src="https://github.com/user-attachments/assets/b3692ff2-2f00-4a04-9a2d-432699797e6b" />

## What is actually implemented

The current `nythosplus-1.py` implementation is a single Python source file and uses only the Python standard library.

The main components are:

- **Effort Policy**: chooses the minimum sufficient effort level.
- **Model Profiles**: maintain model/task-specific response curves.
- **Native vs Emulated Effort**: native controls are used only when configured or explicitly verified; otherwise Nythos uses bounded external strategies.
- **Dynamic Escalation**: a PULSE run can move upward during a request when the observed outcome justifies it.
- **Verification**: E3/E4 can add bounded verification and repair behavior.
- **Hysteresis**: avoids unstable effort switching.
- **Cold Start Priors**: decisions can be made before enough historical data exists.
- **Bounded Exploration**: limited exploration collects useful observations without turning production into a science fair.
- **Resource Guard**: effort is capped when system pressure becomes elevated.
- **SQLite Persistence**: learned curves and run metadata survive process restarts.
- **Benchmark Runner**: compares RAW, PULSE and fixed-effort modes.
- **CLI / MCP / Installer**: operational controls are integrated into the same source file.

---
<img width="989" height="657" alt="published-reference-scores" src="https://github.com/user-attachments/assets/1a924cb7-4bac-4081-9826-7beffd476cd0" />
## What is NYTHOS PLUS?

Most "reasoning" setups pick one of two extremes: always think a lot (slow, expensive) or never think (fast, brittle). NYTHOS PLUS applies a different principle: **minimum sufficient effort**. The runtime decides *when* extra compute is worth it, using only things it can observe: the visible output, token usage, latency and the result of objective checks.

- **Five effort levels**, E0 DIRECT to E4 ESCALATED, each with a prior for quality gain, token multiplier and number of model calls.
- **Utility-based policy**: `utility = quality - lam * cost - risk`. A step up must pay for itself (`mv_min`, marginal value per cost unit).
- **Learns from outcomes** in a local SQLite database (phases `COLD` → `CALIBRATED` → `LEARNED`), with calibration bias correction and a small exploration rate.
- **Hysteresis** (`enter_thr=0.30`, `exit_thr=0.15`) stops the effort level from flapping.
- **Bounded by design**: per-task budgets that are clamped by hard code-level ceilings.
- **Honest labelling**: native vs. emulated effort is always reported as such.
- **Single file, standard library only**: `nythosplus.py`, about 3 700 lines, nothing to `pip install`.

## What it is not

Taken straight from the program's own contract:

- It **does not retrain or modify models**, and does not read hidden activations or private neural state.
- It **never stores** prompts, outputs or hidden chain-of-thought (the self-test asserts there are no such columns in the database).
- It **never loads, unloads, downloads or switches models.** It refuses to send inference to a model LM Studio does not report as *loaded*.
- Emulated effort (prompt hints, a verification pass, a repair pass) is **not identical** to changing a model's internal reasoning, and it says so.

## The effort ladder (E0–E4)

![Effort ladder](assets/02_effort_ladder.png)

| Level | Name | Prior gain recovered | Calls | What happens |
|---|---|---|---|---|
| E0 | DIRECT | 0 % | 1 | Plain request, no hints |
| E1 | FOCUSED | 40 % | 1 | "Check the stated requirements one by one" hint (or native `low`) |
| E2 | DELIBERATE | 68 % | 1 | Native `medium`, or emulated two-phase `PLAN:` / `ANSWER:` prompt |
| E3 | VERIFIED | 82 % | 2 | E2 + a strict verifier pass (`VERDICT: PASS/FAIL`) |
| E4 | ESCALATED | 88 % | 3 | E3 + repair pass(es) using objective violations as feedback |

The percentages are the `EffortPolicy.GAIN_FRAC` prior (diminishing returns). Token multipliers differ for native (`1.0 → 5.5×`) and emulated (`1.0 → 5.0×`) effort. All priors are replaced by observed data as samples accumulate.

## Architecture

![Architecture](assets/01_architecture.png)

| Component | Role |
|---|---|
| `LMStudioBridge` | Streaming client for the official local REST API (`/api/v0/models`, `/v1/chat/completions`); measures TTFT, TPS and usage |
| `SecurityGuard` | Loopback-only URLs, identifier regex, bounded I/O, `shell=False`, no `eval`/`exec` |
| `ResourceGuard` | Observes CPU / RAM / GPU / VRAM and classifies `NORMAL … CRITICAL`; never kills or tunes anything |
| `RequestFeatures` probe | Cheap prompt features (family, multistep, constraints, size) with **no model call** |
| `EffortPolicy` | Builds quality/cost/utility curves per level and picks the minimum sufficient effort |
| `EffortExecutor` / `PulseRunner` | Runs E0–E4 within a `BudgetTracker`; verification and repair passes |
| `OutcomeEvaluator` | Scores answers against **objective checks only** (self-reports are not evidence) |
| `DissentTracker` | Tracks disagreement between attempts / verifier; verification economics |
| `Store` | SQLite, versioned schema, transactions: `models`, `decisions`, `outcomes`, `attempts`, `effort_state`, `events` |
| `MCPServer` | JSON-RPC over stdio: advice and evaluation tools only (no inference) |
| `PluginBuilder` / `Installer` | Generates the LM Studio plugin and registers the MCP server with backups, ownership markers and rollback |

## Life of a request

![Request flow](assets/03_request_flow.png)

1. **Probe**: derive cheap features from the prompt.
2. **Curve**: combine priors with learned per-model/per-family statistics into quality, cost and utility per level.
3. **Decide**: take the lowest level whose utility is within `util_margin` of the best and whose marginal value clears `mv_min`; apply hysteresis, budget cap and resource cap.
4. **Execute**: native effort parameter if it is *configured or verified*, otherwise an emulated hint. E3+ adds a verifier pass.
5. **Evaluate**: run objective checks; weak evidence is not learned from (`min_evidence=0.5`).
6. **Learn**: write attempts, update calibration bias, move the policy phase forward.

## Decision engine

Real, per-prompt output of `advise` on an **unseen model** (cold start, no data). The curves are priors:

![Cold start advice](assets/08_cold_start_advice.png)

**Hysteresis** keeps the level stable around the decision boundary (illustrative signal, thresholds read from `Config`):

![Hysteresis](assets/04_hysteresis.png)

Key tunables (all in `Config`, all clamped on load):

| Setting | Default | Meaning |
|---|---|---|
| `w_tok`, `w_lat` | 1.0, 0.1 | cost units per 1k tokens / per second |
| `lam` | 0.02 | quality value of one cost unit |
| `w_risk` | 0.5 | weight of the risk term |
| `mv_min` | 0.02 | minimum marginal value to justify stepping up |
| `enter_thr` / `exit_thr` | 0.30 / 0.15 | hysteresis band for escalation |
| `stop_conf` | 0.80 | confidence at which extra passes stop |
| `cold_k`, `min_samples`, `learned_min` | 6, 8, 40 | prior strength and phase thresholds |
| `explore_eps`, `ucb_c` | 0.05, 0.12 | exploration |

## Native vs. emulated effort

Native effort (`reasoning_effort` request parameter) is used **only** when it was configured or verified by an explicit probe:

```bash
python nythosplus.py models --probe MODEL_ID          # two tiny inference calls: low vs. high
```

The probe compares reasoning tokens (or, failing that, estimated output tokens) between the lowest and highest value and records `VERIFIED`, `INEFFECTIVE` or `UNSUPPORTED`. Anything not verified runs as **EMULATED**.

![Native effort probe](assets/bench_11_native_effort_probe.png)

## Safety envelope

Budgets can be tuned, but **never above the hard ceilings**:

![Budget envelope](assets/05_budget_envelope.png)

`ResourceGuard` degrades gracefully instead of crashing the machine. At `HIGH` the effort cap drops to E2; at `CRITICAL` the runtime refuses to start new expensive work:

![Resource guard](assets/06_resource_guard.png)

Other guarantees enforced and covered by the self-test: loopback-only networking, `shell=False`, atomic file writes (temp → fsync → replace), byte-identical backups before touching `mcp.json`, rollback on failed verification, and refusal to modify any `nythosplus` entry or folder it does not own.

## Quick start

Requirements: Python 3.9+ (developed and verified here on 3.12.3), LM Studio with its local server enabled (default `http://127.0.0.1:1234`). No third-party packages.

```bash
# 0. verify the program itself (offline, isolated temp dirs)
python nythosplus.py self-test

# 1. see what LM Studio exposes and which models are loaded
python nythosplus.py models

# 2. preview, then perform the install (plugin + MCP registration, with backups)
python nythosplus.py install --dry-run
python nythosplus.py install

# 3. (optional) probe native effort control on a LOADED model
python nythosplus.py models --probe openai/gpt-oss-20b

# 4. ask for advice without calling any model
echo '{"prompt":"Compute 17*23 + 144/12.","model":"openai/gpt-oss-20b"}' | python nythosplus.py advise --stdin --json

# 5. benchmark a loaded model (explicit opt-in, calls the model)
python nythosplus.py benchmark --run --model openai/gpt-oss-20b --modes RAW,PULSE,ORACLE --repeat 3
```

Data lives in `NYTHOSPLUS_HOME` (default `%LOCALAPPDATA%\NythosPlus` on Windows, `~/.local/share/NythosPlus` elsewhere). Remove everything the tool owns with `python nythosplus.py uninstall` (your data is kept unless you add `--purge-data --yes`).

## CLI reference

Global options: `--home DIR`, `--config MCP_JSON`, `--json`, `-v`, `--mcp`, `--version`.

| Command | Purpose |
|---|---|
| `status` | Database, registration, plugin and LM Studio status (`--no-lm-studio` to skip the server) |
| `doctor` | Read-only diagnostics, including a subprocess MCP protocol probe |
| `self-test` | Offline test-suite in isolated temp directories |
| `install` / `repair` / `uninstall` | Plugin + `mcp.json` management (`--dry-run`, `--skip-plugin`, `--skip-mcp`, `--purge-data --yes`) |
| `models` | List models and load state; `--probe MODEL_ID [--values low,medium,high]` |
| `policy` | Effective policy settings; `--model` / `--family` show the learned curve |
| `benchmark` | Summarize stored results; `--run --model M --modes … --repeat 1..5 --tasks ids` runs built-in tasks |
| `history` | Recent run metadata (no prompts, no outputs) |
| `advise` | Reads a bounded JSON object from stdin, prints effort advice, **never calls a model** |

Real output of `policy` after a benchmark (synthetic profile):

```text
Effort policy (minimum sufficient effort, E0..E4)
  budget       : max_effort=E4 calls=4 tokens=8192 elapsed=300s escalations=2
  thresholds   : enter=0.3 exit=0.15 stop_conf=0.8 mv_min=0.02 min_evidence=0.5
  learning     : on (cold_k=6, min_samples=8, learned_min=40)
  gpt-oss-20b / math: kind=native phase=LEARNED samples=[15, 15, 19, 26, 15]
    quality [0.683, 0.657, 0.892, 0.955, 0.984]  cost [0.439, 0.471, 0.73, 1.27, 1.544]
```

More captured outputs: [`results/cli_examples.txt`](results/cli_examples.txt).

## MCP tools

Registered over stdio as server `nythosplus`. **No tool performs inference.**

| Tool | Read-only | Description |
|---|---|---|
| `nythosplus_status` | yes | Phase, stored outcomes, resource state |
| `nythosplus_advise` | no | Recommend E0–E4 for a prompt (≤ 8 000 chars); returns a `trace_id` |
| `nythosplus_evaluate` | no | Evaluate an answer against objective checks for that `trace_id`; the answer text is not stored |
| `nythosplus_policy` | yes | Learned response curve for a model and task family |
| `nythosplus_history` | yes | Recent outcome metadata |

## Objective checks

Checks are the only form of evidence the runtime learns from: `equals`, `contains`, `not_contains`, `number`, `regex`, `json`, `json_keys`, `json_len`, `python` (syntax), `defines`, `words`, `bullets` (and more, see `parse_check`). Task families: `reasoning`, `coding`, `math`, `planning`, `instruction`, `verification`, `consistency`, `tool_use`, `long_context`, `vision`, `general`.

## Benchmarks

### Setup

| Item | Value |
|---|---|
| Runtime under test | the unmodified `nythosplus.py` (v1.0.0), via `benchmark --run` |
| Tasks | 14 built-in tasks over 9 domains (reasoning, math, coding, planning, instruction, verification, consistency, tool use, long context), each scored by objective checks |
| Modes | `RAW` (no hints, no effort parameter), `PULSE` (adaptive), `FIXED_E0…E4`, and `ORACLE` (best fixed mode per task, in hindsight) |
| Repeats | 5 per task → **70 scored runs per mode per model**, 490 runs per model |
| Models | **8 synthetic profiles** named after the requested models (Gemma 4, gpt-oss 20B/120B, GLM 4.6 Flash, GPT Sol, GPT Astra, Fable 5, Sonnet 5.5) |
| Backend | `bench/mock_lmstudio.py`: stdlib HTTP server implementing exactly the endpoints `LMStudioBridge` calls |
| Profile parameters | per-domain base accuracy, responsiveness to effort, hidden-reasoning tokens, verifier accuracy, tokens/s, and whether a native `reasoning_effort` parameter is accepted (all **assumed**) |
| Statistics | 95 % bootstrap CI (2 000 resamples) on mean quality |
| Latency | the mock does not model real speed; "latency" figures are **tokens ÷ the profile's assumed tok/s** |

Profile settings are plain constants at the top of [`bench/mock_lmstudio.py`](bench/mock_lmstudio.py): read them before drawing any conclusion.

### 1. Scorecard

![Scorecard](assets/bench_12_scorecard.png)

### 2. Quality by execution mode

![Quality by mode](assets/bench_01_quality_by_mode.png)

| Model (synthetic) | RAW | PULSE | Fixed E3 | Fixed E4 | ORACLE |
|---|---|---|---|---|---|
| Gemma 4 | 0.687 | 0.857 | 0.829 | 1.000 | 1.000 |
| GLM 4.6 Flash | 0.715 | 0.953 | 0.802 | 1.000 | 1.000 |
| GPT Sol | 0.837 | 0.975 | 0.910 | 1.000 | 1.000 |
| gpt-oss 20B | 0.775 | 1.000 | 0.933 | 1.000 | 1.000 |
| gpt-oss 120B | 0.839 | 1.000 | 0.950 | 1.000 | 1.000 |
| GPT Astra | 0.886 | 1.000 | 0.979 | 1.000 | 1.000 |
| Sonnet 5.5 | 0.907 | 0.996 | 0.936 | 1.000 | 1.000 |
| Fable 5 | 0.946 | 0.986 | 0.979 | 1.000 | 1.000 |

Averaged over the eight profiles: RAW **0.824** → PULSE **0.971** (+0.147). The smaller the model's headroom, the smaller the gain (+0.04 for the strongest profile, +0.24 for the weakest).

### 3. Cost: tokens, calls, time

![Pareto](assets/bench_02_pareto_quality_vs_tokens.png)

| Model (synthetic) | RAW tok | PULSE tok | Fixed E4 tok | PULSE ÷ RAW | saved vs E4 | PULSE avg effort | calls/task |
|---|---|---|---|---|---|---|---|
| Gemma 4 | 24 | 106 | 107 | 4.35× | 1 % | 3.01 | 2.27 |
| GLM 4.6 Flash | 165 | 404 | 468 | 2.45× | 14 % | 2.21 | 1.70 |
| GPT Sol | 187 | 482 | 504 | 2.58× | 5 % | 2.43 | 1.94 |
| gpt-oss 20B | 261 | 631 | 966 | 2.42× | 35 % | 1.80 | 1.54 |
| gpt-oss 120B | 348 | 554 | 1168 | 1.59× | 53 % | 1.34 | 1.17 |
| GPT Astra | 309 | 539 | 1067 | 1.75× | 49 % | 1.64 | 1.27 |
| Sonnet 5.5 | 251 | 416 | 861 | 1.66× | 52 % | 1.56 | 1.07 |
| Fable 5 | 406 | 708 | 1430 | 1.75× | 50 % | 1.20 | 1.17 |

![Tokens per task](assets/bench_04_tokens_per_task.png)
![Token savings vs fixed](assets/bench_07_token_savings_vs_fixed.png)
![Estimated latency](assets/bench_10_estimated_latency.png)

### 4. How effort scales when it is fixed

![Fixed effort curves](assets/bench_03_fixed_effort_curves.png)

### 5. Where PULSE spends its effort

![PULSE effort distribution](assets/bench_06_pulse_effort_distribution.png)
![Calls, repair, escalation](assets/bench_09_calls_repair_escalation.png)

### 6. Where the gain comes from

![First attempt vs. final](assets/bench_08_first_attempt_vs_final.png)

### 7. Domain breakdown

![Domain heatmap](assets/bench_05_domain_heatmap.png)

Raw numbers: [`results/summary.csv`](results/summary.csv), [`results/summary.json`](results/summary.json), one SQLite DB and one JSON per profile in [`results/`](results/).

## Findings and honest limitations

What the runs **do** show about the runtime:

1. **The pipeline works end-to-end**: probe → decide → execute → verify/repair → evaluate → learn ran 3 920 times (8 × 490) without a crash, and `models --probe` classified the synthetic native-effort profiles `VERIFIED` and the others `UNSUPPORTED`.
2. **PULSE lifts quality over RAW on every profile** (+0.04 to +0.24) and approaches or reaches the fixed-E4 ceiling.
3. **Where there is a lot of expensive native reasoning, PULSE saves roughly half the tokens of "always E4"** (gpt-oss 120B 53 %, Sonnet 5.5 52 %, Fable 5 50 %, GPT Astra 49 %).

What deserves caution:

1. **PULSE is not cheaper than RAW.** It cost 1.6× to 4.4× the tokens of a single raw call, because quality is bought with verification and repair.
2. **PULSE does not always beat the fixed levels on cost.** For profiles that use *emulated* effort and little hidden reasoning (Gemma 4, GPT Sol, GLM 4.6 Flash), the plan hint and extra calls dominate; savings versus E4 were only 1–14 %.
3. **Fixed E4 reached 1.000 everywhere.** That is a property of this harness: the built-in tasks provide objective checks, so E4's repair loop receives exact violations. Real workloads without ground-truth checks will not behave like that. PULSE depends on the same checks for learning and repair.
4. **Cold-start advice is conservative.** On a fresh model `advise` returned E2 for *every* prompt, including "What is the capital of France?" (see the cold-start figure). The "minimum sufficient effort" behaviour only becomes prompt-specific as data accumulates (`COLD → CALIBRATED → LEARNED`). Expect to run a benchmark or use the tool for a while before relying on E0 for easy prompts. This is an observation, not a claimed defect; the cause is the shipped priors.
5. **The ORACLE row is hindsight**, an upper bound, not an achievable policy.
6. **Small samples**: 70 runs per bar. Several confidence intervals overlap.
7. **Resource guard on a 1-vCPU host.** The CPU signal is `loadavg / cpu_count`. On the 1-vCPU sandbox used here the first benchmark attempt was refused with `resource_critical`, which is the guard working as designed. The harness therefore launches the program through [`bench/nythos_wrap.py`](bench/nythos_wrap.py), which only pins the resource sampler to a fixed NORMAL reading; **`nythosplus.py` itself is unmodified** (checked byte-for-byte against the uploaded file). On your own machine you do not need the wrapper.
8. **Not verified live**: the self-test reports `SKIP lm_studio_live` ("NOT VERIFIED LIVE"). Launching the generated plugin / MCP server *from inside LM Studio* was not exercised here.

## Reproduce with real models

Run the real thing against models you have loaded in LM Studio (the tool never loads them for you):

```bash
# once per model: load it in LM Studio first, then
python nythosplus.py models --probe MODEL_ID
python nythosplus.py benchmark --run --model MODEL_ID --modes RAW,PULSE,ORACLE --repeat 5 --json > results/MODEL_ID.json
```

To regenerate every figure from real results, use one `NYTHOSPLUS_HOME` per model, copy each `nythosplus.db` to `results/<name>.db` (and the probe output to `results/<name>.json` with a `"probe"` key), edit the `MODELS` list and labels at the top of `bench/make_charts.py`, and remove the `SIMULATED` watermark (`wm=False`) in `save(...)`. Then:

```bash
python bench/make_charts.py
```

To re-run the *synthetic* study instead:

```bash
python bench/run_all.py        # starts 8 mock servers, runs probe + benchmark for each (≈ 70 s)
python bench/make_charts.py    # rebuilds assets/*.png and results/summary.*
```

## Repository layout

```text
nythosplus.py            the whole runtime (single file, stdlib only)
assets/                  21 figures used in this README
bench/
  mock_lmstudio.py       mock LM Studio server + SYNTHETIC model profiles
  run_all.py             runs probe + benchmark per profile, in parallel
  nythos_wrap.py         benchmark-host launcher (pins resource sampler only)
  make_charts.py         benchmark figures
  make_diagrams.py       architecture and concept figures
results/
  summary.csv|json       aggregated numbers behind every chart
  <profile>.json|.db     raw run results per synthetic profile
  self_test.txt          real self-test output
  cli_examples.txt       real CLI outputs (advise, policy, history, help)
  advise_cold.json       cold-start decisions
```

## Verification status

![Self-test](assets/07_self_test.png)

| Check | Result |
|---|---|
| `python nythosplus.py self-test` | 13 PASS, 1 SKIP (`lm_studio_live`), 0 FAIL |
| Full benchmark pipeline against the mock backend | 8 profiles × 490 runs completed, exit code 0 |
| `nythosplus.py` modified by the benchmark harness | No (byte-identical to the uploaded file) |
| Real models (Gemma 4, gpt-oss, GLM 4.6 Flash, GPT Sol/Astra, Fable 5, Sonnet 5.5) | **Not run.** Synthetic profiles only |
| Live LM Studio plugin / MCP launch | **Not verified** |

---

<div align="center">

*NYTHOS PLUS: spend compute where it changes the answer.*

</div>

## The five effort levels

| Level | Meaning | Typical behavior |
|---|---|---|
| **E0** | DIRECT | Fastest path, no additional effort |
| **E1** | FOCUSED | Small increase in deliberate computation |
| **E2** | DELIBERATE | More compute when the task appears sensitive to effort |
| **E3** | VERIFIED | Generation plus verification |
| **E4** | ESCALATED | Verification plus bounded repair / further escalation |

The important engineering distinction is:

> **Difficulty is not the same thing as effort sensitivity.**

A difficult-looking request can be stable at low effort. A short request can have a large quality gain from additional computation. The runtime is designed to learn that difference.

---

## Effort sensitivity

Nythos Plus tries to estimate:

**How much does this model benefit from additional computation for this kind of task?**

That becomes a model/task response curve:

```text
quality
  ^
  |                         ______ E4
  |                   _____/
  |              ____/       E3
  |         ____/
  |    ____/                 E2
  |___/____________________________> effort
      E0   E1   E2   E3   E4
```

The curve is intentionally not treated as linear.

More compute often has diminishing returns. The runtime therefore compares the expected improvement against the incremental cost.

A conceptual marginal-value measure is:

```text
Marginal Value of Compute
≈
expected quality gain
----------------------
incremental compute cost
```
<img width="1287" height="513" alt="nythosplus-effort-ladder-1" src="https://github.com/user-attachments/assets/d913692c-6098-4db9-9f8f-b1392a90f225" />

The decision layer then uses a utility model instead of blindly selecting the highest effort.

Conceptually:

```text
utility =
    expected quality
  - latency cost
  - token cost
  - risk cost
```

This is one of the central ideas behind Nythos Plus.

---

## Native effort vs emulated effort

Nythos Plus makes an explicit distinction between:

### Native effort

The model/runtime exposes a real effort control and Nythos Plus has a configured or verified mapping.

Example:

```text
E0 → low
E1 → low
E2 → medium
E3 → high
E4 → high / model-specific
```

The exact mapping is model dependent.

### Emulated effort

The model does not expose a verified native control, so Nythos Plus uses bounded external mechanisms such as:

- prompt-level effort hints
- a verification pass
- a repair pass
- a bounded escalation sequence

**Emulated effort is not claimed to be identical to changing internal model reasoning.**

That distinction matters. Renaming a prompt parameter does not magically create a new neural control surface. Humanity has already done enough marketing archaeology.

---

## Dynamic escalation

PULSE is the adaptive mode.

A simplified runtime loop is:

```text
Choose initial effort
        │
        ▼
Generate
        │
        ▼
Evaluate observable outcome
        │
        ├── good enough ─────► STOP
        │
        └── not good enough
                 │
                 ▼
        Is more effort worth it?
                 │
          ┌──────┴──────┐
          │             │
         NO            YES
          │             │
          ▼             ▼
         STOP        E(n+1)
                         │
                         └──► generate again
```

The escalation controller also considers:

- current budget
- resource pressure
- prior effort
- uncertainty
- verification results
- marginal value
- learned model/task statistics

The policy includes hysteresis so the runtime does not bounce between effort states because one signal moved slightly.

---

## Learning

Nythos Plus stores **metadata about outcomes**, not hidden chain-of-thought.
<img width="2016" height="1017" alt="nythosplus-architecture" src="https://github.com/user-attachments/assets/8ac34a78-0203-47ef-818c-2877515646c1" />

Examples of stored signals include:

- task family
- selected effort
- final effort
- quality score
- success status
- latency
- token usage
- escalation count
- resource state
- model identifier
- benchmark mode

The code explicitly rejects storing hidden/private chain-of-thought fields.

This makes the learning loop much more auditable:

```text
observable outcome
      │
      ▼
quality estimate
      │
      ▼
model × task × effort curve
      │
      ▼
future policy
```

A learned policy only becomes more confident as evidence accumulates.

---

## Cold start, exploration and UCB-style learning

At cold start there is not enough history to know the true response curve.

Nythos Plus therefore begins with bounded priors and gradually replaces them with observations.

After sufficient observations, a contextual-bandit-style selection path can be used:

```text
context = model × task family × task kind

reward ≈ quality - λ × cost

selection ≈ reward + uncertainty bonus
```

The uncertainty bonus is intentionally bounded.

The objective is not to gamble compute forever. It is to learn where compute is useful.

---

## Safety and scope

Nythos Plus is designed around explicit boundaries.
<img width="1008" height="609" alt="nythosplus-benchmark-protocol" src="https://github.com/user-attachments/assets/eb455ed9-eb14-4785-96ca-685a2a04ae83" />

### It does not

- load models
- unload models
- download models
- switch models
- modify model weights
- inspect hidden activations
- access private neural states
- store private hidden chain-of-thought
- execute model-generated shell commands
- use `shell=True`
- require a remote cloud service
- silently route requests to arbitrary machines

### LM Studio lifecycle remains user-owned

The runtime checks whether a model is already reported as loaded and refuses to use a model that is not confirmed as loaded.

The principle is:

> **The user chooses. LM Studio runs the models. Nythos Plus controls bounded effort.**

---

## LM Studio integration

The current implementation uses a local HTTP bridge and a generated LM Studio plugin project.

The Python file is the **source of truth**.

During installation, the source can generate the runtime plugin project used by LM Studio. That project contains files such as:

```text
manifest.json
package.json
tsconfig.json
src/
  advise.ts
  config.ts
  toolsProvider.ts
```

Therefore:

> **One Python source file** does not mean **zero generated runtime artifacts**.

Those generated files are installation artifacts, not additional Python source modules.

The current repository code also explicitly labels live LM Studio plugin compatibility as **NOT VERIFIED LIVE**. The installer can generate, hash and validate its artifacts, but an actual live launch inside LM Studio still requires a real environment test.

---

## CLI

### Installation

```bash
python nythosplus.py install --dry-run
python nythosplus.py install
```

### Health

```bash
python nythosplus.py status
python nythosplus.py doctor
python nythosplus.py self-test
```

### Models

```bash
python nythosplus.py models
```

### Policy

```bash
python nythosplus.py policy
python nythosplus.py policy --model YOUR_LOADED_MODEL
```

### Benchmark

A read-only summary:

```bash
python nythosplus.py benchmark
```

A real benchmark run requires an already loaded LM Studio model:

```bash
python nythosplus.py benchmark \
  --run \
  --model YOUR_LOADED_MODEL \
  --modes RAW,PULSE,ORACLE \
  --repeat 3
```

The current CLI allows `repeat` values from 1 to 5.

### History

```bash
python nythosplus.py history --limit 50
```

### MCP

```bash
python nythosplus.py --mcp
```

### Repair

```bash
python nythosplus.py repair
```

### Uninstall

```bash
python nythosplus.py uninstall
```

Destructive data purge is deliberately guarded by an explicit confirmation flag.

---

## Benchmark design

Nythos Plus has four particularly useful benchmark modes.

### RAW

Baseline path.

```text
request → model → answer
```

No adaptive effort selection.

### PULSE

Adaptive path.

```text
request
   ↓
policy chooses effort
   ↓
model execution
   ↓
evaluate
   ↓
possibly escalate
```

### FIXED_E0 … FIXED_E4

Controlled references.

Each request is run at a fixed effort so the measured response curve can be compared with PULSE.

### ORACLE

ORACLE expands into fixed-effort references so the experiment can estimate which effort level actually worked best for the task.

This is useful because an adaptive controller needs something to learn against.

---

## What the benchmark measures

The current benchmark output includes:

- mean quality
- first-pass success
- average final effort
- output tokens
- quality per 1k tokens
- latency / timing information
- escalation count
- resource state

The important result is **not**:

> “Which model has the biggest raw score?”

The interesting question is:

> **How much useful quality can the system obtain per unit of additional computation?**

That is the problem Nythos Plus is actually trying to solve.

---

## Built-in benchmark task families

The current built-in suite includes examples for:

- reasoning / math
- coding
- instruction following
- verification
- structured JSON output
- tool-call formatting
- long-context retrieval
- consistency

A representative benchmark record looks conceptually like:

```text
model
task family
run mode
initial effort
final effort
quality
success
output tokens
latency
escalations
resource state
```

---

# Model evaluation matrix

The following model families are useful for evaluating different aspects of the design.

| Model | Local / LM Studio | Effort control | Role in Nythos Plus evaluation |
|---|---|---|---|
| **Gemma 4 E4B** | Yes | Verify per runtime | Small local efficiency target |
| **Gemma 4 12B** | Yes | Verify per runtime | Mid-size local quality/efficiency target |
| **gpt-oss-20b** | Yes | Native low/medium/high | Strong native-effort baseline |
| **gpt-oss-120b** | Yes, high-end hardware | Native low/medium/high | High-capacity local reference |
| **GLM-4.6V-Flash** | Yes | Verify per runtime | Lightweight multimodal/local target |
| **GPT-6 Sol** | Cloud | Native effort | External adaptive-effort reference |
| **GPT-6 Astra** | Cloud | Native effort | Frontier external reference |
| **Claude Fable 5** | Cloud | Native/adaptive, model-specific | Long-horizon reasoning reference |
| **Claude Sonnet 5.5** | Cloud | Adaptive | Fast frontier work reference |

### Important comparability rule

The first five rows can be tested through the local LM Studio pathway.

The final four are **not currently equivalent Nythos Plus LM Studio targets**.

Nythos Plus 1.0 does not contain native OpenAI or Anthropic cloud adapters. Therefore, a claim such as:

> “Nythos Plus beat Astra”

would be methodologically invalid without adding a separate adapter and controlling for API behavior, tools, system prompts, rate limits, effort settings, context windows and cost.

A benchmark article should be honest before it is impressive.

---

# Published model reference points

The repository can include published reference numbers, but they must remain clearly separated from Nythos-run measurements.

## Gemma 4

Google reports, among other figures:

- **Gemma 4 12B**: 77.2% MMLU Pro
- **Gemma 4 12B**: 78.8% GPQA Diamond
- **Gemma 4 12B**: 77.5% AIME 2026 without tools
- **Gemma 4 12B**: 72.0% LiveCodeBench v6
- **Gemma 4 E4B**: 69.4% MMLU Pro
- **Gemma 4 E4B**: 58.6% GPQA Diamond

These are Google-published evaluations, not measurements produced by this repository.

## gpt-oss

OpenAI reports:

| Benchmark | gpt-oss-20b | gpt-oss-120b |
|---|---:|---:|
| MMLU | 85.3 | 90.0 |
| GPQA Diamond | 71.5 | 80.1 |
| AIME 2024 | 96.0 | 96.6 |
| AIME 2025 | 98.7 | 97.9 |

OpenAI also documents native reasoning-effort controls for the two open-weight models.

## GLM-4.6V-Flash

Z.ai describes GLM-4.6V-Flash as a 9B model optimized for local deployment and low-latency applications, with a 128K context and native multimodal function calling.

Because Nythos Plus currently uses a text-oriented chat request structure, a rigorous **vision benchmark should be treated as a future extension**, not silently presented as already supported by the current benchmark runner.

## GPT-6 Sol and GPT-6 Astra

OpenAI's current reasoning stack exposes model-dependent effort controls such as low, medium, high, xhigh and max. GPT-6 Sol is positioned as a lower-cost complex-work model, while GPT-6 Astra is positioned as OpenAI's most capable model for demanding work.

These are valuable external references for the theory of adaptive effort, but they are not local LM Studio measurements in this repository.

## Claude Fable 5 and Sonnet 5.5

Anthropic describes Fable 5.1 as a high-end model for long-horizon agentic coding, knowledge work and research, with adaptive thinking and configurable effort.

Anthropic describes Sonnet 5.5 as a faster model for well-scoped work, coding, documents, spreadsheets and everyday tasks, with adaptive effort.

Again, those public benchmark scores should be treated as **external references**, not Nythos measurements.

---

# A better benchmark question

Instead of building another leaderboard that turns into a bar chart cemetery, Nythos Plus should answer four questions:

### 1. Quality gain

How much does additional effort improve the task outcome?

### 2. Marginal value

How much quality do we gain per additional unit of compute?

### 3. Allocation accuracy

How often does PULSE choose an effort close to the task's empirically useful level?

### 4. Resource efficiency

Does PULSE preserve quality while reducing average tokens, latency or unnecessary escalations?

A successful result may look like:

```text
RAW
quality          ██████████
tokens           ████████████████

PULSE
quality          ██████████████
tokens           ███████████

FIXED_E4
quality          ███████████████
tokens           █████████████████████████
```

The goal is not to maximize the final bar.

The goal is to occupy a better point on the quality/cost frontier.

---

# Recommended benchmark matrix

For a serious evaluation, run the same task set across each supported local model.

```text
                 RAW    PULSE    E0    E1    E2    E3    E4
----------------------------------------------------------------
Reasoning         ✓       ✓      ✓     ✓     ✓     ✓     ✓
Coding            ✓       ✓      ✓     ✓     ✓     ✓     ✓
Instruction       ✓       ✓      ✓     ✓     ✓     ✓     ✓
Verification      ✓       ✓      ✓     ✓     ✓     ✓     ✓
Structured JSON   ✓       ✓      ✓     ✓     ✓     ✓     ✓
Tool formatting   ✓       ✓      ✓     ✓     ✓     ✓     ✓
Long context      ✓       ✓      ✓     ✓     ✓     ✓     ✓
```

For each model, preserve:

- exact model revision
- quantization
- context length
- system prompt
- temperature
- seed, when supported
- hardware
- driver
- LM Studio version
- runtime version
- prompt set
- number of repetitions

Otherwise the benchmark becomes a ritual rather than an experiment.

---

# Benchmark image set

The repository assets include:

![Nythos Plus header](nythosplus-header.png)

![Nythos Plus architecture](nythosplus-architecture.png)

![Effort ladder](nythosplus-effort-ladder.png)

![Benchmark protocol](nythosplus-benchmark-protocol.png)

![Published reference scores](published-reference-scores.png)

The last graphic is deliberately labeled as a **published reference** rather than a Nythos Plus result. It uses model-family results published by the model vendors and should not be read as one unified leaderboard.

---

# Benchmark interpretation

A strong Nythos Plus result would not necessarily mean:

```text
PULSE > E4
```

That would be expected only if PULSE always spent the maximum budget.

A more meaningful result is:

```text
PULSE ≈ best useful quality
while
PULSE < fixed-high effort cost
```

For example, a useful research target is:

```text
quality(PULSE) ≥ quality(best fixed effort) - ε

and

tokens(PULSE) < tokens(best fixed effort)

and/or

latency(PULSE) < latency(best fixed effort)
```

where `ε` is a small, predeclared tolerance.

This turns Nythos Plus from a vague “AI makes decisions about thinking” concept into an experimentally testable controller.

---

# Verification status

The current `nythosplus-1.py` source was independently executed in a clean runtime test for this repository review.

Current result:

```text
13 automated tests passed
1 live LM Studio integration check skipped
```

Passed areas include:

- CLI dispatch
- help
- database initialization
- install dry-run
- plugin generation
- install backup
- ownership protection
- rollback
- uninstall preservation
- repair
- bounded `advise --stdin --json`
- MCP startup / stdout protocol purity
- security checks

The skipped item is intentional:

```text
NOT VERIFIED LIVE:
LM Studio launching Nythos Plus is not covered by the offline self-test.
```

That distinction should remain visible in the repository until a real LM Studio environment test has been performed.

---

# What makes the project interesting

Nythos Plus is not trying to invent a new foundation model.

It is trying to solve a different systems problem:

```text
given a model
given a task
given a budget
given a resource state

when is more computation actually worth paying for?
```

That makes the project complementary to model improvements rather than competitive with them.

A smarter model is useful.

A smarter **compute allocator** can be useful too.

The long-term ambition is a runtime where the answer to a request is not:

> “Always reason at maximum.”

and not:

> “Always be fast.”

but:

> **“Spend the smallest amount of computation that reliably solves this task.”**

---

# Project philosophy

Nythos Plus follows several hard principles:

```text
USER CHOOSES
      ↓
LM STUDIO RUNS
      ↓
MODEL REASONS
      ↓
NYTHOS PLUS ADAPTS EFFORT
      ↓
OBSERVABLE OUTCOME
      ↓
POLICY LEARNS
```

No hidden access to neural activations.

No claim of internal J-space access.

No fake benchmark numbers.

No pretending a prompt hint is a native neural control.

No silent model lifecycle management.

No cloud dependency for the core runtime.

---

# Roadmap

The next meaningful milestones are not more decorative abstractions.

They are experimental.

### Phase 1

Complete the real LM Studio live integration test.

### Phase 2

Run the local benchmark matrix against:

- Gemma 4 E4B
- Gemma 4 12B
- gpt-oss-20b
- gpt-oss-120b
- GLM-4.6V-Flash

### Phase 3

Measure:

- quality delta
- token delta
- latency delta
- escalation rate
- effort-selection regret
- resource-aware performance

### Phase 4

Add opt-in external adapters for cloud reference models if strict benchmark parity can be maintained.

---

# Research direction

The design is inspired by a broader line of work on:

- adaptive computation
- selective verification
- early stopping
- compute-aware inference
- contextual decision policies
- reasoning-effort controls
- quality/cost Pareto optimization

Nythos Plus deliberately treats these as a systems problem:

```text
intelligence is only useful
when computation is allocated intelligently.
```

---

# License and authorship

**Nythos Plus** is part of the Nythos / Angra ecosystem.

The repository's actual software license should remain the authoritative legal source.

---

# References

### Nythos Plus source

- `nythosplus-1.py` — project implementation reviewed for this document.

### Model documentation

- Google DeepMind / Google AI Developers — Gemma 4 model family and model card.
- OpenAI — gpt-oss open-weight models and model documentation.
- Z.ai — GLM-4.6V-Flash documentation and model card.
- OpenAI — GPT-6 Sol and GPT-6 Astra documentation.
- Anthropic — Claude Fable 5.1 and Claude Sonnet 5.5 documentation.

### Methodological note

Vendor-reported model benchmarks are included only as external references. Nythos Plus benchmark results should always be generated from a controlled local run and stored with environment metadata.

---

## Final principle

> **More thinking is not always better.**
>
> **Knowing when more thinking is worth it is a systems problem.**

Nythos Plus is an attempt to build that system.
