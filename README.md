# Nythos-Plus-


# NYTHOS PLUS

### Adaptive Effort Runtime for Local AI

> **Don't make the model think harder. Make it know when it should.**

Nythos Plus is a single-file, standard-library-only Python runtime built around one idea:

**reasoning effort is a resource, not a personality trait.**

Instead of sending every request through the same amount of computation, Nythos Plus estimates how much additional effort is likely to help, what that effort will cost, and whether the expected gain is large enough to justify it.

The goal is not maximum reasoning.

The goal is **minimum sufficient effort**.

---
python nythosplus.py install --dry-run
python nythosplus.py install
python nythosplus.py status
python nythosplus.py doctor
python nythosplus.py self-test
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
