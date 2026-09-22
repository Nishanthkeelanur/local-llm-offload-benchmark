# Local LLM: offload benchmark and quality evaluation

Running large language models on a single consumer GPU by splitting them across
VRAM, system RAM and NVMe storage — measuring what each tier costs in speed, and
whether the output is actually correct.

**Status:** work in progress. Session 1 (2026-09-22) complete for the dense-model
tier. MoE and disk-streaming tiers pending.

---

## 1. Hardware and software

| Component | Detail |
|---|---|
| CPU | Intel i9-14900K (8 P-cores, 16 E-cores, 32 threads) |
| RAM | 64 GB DDR5 — measured effective bandwidth ~53 GB/s (see §4.1) |
| GPU | NVIDIA RTX 4060, 8187 MiB VRAM, compute capability 8.9, ~272 GB/s |
| Storage (primary) | Samsung MZVL22T0HBLB-00BH1, 2 TB, PCIe 4.0 NVMe (~6–7 GB/s seq. read) |
| Storage (secondary) | HGST HUS722T2TALA604, 2 TB, SATA HDD (~0.2 GB/s — possible "worst case" tier) |
| OS | Windows 10 Pro |
| Inference engine | llama.cpp, build `bfd73a876` (11102), Windows CUDA 12.4 x64 build |
| Backends loaded | `ggml-cuda.dll`, `ggml-rpc.dll`, `ggml-cpu-alderlake.dll` |

### Models

| Model | File | Size | Params | Notes |
|---|---|---|---|---|
| Qwen3 8B | `Qwen3-8B-Q4_K_M.gguf` | 4.68 GiB | 8.19 B | Dense, 36 layers. Fits fully in VRAM. |
| Qwen3 30B-A3B | `Qwen3-30B-A3B-Q4_K_M.gguf` | ~18.6 GB | 30 B total / ~3 B active | MoE, 48 layers. |
| gpt-oss-120b | *not yet downloaded* | ~65 GB | 120 B total / ~5 B active | Planned: disk-streaming tier. |

---

## 2. Setup steps (reproducible)

```powershell
# 1. Identify storage
Get-PhysicalDisk | Select FriendlyName, MediaType, BusType, Size

# 2. Hugging Face CLI
pip install -U huggingface_hub
# Scripts dir was not on PATH — added permanently:
[Environment]::SetEnvironmentVariable("Path",
  [Environment]::GetEnvironmentVariable("Path","User") +
  ";C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\Scripts", "User")

# 3. Models
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir C:\llm\models
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir C:\llm\models

# 4. llama.cpp — needs BOTH the main build and the matching cudart zip
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow otherwise
$rels = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
$r = $rels | Where-Object { $_.assets.name -match 'bin-win-cuda.*x64\.zip' } | Select-Object -First 1
# download + Expand-Archive both zips into C:\llm\llama
```

### Setup gotchas encountered

1. **`hf` not found after install** — the `Scripts` directory wasn't on `PATH`.
   The pip output warns about this in the middle of a wall of text.
2. **GitHub "latest release" was wrong** — `/releases/latest` returned tag
   `v0.4.1` with a single asset, not one of the usual `bXXXX` builds. Had to
   enumerate releases and pick the newest one containing a `bin-win-cuda` asset.
3. **Asset filter matched the wrong file** — a naive regex (`win-cuda-12.*x64\.zip`)
   matched the `cudart-` zip as well, so the main build was never downloaded. The
   filter must anchor on `^llama-`.
4. **`Invoke-WebRequest` progress bar** repaints constantly on Windows PowerShell,
   badly corrupting terminal output and slowing the download. `$ProgressPreference
   = 'SilentlyContinue'` fixes both.

---

## 3. The hypothesis

Token generation is **memory-bandwidth bound**: every generated token requires
reading every weight in the model once. So generation speed should be roughly:

```
tokens/sec ≈ memory bandwidth / bytes read per token
```

With weights split across tiers, the per-token time should be the sum of the
per-layer times in each tier:

```
t_token = (layers_on_gpu × t_gpu_layer) + (layers_on_cpu × t_cpu_layer)
```

Prompt processing, by contrast, is **compute bound** — it processes many tokens
in parallel, so it should scale with FLOPS rather than bandwidth.

**Predictions before measuring:**
- Full GPU ceiling: 272 GB/s ÷ 4.68 GB ≈ **58 tok/s**
- Offloading should degrade speed non-linearly and steeply
- More CPU threads should stop helping for generation but keep helping for prompts

---

## 4. Results — Qwen3 8B (dense)

### 4.1 Offload cliff

```powershell
.\llama-bench.exe -m C:\llm\models\Qwen3-8B-Q4_K_M.gguf -ngl 99,27,18,9,0 --progress
```

| GPU layers | % on GPU | pp512 (tok/s) | tg128 (tok/s) | Predicted tg |
|---:|---:|---:|---:|---:|
| 36 (all) | 100% | 2390.10 ± 30.92 | **47.71 ± 2.00** | 47.7 (anchor) |
| 27 | 75% | 1551.66 ± 22.41 | **25.55 ± 0.88** | 26.4 |
| 18 | 50% | 1157.47 ± 20.39 | **16.11 ± 1.12** | 18.2 |
| 9 | 25% | 987.10 ± 12.83 | **11.95 ± 0.90** | 14.0 |
| 0 | 0% | 789.28 ± 24.25 | **11.30 ± 0.20** | 11.3 (anchor) |

**Derived per-layer costs** (from the two anchor rows):
- GPU layer: 1/47.71 ÷ 36 ≈ **0.58 ms/token**
- CPU+RAM layer: 1/11.30 ÷ 36 ≈ **2.46 ms/token** (~4.2× slower)

**Findings:**
- Offloading **25% of layers costs 46% of throughput**. The relationship is
  strongly non-linear from the user's point of view: the slow tier dominates.
- The simple additive model predicts measured values within ~2–15%, over-predicting
  at the low-GPU end (the real CPU path is slightly worse than linear).
- Measured 47.71 tok/s vs the 58 tok/s bandwidth ceiling = **82% of theoretical**,
  which is a healthy figure for a real implementation.
- Back-solving the CPU-only row: 11.30 × 4.68 GB ≈ **53 GB/s effective RAM
  bandwidth**, consistent with real-world DDR5 (vs a theoretical ~80–90 GB/s).
- **`pp512` never collapses** the way `tg128` does (789 tok/s even at `-ngl 0`),
  because the CUDA build still offloads prompt-processing matmuls to the GPU even
  when no layers are resident. The `-ngl 0` row is therefore *not* a pure CPU test —
  see §4.2 for that.

### 4.2 CPU thread scaling (true GPU-free)

```powershell
.\llama-bench.exe -m C:\llm\models\Qwen3-8B-Q4_K_M.gguf -dev none -t 4,8,16,24,32 --progress
```

| Threads | pp512 (tok/s) | tg128 (tok/s) |
|---:|---:|---:|
| 4 | 45.21 ± 0.51 | 8.92 ± 0.04 |
| 8 | 76.98 ± 4.02 | 10.75 ± 0.06 |
| 16 | 88.44 ± 1.61 | **11.77 ± 0.05** |
| 24 | 101.03 ± 0.11 | 11.38 ± 0.39 |
| 32 | **117.33 ± 2.84** | 11.43 ± 0.13 |

**Findings:**
- **Generation plateaus at ~16 threads** and does not improve with more. 4 threads
  already achieve 76% of peak. This is the memory-bound prediction confirmed.
- **Prompt processing scales all the way to 32 threads** (2.6× from 4 → 32) and had
  not flattened. Compute-bound, as predicted.
- Prediction miss: I expected 24/32 threads to be measurably *worse* than 8–16 due
  to E-core stragglers in the P-core/E-core split. Measured degradation was within
  noise. Worth retesting with `--cpu-mask` pinned to P-cores only.

**GPU vs CPU advantage, same model:**

| Workload | CPU best | GPU (full offload) | Ratio |
|---|---:|---:|---:|
| Prompt processing | 117 tok/s | 2390 tok/s | **~20×** (compute gap) |
| Token generation | 11.8 tok/s | 47.7 tok/s | **~4×** (bandwidth gap: 272/53 ≈ 5×) |

The two ratios differ by 5× and each tracks a different hardware spec. This is
the clearest single illustration of the compute-bound / memory-bound split.

> ⚠️ **Caveat:** an 18 GB model download was running during the thread sweep, using
> CPU for chunk reconstruction. The ± values are tight, but this sweep must be
> re-run on an idle machine before publication.

### 4.3 MoE — Qwen3 30B-A3B

Served interactively at:

```powershell
.\llama-server.exe -m C:\llm\models\Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -ncmoe 40 -t 16 -c 8192 --port 8080
```

**Observed interactively: ~29 tok/s.**

This is the headline anomaly of the project: a **30B model running faster than the
8B model does at 75% offload** (25.55 tok/s). The reason is that MoE routing
activates only ~3B of 30B parameters per token, so bytes-read-per-token — not file
size — governs speed. `-ncmoe N` keeps attention/shared tensors on the GPU and
places the expert FFN tensors for N layers in system RAM.

**Still to run:**

```powershell
# naive whole-layer split
.\llama-bench.exe -m C:\llm\models\Qwen3-30B-A3B-Q4_K_M.gguf -ngl 16 -t 16 --progress
# smart MoE-aware offload sweep
.\llama-bench.exe -m C:\llm\models\Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -ncmoe 48,44,40,36 -t 16 --progress
```

**Server log observations worth following up:**
- `tensor overrides to CPU are used with mmap enabled - consider using --load-mode
  none for better performance` — worth benchmarking `mmap` vs `--load-mode none`
  when tensors are CPU-pinned.
- Model load time was ~2.7 s for an 18 GB file, i.e. the OS page cache still held it
  from a previous run. Cold-cache load times must be measured separately (reboot or
  cache flush) for the disk-streaming tier to mean anything.
- `security: no API key is set and CORS allows all origins` — bound to 127.0.0.1
  only, but any local web page can reach it. Use `--api-key` for anything long-lived.

---

## 5. Quality evaluation — the postcode experiment

Speed is only half the question. Offloading does not change output quality (same
weights, same tokens), so quality is only comparable *across models*. This section
tests Qwen3 30B-A3B on a single, objectively checkable task.

**Task:** "Write a Python function that validates a UK postcode, with tests."

**Why this task:** UK postcode rules are precise, publicly specified, and full of
non-obvious constraints — an excellent test of whether a model *knows* a spec or
merely produces something that *looks* like a postcode regex. Crucially, the output
is machine-verifiable: run the asserts.

### Attempt 1 — unprompted first answer

Produced a clean, well-documented function with 13 asserts:

```python
pattern = r'^[A-CEGHIKLMNOPQRSTUVWX]{1,2}[0-9][0-9A-Z]?\s[0-9][A-Z]{2}$'
```

Failures:
- `assert validate_postcode("AB12 3CD") == False` — **fails; its own test suite does
  not pass.**
- Character class omits D, F, Y, Z → rejects real areas (`DN`, `FY`, `YO`).
- Includes Q, V, X → accepts letters that never start a postcode.
- Mandatory `\s` → rejects unspaced input (`M11AA`).

### Attempt 2 — after being shown the AssertionError

Diagnosed the failure as "two-digit districts are not allowed" (**wrong**: `AB12`
is a valid Aberdeen district; the real reason `AB12 3CD` is invalid is that the
inward code's two letters can never include C).

```python
pattern = r'^[A-CEGHIKLMNOPQRSTUVWX]{1,2}[0-9]\s[0-9][A-Z]{2}$'
```

This "fix" broke five previously-passing tests (`EC1A 1BB`, `W1A 1AA`, `SW1A 1AA`,
`B33 8TH`, `M11 1AA`). It also kept the comment `# Valid (two digits in district)`
directly above a regex forbidding exactly that, and appended a note suggesting *the
test case* be revised.

### Attempt 3 — after the next AssertionError

```python
pattern = r'^[A-CEGHIKLMNOPQRSTUVWX]{1,2}[0-9][A-Z]?\s[0-9][A-Z]{2}$'
```

Fixed `EC1A`, broke `B33`/`M11` (second *digit* now disallowed). Rejected
`AB12 3CD` for the wrong reason. Stated: *"This regex now: Passes all your original
test cases."* — **false, and unverifiable by the model, which cannot execute code.**

### Attempt 4 — after being told it still fails

Reverted to the attempt-2 regex and asserted that the **test suite was internally
inconsistent**, while simultaneously (a) labelling the code block "all pass" and
(b) admitting `B33 8TH` would fail. Its prose description of its own character
class was also wrong (claimed it excluded I and Q; it contains both).

**Conclusion of the error-feedback loop: 4 attempts, 0 working versions, and a
progressive drift into defending the original misdiagnosis.**

### Attempt 5 — fresh chat, "state the rules before writing the regex"

Structure improved dramatically on the first try: `[A-PR-RT-Za-z]{1,2}[0-9][A-Za-z0-9]?\s?[0-9][A-PR-RT-Z]{2}`
handled `EC1A`, `B33` and `M11` correctly — something four rounds of error feedback
never achieved. **12/13 tests passed.**

Remaining failure was pure knowledge, not reasoning: it believed the excluded
inward letters were Q, X, Z (they are **C, I, K, M, O, V**), so `AB12 3CD` passed.
Its regex also did not match its own stated rules (`[A-PR-RT-Z]` excludes Q and S,
not Q, X, Z), and `re.IGNORECASE` plus an `a-z` range neutralised the restriction
entirely.

### Attempt 6 — fresh chat, real rules supplied in the prompt

Given the exact positional rules, it reproduced them correctly and built the six
outward-code shapes properly. But it then emitted:

```python
pattern = r'^(...)\s?\d[A-Z&&[^CIOKV]]{2}$'
```

- `[A-Z&&[^CIOKV]]` is **Java** character-class intersection syntax. Python's `re`
  treats those as literals. The model's own note claimed this syntax "is specific to
  Python's `re` module" — exactly backwards.
- Literal spaces inside a character class: `[A-BEHMNP RWX Y]`.
- Alternation ordered shortest-first, so `M11` matches the `A9` branch and then
  fails the remainder.

Net result: **every valid postcode now fails.**

### Summary

| # | Approach | Outcome |
|---|---|---|
| 1 | Zero-shot | Own test suite fails; 4 distinct regex bugs |
| 2 | Error feedback | Misdiagnosis; broke 5 passing tests |
| 3 | Error feedback | Fixed one case, broke two; **falsely claimed all tests pass** |
| 4 | Error feedback | Reverted to #2; blamed the test suite; self-contradictory |
| 5 | Fresh chat + plan-first | **12/13** — structure correct, knowledge wrong |
| 6 | Fresh chat + rules supplied | Rules applied correctly, **wrong language's regex syntax** |

**Findings:**
1. **Error feedback within a single context degraded performance.** Once a wrong
   theory ("two-digit districts are invalid") entered the context, every subsequent
   turn defended it. A fresh context outperformed four rounds of correction.
2. **Confident false verification claims appeared repeatedly.** The model cannot
   execute code, yet asserted test outcomes as fact in attempts 3, 5 and 6. This is
   the single most dangerous behaviour observed — it *looks* like verification.
3. **Planning fixed reasoning; it could not fix knowledge.** "State the rules first"
   produced correct structure but could not supply facts the model didn't have.
4. **Supplying knowledge introduced errors elsewhere.** Every round fixed the thing
   it was pointed at and broke something it wasn't. This is the strongest argument
   for an automated full-suite regression check rather than spot-fixing.
5. Tool-calling behaviour was observed unprompted: the model attempted to invoke a
   browser `get_info` tool exposed by the llama.cpp web UI. Relevant to future agent
   work; denied here.

### Reference implementation

A regex that passes all 13 asserts:

```python
r'^[A-PR-UWYZ]([0-9]{1,2}|[A-HK-Y][0-9]{1,2}|[0-9][A-HJKPS-UW]|[A-HK-Y][0-9][ABEHMNPRV-Y])\s?[0-9][ABD-HJLNP-UW-Z]{2}$'
```

---

## 6. What's next

### Immediate (finishes session 1)
- [ ] Re-run §4.2 thread sweep on an idle machine (download was competing for CPU)
- [ ] MoE naive vs `-ncmoe` sweep (§4.3) — expected to be the headline chart
- [ ] Save every run as CSV: `-o csv > results/<name>.csv`
- [ ] Record NVIDIA driver version and verify XMP/EXPO is enabled (Task Manager →
      Performance → Memory → Speed; DDR5 rated ≈ 5600–6400, 4800 means XMP is off)
- [ ] Set **CUDA → Sysmem Fallback Policy → Prefer No Sysmem Fallback** for
      `llama-bench.exe` and `llama-server.exe`, so VRAM overflow errors out loudly
      instead of silently spilling into RAM and contaminating results

### Tier 3 — disk streaming
- [ ] Download gpt-oss-120b (~65 GB; get an HF token first — resumable and faster)
- [ ] Measure cold-cache load and generation from NVMe (must exceed RAM+VRAM ≈ 72 GB
      to genuinely stream; record free RAM before each run)
- [ ] Optional "worst case" tier: same model from the SATA HDD (~0.2 GB/s)
- [ ] `mmap` vs `--load-mode none` comparison

### Context / KV cache
- [ ] `llama-bench -d 0,4096,8192,16384` — speed vs context depth
- [ ] `-fa on` and `-ctk q8_0 -ctv q8_0` impact on VRAM headroom and speed

### Quality track
- [ ] Attempt 7: point out the `&&` syntax error and literal spaces, count rounds
- [ ] Compare Qwen3-30B-A3B vs **Qwen3-Coder-30B-A3B** on the same task
- [ ] Fixed sampling for comparability: `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0`
- [ ] Build a 10-question suite (reasoning + coding), auto-scored by execution

### Harness (project #1)
The manual process here — running commands, pasting tables, eyeballing code — is
the argument for automating it:
- YAML config → matrix of model × `-ngl` × `-ncmoe` × threads runs
- Capture hardware metadata (GPU, driver, RAM speed, llama.cpp build) with every row
- Results into SQLite/Parquet; charts generated, not hand-made
- Quality track via `llama-server`'s OpenAI-compatible API: generate → **execute** →
  score → feed real errors back. Directly addresses finding #2 above.
- Flag noisy runs (large ±, competing CPU load) automatically

### Stretch — autonomous strategy search
A 24-hour agent loop where the model proposes trading rules as JSON and a
backtester scores them against historical Indian market data, with the best
candidates then validated on held-out years to measure overfitting. Paper trading
only; single-tool agent (backtester only); no broker connectivity. Feeds the
separate MCP paper-trading platform project. **Not investment advice — the point of
the experiment is to measure how much apparent edge survives out-of-sample.**

---

## 7. Commands reference

```powershell
# Interactive chat
.\llama-cli.exe -m C:\llm\models\Qwen3-8B-Q4_K_M.gguf -ngl 99

# Web UI + OpenAI-compatible API on :8080
.\llama-server.exe -m C:\llm\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  -ngl 99 -ncmoe 44 -t 16 -c 32768 -ctk q8_0 -ctv q8_0 -fa on `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --port 8080

# Benchmark, CSV out
.\llama-bench.exe -m <model> -ngl 99,27,18,9,0 -o csv > results\run.csv
```

| Flag | Meaning |
|---|---|
| `-ngl N` | Layers resident on GPU (99 = all) |
| `-ncmoe N` | Keep expert (MoE) tensors for N layers on CPU |
| `-t N` | CPU threads |
| `-dev none` | Disable GPU entirely (true CPU-only test) |
| `-c N` | Context size in tokens |
| `-ctk/-ctv` | KV cache quantisation (e.g. `q8_0`) |
| `-fa on` | Flash attention |
| `-d N` | Benchmark at a given context depth |
| `-o csv` | Machine-readable output |
