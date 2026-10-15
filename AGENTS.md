# AGENTS.md — handoff for the next agent

Read this first. It says what the project is for, what has been done, what the results are, and what to
do next. Last updated 2026-09-26 (repo organised and committed, see section 5.3).
Executed tutorial notebooks added 2026-09-27 for repo clarity (see `notebooks/` row in section 3).

## 1. Purpose

This project investigates whether **attention can be implemented by coupled oscillators**: phase alignment
does the query–key match, and a travelling wave (phase lag) supplies position/order.

**The central hypothesis:** an attention mechanism built from coupled oscillators works, and is
*more efficient at learning an association task* than a conventional network with standard softmax attention.

The job is to test that hypothesis honestly, in a small model, and produce a figure + short text that
survive obvious controls. **A result that falls to an obvious control is worse than a modest result that
survives it.** Do not tune until the oscillator wins; see section 6.

## 2. Environment

- Windows 11, GPU RTX 5070 Ti. Conda env `osc-attention` (Python 3.12, PyTorch 2.11 + CUDA 12.8).
- Python: `C:/Users/dell/anaconda3/envs/osc-attention/python.exe` (plain `python` on PATH is a Store stub).
- Package installed editable (`pip install -e .`), import as `oscattn`. Un-0 (MIT) is installed for its
  Kuramoto update; only the old binding experiment uses it.
- Tests: `python -m pytest tests -q` → 7 pass.
- Notebooks: nbformat/nbclient/ipykernel are in `environment.yml`. Re-execute with
  `PYTHONUTF8=1 .../Scripts/jupyter-execute.exe --inplace <nb>` — without PYTHONUTF8 the GBK console
  locale makes the executor misread the UTF-8 notebooks. Notebooks resolve the repo root themselves.
- Git: all work is committed. Commit the current state before changing anything.

## 3. Files

| File | What |
|---|---|
| `oscattn/assoc.py` | **Association task.** `make_batch`: n key→value pairs then a query key; target = its value. Fresh pairings every sequence (in-context associative recall). |
| `oscattn/models.py` | **Models.** `SoftmaxAttention` (RoPE), `OscAttention` (oscillators), `Block`, `Model` (supports `n_layers`, `loops` = weight-tied repeats). Docstring explains the maths. |
| `experiments/assoc_learning.py` | **Main comparison.** Trains every model in `MODELS` × `--lrs` × `--seeds`, logs a learning curve, examples-to-95%, and length generalisation. Writes `results/assoc_<tag>.json`. |
| `experiments/make_assoc_figure.py` | Summary table (stdout) + 3-panel figure `results/assoc_<tag>.png/.pdf` (`--lang zh` for Chinese labels). Picks each model's lr by one rule (best mean accuracy over the curve). |
| `experiments/assoc_mechanism.py` | Trains one oscillator model and measures how the query's phase coupling to the answer grows over the dynamics steps. `--variant staticq` for the frozen-query variant. → `results/assoc_mechanism*.json/.png` |
| `experiments/screen_variants.py` | 3000-step seed-0 screening used to test option-B variants before full runs (edit `CONFIGS`). |
| `oscattn/rhythm.py`, `experiments/rhythm_learning.py` | **Rhythmic task** (cycle of C tokens repeated R times, query → cyclic successor). Same comparison protocol; writes `results/rhythm_<tag>.json`. |
| `experiments/sweep_osc.py` | The ad-hoc sweep used to pick oscillator settings (results in section 5). |
| `tests/test_models.py` | Checks the vectorised oscillator step equals the explicit Kuramoto–Sakaguchi sum; checks parameter counts match; regression test that receiver phases evolve. |
| `oscattn/{tasks,coupling,dynamics,readout}.py`, `experiments/{run_experiments,make_figure}.py`, `tests/test_dynamics.py`, `results/{results.json,figure*}` | **Older experiment** (grouping moving objects by synchrony, hand-built coupling, nothing learned). The owner judged it the wrong example for the main claim, but it became panel A of the summary figure. Keep it, don't extend it. README.md describes it. |
| `experiments/un0_samples.py`, `experiments/un0_dynamics.py` | Run Un-0's released 16,384-oscillator ImageNet-64 model locally (checkpoint in git-ignored `checkpoints/`, 3.9 GB) and track the oscillator state over its 10 steps: PCA trajectory + class decoding per step → `results/un0_dynamics.pt`, `un0_dynamics*.png`, `un0_grid.png`. |
| `experiments/make_simple_figure.py`, `make_overview_figure.py`, `make_final_figure.py` | Summary figures, in order of creation. **`make_final_figure.py` makes the final Fig 7** (A binding, B Un-0 scale, C associative recall from `assoc_p8.json`). See `fig7/README.md`. |
| `fig7/` | Fig 7 as finalised (hand-arranged from `results/final_figure_zh.*`), caption, panel provenance. |
| `archive/fig7_draft_v1/` | First NumPy draft of Fig 7 (superseded). |
| `notebooks/` | **Four executed tutorial notebooks** (see `notebooks/README.md`), one per Fig 7 panel plus the fair comparison: 01 binding (panel A), 02 Un-0 at scale (panel B + all Un-0 artefacts: samples grid, subspace analysis), 03 oscillator attention (panel C), 04 fair comparison vs looped softmax + rhythm-task negative result. Run on CPU in minutes; long trainings are loaded from `results/`, not re-run. |

Run the main comparison (~35 min on the GPU):

```
python experiments/assoc_learning.py --steps 4000 --tag p8c     # 10 models x 2 lrs x 5 seeds
python experiments/make_assoc_figure.py --tag p8f               # p8f = merged final (see section 5)
python experiments/assoc_mechanism.py [--variant staticq]
python experiments/rhythm_learning.py --tag r1
```

## 4. The oscillator layer (current design)

Per head, per token i, m = d/heads phases:

1. `θ_i = atan2(W_s h_i, W_c h_i) + f_k · i` — learned phase + positional phase (same frequencies as RoPE).
2. `A_ij = softmax_j( β Σ_k w_k cos(θ_ik − θ_jk − α_k) )`, causal. `α` = learned phase lag, `w` = learned
   per-oscillator weights, `β` learned.
3. T steps of Kuramoto–Sakaguchi: `θ_ik += dt · κ Σ_j A_ij sin(θ_jk + α_k − θ_ik)`, recomputing A each step.
4. `out_i = Σ_j A_ij(T) v_j` — values read **once**, from the final phases.

Setting: T = 4, dt = 1.0. W_c, W_s replace W_q, W_k, so the parameter count matches softmax attention
(58,248 vs 58,112 for the 1-layer model, d = 64, 4 heads, MLP 4d).

`OscAttention` options (all default off = the original layer; see its docstring):
- `static_query` — freeze the receiver/query phases at their initial values; only the key phases evolve.
- `separate_qk` — query and key get separate d×d phase projections (params unchanged, m halves); static query by construction.
- `value_consensus` — also diffuse value vectors along the attention graph each dynamics step (failed; see section 5).
- `pos` — 'full' (default) / 'low' (positional ramp only on slow oscillators) / 'none' (failed; see section 5).

## 5. Results so far

### 5.1 Association task (option B executed 2026-09-25)

Task: 8 pairs, 32 keys, 32 values (chance ≈ 3%; picking a random value in the context ≈ 12.5%).
Batch 64, 4000 steps (256k sequences), AdamW, lr ∈ {1e-3, 3e-3}, 5 seeds; lr per model picked by the
same rule (best mean accuracy over the curve). **Definitive source: `results/assoc_p8f.json`**
(= `assoc_p8b.json` softmax/T0/qk runs + renamed static-query runs, + corrected `assoc_p8c.json` runs;
see the bug note below). Figure: `results/assoc_p8f.png`.

| Model | Params | Final acc | Sequences to 95% (median) | Acc on 12 / 16 / 24 pairs (trained on 8) |
|---|---|---|---|---|
| softmax-1L | 58k | 0.21 | never | 0.17 / 0.14 / 0.11 |
| softmax-2L | 108k | 1.00 | 70k | 0.93 / 0.83 / 0.59 |
| softmax-1Lx4 (1 layer looped 4×, shared weights) | 58k | 1.00 | **51k** | 0.92 / 0.82 / 0.61 |
| osc-1L-T4 | 58k | 1.00 | 96k | 0.31 / 0.26 / 0.14 |
| **osc-1L-T4-staticq** (query phases frozen) | 58k | 1.00 | **96k** | **0.78 / 0.62 / 0.41** |
| osc-1L-T4-qk (separate q/k phase projections) | 58k | 1.00 | 115k | 0.90 / 0.77 / 0.37 |
| osc-1L-T0 (no dynamics) | 58k | 0.18 | never | — |
| osc-1L-T4-nolag (α fixed at 0) | 58k | 0.89 | never | 0.74 / 0.52 / 0.28 |
| osc-1L-T1x4 (osc block looped 4×) | 58k | 0.25 | never | — |
| osc-1L-T1x4-staticq | 58k | 0.99 | 173k | 0.78 / 0.62 / 0.35 |

**Bug history (kept for honesty):** the 2026-09-25 refactor of `OscAttention` accidentally froze the
receiver/query phases for non-separate models (aliasing bug). `assoc_p8b.json`'s osc-T4/T1x4/nolag rows
were run with that behavior. It was fixed (current code is bit-identical to the original layer with
default options; `tests/test_models.py::test_receiver_phases_evolve_over_steps` guards it), the affected
models were re-run (`assoc_p8c.json`), and the accidental behavior was kept as the named
`static_query` option because it is better (see below).

**Findings:**
- **Efficiency gap NOT closed.** Best oscillator: 96k sequences to 95% vs 51k for looped softmax (1.9×),
  same as before option B. None of the three option-B ideas beat the base oscillator layer.
- **But the length-generalisation failure is SOLVED and explained:** freezing the query phases
  (`static_query`) fixes it at zero learning-speed cost (0.78/0.62/0.41 vs 0.31/0.26/0.14 at 12/16/24
  pairs). Evolving receiver phases drift into self-locking; the frozen query keeps a stable identity.
  Separate q/k phases (qk) generalise about as well but learn slower.
- **Block-level iteration does not help oscillators.** Looping the osc block 4× (T1x4) *fails* with
  evolving receiver phases (0.25 final) and is 1.8× slower than T4 even with static phases, whereas
  the identical loop structure is what makes 1-layer softmax competitive. The oscillator's own dynamics,
  not block repetition, is its iteration mechanism.
- T0 (0.18) and no-lag (0.89, never 95%) reproduce: dynamics and lag are both necessary.

**Mechanism** (`results/assoc_mechanism.json` base, `assoc_mechanism_staticq.json`, both 99.8% test acc):
- Base: answer coupling rises 0.10 → 0.55 over the 4 steps; final attention spread (answer 11%, matching
  key 16%, self 34%); heads essentially never attend most to the answer (0–2% of sequences).
- staticq: answer coupling 0.05 → 0.32 while other responses stay flat (0.04 → 0.07) — better
  selectivity; final attention answer 25%, self 16%; 2 of 4 heads attend most to the answer in 53–58%
  of sequences. Cleaner "attend to the answer" readout, consistent with its better generalisation.

**Option-B variant screening** (3000 steps, seed 0, both lrs; `results/screen_variants*.log`):
- `value_consensus` κ_v=1.0: **fails** (0.19). Full-replacement consensus `r ← A@r` smears value identity
  before attention sharpens. κ_v=0.25 learns (0.95 at 3e-3) but slower than base. Dead end.
- `pos='low'` (ramp only on slow oscillators): **fails** (0.75 at best) — the task needs the fast
  positional phases. `pos='none'` was never expected to work (ablation).
- `separate_qk`: learns (0.99@3e-3) at base-like speed; kept for its generalisation.
- T2-x2, T1-x4-qk: slower than base; not pursued.

**Tuning history** (3000 steps, seed 0, accuracy every 500 steps):
- Without lag: T=8/dt=0.5/lr=3e-3 → 0.86; T=4/dt=1.0/lr=1e-3 → 0.86; T=16 variants 0.19–0.58
  (longer dynamics train worse); initial β=10 → 0.17.
- With lag: T=4/dt=1.0 → 0.99 (lr 1e-3 and 3e-3); T=8/dt=0.5 → 0.995–0.999.
- Looped softmax ×4 at lr 3e-3 → 1.0 by step 1000.

### 5.2 Rhythmic-world task (the owner's "oscillatory events" hypothesis)

Task (`oscattn/rhythm.py`): cycle of C∈{5..8} tokens repeated R=3, query one cycle token, target its
cyclic successor (chance 1/C). Same protocol/lr grid/seeds; softmax-2L and looped softmax included.
Source: `results/rhythm_r1.json`.

| Model | Final acc | Sequences to 95% | Gen: R=1/2/4/6 | Gen: C=4/10/12 |
|---|---|---|---|---|
| softmax-1L | 0.19 (chance) | never | — | — |
| softmax-2L | 1.00 | **32k** | 0.80/0.99/0.99/0.94 | 0.99/0.99/0.95 |
| softmax-1Lx4 | 1.00 | 51k | 0.70/0.94/0.98/0.81 | 0.96/0.96/0.89 |
| **osc-1L-T4** | **0.19 (chance)** | never | — | — |
| osc-1L-T4-staticq | 0.67 | never | 0.46/0.53/0.56/0.39 | 0.69/0.32/0.20 |

**The oscillatory-world hypothesis is NOT supported in this form.** Periodic input makes the task
*easier* for softmax (successor relation repeated R× per sequence; 2L needs only 32k sequences) and
*harder* for the oscillator (fails at chance): identical tokens repeating at different cycle phases
lock onto each other — content phases are identical, only the positional ramp separates copies — so the
query couples to its own occurrences instead of the successor. Phase identity conflates "same token"
with "same phase"; repeating content is precisely the adversarial case. If a "world with rhythms" story
is wanted, the rhythm must live in the *timing* (distinct tokens at rhythmic intervals), not in repeated
identical tokens.

### 5.3 Summary figure (Fig 7) (2026-09-25 evening, organised 2026-09-26)

The final 图7 (`fig7/fig7_zh.png`) combines three results that survive obvious controls:
A binding by synchrony (stage 1, 100%), B Un-0 at scale (class decoding 10% → 97% over the oscillator dynamics),
C 1-layer oscillator attention vs 1-layer softmax on associative recall (99.5% vs 21%, `assoc_p8.json`) with the
no-dynamics and no-lag ablations. **The efficiency claim was dropped from the figure** (option A of section 7);
learning efficiency is stated as an open hypothesis, not a result. `make_final_figure.py --lang zh`
reproduces `results/final_figure_zh.png` byte-for-byte (checked 2026-09-26). The owner then adjusted the layout by hand
(`-V2`, `-V3`), and V3 is the final version.

## 6. Rules for fair comparison (keep these)

1. Same data stream (same seed), same test set, same embedding/MLP/readout, parameter counts within 1%.
2. Every model gets the same lr grid and the same seed count; the lr is picked by the same rule.
3. **Always include the looped-softmax control** (`softmax-1Lx4`) and the two ablations. Without the
   looped control, "oscillators beat 1-layer attention" is really "iteration beats no iteration".
4. Decide the evaluation (task, metrics, criterion) **before** changing the model, and report whatever
   comes out. Record every variant tried, including failures, in section 5.

## 7. Next steps (option A was taken in the Fig 7 write-up; B–D remain open)

- **A. Write up the supported claim** for the write-up: "coupled-oscillator dynamics with a travelling-wave
  phase lag implement attention-like association; a single oscillator layer achieves associative recall
  that a single attention layer cannot; both synchronisation and the lag are necessary; freezing the
  query phases preserves this ability on longer sequences than trained." Figure: learning
  curves + ablations + mechanism panel (`results/assoc_p8f.png`, `assoc_mechanism_staticq.png`).
  Drop the efficiency claim.
- **B. Remaining bounded ideas for the efficiency gap** (screen first, 3000 steps seed 0):
  1. staticq + qk combined (both help generalisation; maybe combined they help speed).
  2. lr 1e-2 added to the grid for *all* models (looped softmax is lr-robust, osc-T1x4-staticq is not;
     a wider shared grid is fair and might favour osc-staticq).
  3. Multi-hop association (A→B, B→C, query A → C): the one setting where phase synchrony *spreading*
     over T steps could genuinely beat fixed-depth softmax loops. Pre-register before running.
  4. `--pairs 16` (longer context): does the gap shrink when retrieval is harder?
- **C. Rhythmic timing instead of repeated tokens** (follow-up to 5.2): distinct tokens presented at
  rhythmic intervals (rhythm in *timing*, not content). Only worth doing if the owner wants to keep
  pushing the oscillatory-world story.
- **D. Test efficiency where oscillators are meant to win**: energy/compute on analog hardware (Un-0's
  argument), not training speed on a GPU. On a GPU the oscillator is not faster.

## 8. Context

- The motivating biological story (whole-brain rotating waves, Ye et al., Science 2026):
  waves sweep brain regions in order, so temporally adjacent events (stimulus then response) get bound,
  as with STDP-based associative learning. The phase lag in this model is that idea.
- Chinese figure labels are available where needed (`--lang zh`).
- Be candid about negative results; the aim is a defensible claim.
