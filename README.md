# oscillator-attention

Modelling work testing whether a network of coupled oscillators can do what Transformer attention does,
with the attention weights read straight from phase differences. The Kuramoto dynamics come from
[Un-0](https://github.com/unconv-ai/Un-0) (MIT licence).

- **The three-panel summary figure (Fig 7)** and how to rebuild it: [`fig7/README.md`](fig7/README.md)
- **Full lab notebook** (all results, including negative ones, bug history, fairness rules, next steps): [`AGENTS.md`](AGENTS.md)

## Tutorials

Four executed notebooks walk through the work, one per Fig 7 panel plus the fair comparison
(index: [`notebooks/README.md`](notebooks/README.md)). They run start-to-finish in a few minutes on CPU —
long trainings are not re-run; saved results are loaded from `results/`:

1. [`notebooks/01_binding_by_synchrony.ipynb`](notebooks/01_binding_by_synchrony.ipynb) — panel A: synchrony groups shuffled observations of moving objects.
2. [`notebooks/02_un0_scaling.ipynb`](notebooks/02_un0_scaling.ipynb) — panel B plus all Un-0 artefacts (class decoding over the dynamics, class-conditional samples, subspace analysis).
3. [`notebooks/03_oscillator_attention.ipynb`](notebooks/03_oscillator_attention.ipynb) — panel C: a learned oscillator attention layer does associative recall.
4. [`notebooks/04_fair_comparison.ipynb`](notebooks/04_fair_comparison.ipynb) — the honest comparison against looped softmax, and the rhythm-task negative result.

## Project record

All work was done 2026-09-25/26.

| Stage | Question | Code | Result | In Fig 7? |
|---|---|---|---|---|
| 0. Draft | first NumPy sketch of binding by synchrony | `archive/fig7_draft_v1/` | six-panel draft figure | no |
| 1. Binding | can synchrony group shuffled observations of moving objects? (hand-built coupling, nothing learned) | `oscattn/{tasks,coupling,dynamics,readout}.py`, `experiments/run_experiments.py` | 100% binding; travelling wave encodes order; heads raise capacity | **Fig 7A** |
| 2. Scale | do real, large oscillator networks show the same dynamics? | `experiments/un0_samples.py`, `experiments/un0_dynamics.py` | Un-0 (16,384 oscillators): random phases → class clusters, class decoding 10% → 97%. Beyond the figure: uncurated class-conditional samples (`results/un0_grid.png`, `results/un0_samples.pt`) and a subspace analysis (`results/un0_subspace.png`) — see notebook 02 | **Fig 7B** |
| 3. Attention | can a *learned* oscillator layer do associative recall? | `oscattn/{assoc,models}.py`, `experiments/assoc_learning.py --tag p8` | 1-layer oscillator 99.5% vs 1-layer softmax 21%; dynamics and lag both needed | **Fig 7C** |
| 4. Fair efficiency test | is the oscillator *more efficient* than softmax, against a looped-softmax control? | same, `--tag p8b/p8c` → merged `p8f`; `assoc_mechanism.py` | **No** (96k vs 51k sequences to 95%). Freezing query phases fixes length generalisation. | framed as an open question, not a result |
| 5. Rhythmic task | does a world of repeating events favour oscillators? | `oscattn/rhythm.py`, `experiments/rhythm_learning.py` | **No**: the oscillator fails at chance level | no |

## Repository layout

```
oscattn/          library (pip install -e .)
  tasks.py coupling.py dynamics.py readout.py   stage 1: moving-objects binding (Un-0 Kuramoto update)
  assoc.py models.py rhythm.py                  stages 3-5: associative recall, SoftmaxAttention vs OscAttention
experiments/      one script per experiment or figure (each has a usage docstring)
  make_final_figure.py                          -> Fig 7 (reuses make_simple_figure.py, make_overview_figure.py)
results/          JSON/logs of every run + figures (.pt files are git-ignored except un0_dynamics.pt)
fig7/             the final Fig 7 (panels + caption + provenance)
archive/          superseded first draft of Fig 7
notebooks/        executed tutorial notebooks, one per Fig 7 panel + the fair comparison (index: notebooks/README.md)
tests/            maths checks (Un-0 update = textbook Kuramoto sum; oscillator layer; parameter counts)
checkpoints/      Un-0 ImageNet-64 checkpoint, 3.9 GB, git-ignored (download: see experiments/un0_samples.py)
```

## Setup

From the repository root (the `-e .` in `environment.yml` installs the `oscattn` package itself):

```
conda env create -f environment.yml      # creates the "osc-attention" environment
conda activate osc-attention
python -m pytest tests -q                # verify: 7 checks pass
```

Optional — only needed to re-run the Un-0 stage-2 scripts (`experiments/un0_*.py`); the notebooks do **not** need it:

```
pip install --no-deps "un0 @ git+https://github.com/unconv-ai/Un-0.git@75243cab846c092527734925c2433b556f5e5ee7"
curl -L -o checkpoints/imagenet64_n16384.pt https://huggingface.co/un-ai/Un-0/resolve/main/imagenet64_n16384.pt   # 3.9 GB
```

PyTorch is the CUDA 12.8 build (needed for RTX 50-series GPUs); it installs and runs fine on CPU-only machines
too — everything except training and the Un-0 runs works on a CPU.
nbformat, nbclient and ipykernel are included so the tutorial notebooks in `notebooks/` can be executed headlessly.

## Run

```
python -m pytest tests -q                              # 7 checks
python experiments/make_final_figure.py --lang zh      # Fig 7 from saved results (seconds, CPU)

# re-run the underlying experiments
python experiments/run_experiments.py                  # stage 1 -> results/results.json (~1 min, CPU)
python experiments/un0_dynamics.py                     # stage 2 -> results/un0_dynamics.pt (needs checkpoint + GPU)
python experiments/assoc_learning.py --steps 4000 --tag p8    # stage 3 (~35 min, GPU)
python experiments/make_assoc_figure.py --tag p8f      # stage 4 summary table + figure
python experiments/rhythm_learning.py --tag r1         # stage 5
```

## Stage 1 in more detail: binding by synchrony

| Transformer | Oscillator network |
| --- | --- |
| each token has a query and a key | each token is an oscillator with a phase θ |
| attention = softmax(q·k) | tokens that belong together synchronise; others drift apart |
| weights computed for every pair | attention read out as A_ij = (1 + cos(θi − θj)) / 2 |
| position encoding added by hand | a phase lag makes a travelling wave; the phase gradient encodes order |
| multi-head attention | several independent phase sets per token ("heads") |

**Task.** Three (or more) objects move along smooth, crossing paths. Each observation (position and
velocity at one frame) is a token; tokens are shuffled and unlabelled. The network must group the tokens
by object. Coupling is local: a token only pulls on tokens where its own motion predicts it will be 1–3
frames away. Binding a whole trajectory therefore has to emerge from the dynamics.

**Results** (`results/results.json`):

1. **Binding.** From random phases, the phase readout groups every token correctly (accuracy 0.50 → 1.00 by step 600).
2. **Depth.** One oscillator network with a single shared coupling matrix reaches 0.94 (8 tasks), which takes about
   6 stacked local-attention layers; deeper stacks reach 1.00.
3. **Order.** With a phase lag, phase rises linearly with time inside every object (r = 1.00): a travelling wave that
   encodes order. It costs some binding accuracy (0.89).
4. **Capacity.** With one phase per token, accuracy falls from 0.90 (3 objects) to 0.83 (9 objects); four heads keep
   it at 0.96.

Caveat: on a digital computer the ~1,200 integration steps cost more than 6 attention layers. The case for oscillators is
weight sharing and physical (analog) hardware, not speed on a GPU.

## Licence and credits

The Kuramoto update is ported from Un-0 (MIT, © 2026 Unconventional, Inc.). The Un-0 checkpoint belongs to its authors and is not
redistributed here.
