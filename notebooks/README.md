# Tutorial notebooks

Four executed tutorials, one per panel of the summary figure (图7) plus the fair comparison.
All run start-to-finish in a few minutes on CPU; long trainings are not re-run — the notebooks load
saved results from `results/` and only re-compute the cheap parts.

| Notebook | Question | What it runs | Runtime |
|---|---|---|---|
| `01_binding_by_synchrony.ipynb` | Can synchrony group shuffled observations of moving objects? (panel A) | Simulates the stage-1 coupled oscillators live (hand-built coupling, nothing learned) and plots binding, order, depth, capacity | ~1 min, all live, CPU |
| `02_un0_scaling.ipynb` | Does a real 16,384-oscillator network (Un-0, ImageNet-64) show the same dynamics? (panel B) | Loads `results/un0_dynamics.pt` (PCA trajectory + class decoding per step), shows uncurated class-conditional samples (`un0_grid.png` / `un0_samples.pt`) and the subspace analysis (`un0_subspace.png`) | seconds, saved results only (the model itself needs the 3.9 GB checkpoint + GPU; see `experiments/un0_samples.py`) |
| `03_oscillator_attention.ipynb` | Can a *learned* oscillator attention layer do associative recall? (panel C) | Trains a tiny oscillator model live for a quick demo, then loads `results/assoc_p8*.json` for the full curves, ablations, and the mechanism analysis (`assoc_mechanism*.json`) | a few minutes, CPU |
| `04_fair_comparison.ipynb` | Is the oscillator *more efficient* than softmax, against a looped-softmax control? And does a rhythmic world favour oscillators? | Loads `results/assoc_p8f.json` and `results/rhythm_r1.json`; states both negative results (1.9× slower to 95%; chance level on the rhythm task) | seconds, saved results only |

## Re-executing

The notebooks resolve the repo root themselves, so they work whether the working directory is the repo
root or `notebooks/`. On this machine, set `PYTHONUTF8=1` — without it, the GBK console locale makes the
executor misread the UTF-8 notebook files:

```
PYTHONUTF8=1 C:/Users/dell/anaconda3/envs/osc-attention/Scripts/jupyter-execute.exe --inplace notebooks/01_binding_by_synchrony.ipynb
```

or open any notebook in Jupyter and run all cells (kernel `python3` from the `osc-attention` conda env).
The committed notebooks are already executed top-to-bottom with outputs embedded.
