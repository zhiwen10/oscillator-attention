# Fig. 7 draft v1 (superseded)

The first version of the summary figure (six panels, 2026-09-25), drafted before this repo existed.
It was replaced by `fig7/fig7_zh.png`. Copied here unchanged, for the record.

| File | What |
|---|---|
| `osc_attn.py` | NumPy-only first draft of the binding simulation (Un-0 Kuramoto term, continuity coupling, phase readout, stacked local-attention baseline). Later rewritten in PyTorch as `oscattn/`. |
| `fig7.py` | Six-panel figure: A task, B phase synchrony, C attention matrices, D oscillator vs stacked local attention, E travelling wave, F capacity vs heads |
| `figstyle.py` | Shared matplotlib style for these figures |
| `capacity.npy` | Data for panel F: rows of (n objects, n heads, mean accuracy, s.d.) |
| `fig7_draft_v1_zh.png/.pdf` | The rendered figure (图7 振荡注意力模拟） |

```
cd archive/fig7_draft_v1
python fig7.py fig7_out          # -> fig7_out.png/.pdf/.svg  (numpy + matplotlib only; still runs as of 2026-09-26)
```

**Caveat:** the script that produced `capacity.npy` was not saved. Its numbers do not match the later
`experiments/run_experiments.py` capacity experiment (`results/results.json`, `exp4_capacity`); for example, 1 head with
3 objects gives 0.948 here and 0.90 there. Use the repo version if you need capacity numbers.
