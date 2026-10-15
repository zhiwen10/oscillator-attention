# Figure 7: binding, scaling, and attention with coupled oscillators

This folder records the final three-panel summary figure (finalised 2026-09-26).

| File | What |
|---|---|
| `fig7_zh.png/.pdf` | **Final version.** |
| `fig7_draft_v2_zh.png/.pdf` | Intermediate hand-edited version (not used). |

Both were hand-arranged versions of the script output (same panels and data, the panel layout adjusted by hand).
They were saved as `results/final_figure_zh -V2/-V3.*` and moved here on 2026-09-26.

## How to regenerate

```
conda activate osc-attention
python experiments/make_final_figure.py --lang zh     # -> results/final_figure_zh.png/.pdf
python experiments/make_final_figure.py               # English labels -> results/final_figure.png/.pdf
```

Verified 2026-09-26: rerunning the script reproduces `results/final_figure_zh.png` byte-for-byte. It runs on a CPU in a few
seconds and does **not** need the 3.9 GB Un-0 checkpoint, because panel B reads the saved `results/un0_dynamics.pt`.

## Where each panel comes from

| Panel | Claim in the caption | Source |
|---|---|---|
| A 振荡同步实现绑定 | 3 objects, 72 shuffled tokens; binding accuracy 100% | recomputed on each run: `oscattn.make_task(seed=0)` → `continuity_coupling` → `simulate(steps=1200)` (`oscattn/{tasks,coupling,dynamics,readout}.py`) |
| B 振荡网络可扩展 | Un-0, 16,384 oscillators; 10 classes × 40 samples; class decoding 10% → 97% | `results/un0_dynamics.pt`, made by `experiments/un0_dynamics.py` (needs `checkpoints/imagenet64_n16384.pt` and a GPU; see `experiments/un0_samples.py` for the download) |
| C 振荡实现注意力 | 1-layer oscillator attention 99.5% vs 1-layer softmax 21% (~58k params each); removing dynamics or the phase lag hurts; one dot per seed | `results/assoc_p8.json`, made by `python experiments/assoc_learning.py --tag p8` (5 seeds) |

Panel layout code: `experiments/make_final_figure.py`, which reuses panel functions from `make_simple_figure.py` (A) and
`make_overview_figure.py` (B, C).

## Figure caption

> 图7  耦合振荡网络的绑定、扩展与注意力（数值模拟）。（A）振荡同步实现绑定：三个运动物体的72个观测（位置+速度）顺序打乱且无标签；每个观测对应一个振荡器，只与相邻时刻、运动相符的观测局部耦合；由最终相位差读出的注意力矩阵（箭头上）呈按物体划分的块状结构，按相位分组即恢复三条完整轨迹（绑定准确率100%）。（B）振荡网络可扩展：本地运行Un-0（16,384个耦合振荡器），振荡器状态的主成分投影显示，所有样本从随机相位出发演化为各类别簇（10类×40个样本），虚线示两个样本的轨迹及其生成图像；仅由振荡器状态解码类别的准确率由10%升至97%。（C）振荡实现注意力：在联想回忆任务（先给出若干“刺激—反应”配对，再以其中一个刺激提问）中，单层振荡注意力的准确率达99.5%，同等参数（约5.8万）的单层常规注意力为21%；去除同步动力学或相位滞后后准确率明显下降。每点为一个随机种子。

## What the figure does not claim

The later, more careful comparison (`results/assoc_p8f.json`, AGENTS.md §5) found that oscillator attention does **not**
learn faster than a looped 1-layer softmax control (96k vs 51k sequences to 95%). Learning efficiency is therefore
stated as an open hypothesis, not as a preliminary result.
