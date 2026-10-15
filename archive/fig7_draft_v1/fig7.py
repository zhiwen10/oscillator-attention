"""Figure 7: attention in a coupled-oscillator network, simulated (osc_attn.py; Kuramoto term from Un-0)."""
import sys
from figstyle import *
from osc_attn import (make_task, continuity_kernel, run_oscillators, phase_attention,
                      pairwise_binding_accuracy, stacked_local_attention_accuracy)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'fig7'
OBJ = [BLUE, ORANGE, AQUA]
fig = plt.figure(figsize=(15 / 2.54, 10.0 / 2.54), dpi=300)

P, V, L, F = make_task(seed=0)
K = continuity_kernel(P, V, F)
theta, snaps = run_oscillators(K, steps=600, record=(0, 150, 600))

# ---------------- A: task ----------------
axA = fig.add_axes([0.045, 0.56, 0.24, 0.36])
for o in range(3):
    m = L == o
    idx = np.argsort(F[m])
    axA.plot(P[m][idx, 0], P[m][idx, 1], color=tint(OBJ[o], 0.35), lw=2.2, solid_capstyle='round', zorder=1)
axA.scatter(P[:, 0], P[:, 1], s=4, c=INK2, lw=0, zorder=2)
axA.set_xticks([]); axA.set_yticks([]); axA.set_aspect('equal')
for s in axA.spines.values():
    s.set_color(GRID)
axA.set_title('连续运动的多个物体', fontsize=6.8, color=INK, pad=3)
axA.text(0.5, -0.07, '每个点 = 一个词元（位置 + 速度）\n顺序打乱、无标签', transform=axA.transAxes, ha='center',
         va='top', fontsize=5.7, color=INK2, linespacing=1.2)

# ---------------- B: phases synchronise ----------------
for k, (st, lab) in enumerate([(0, '初始'), (150, '演化中'), (600, '收敛')]):
    ax = fig.add_axes([0.34 + k * 0.105, 0.60, 0.095, 0.30], projection='polar')
    th = snaps[st]
    for o in range(3):
        m = L == o
        ax.scatter(th[m], np.ones(m.sum()) + 0.06 * np.random.default_rng(o).standard_normal(m.sum()), s=3.2,
                   color=OBJ[o], lw=0)
    ax.set_ylim(0, 1.3); ax.set_yticks([]); ax.set_xticks([])
    ax.spines['polar'].set_color(BASE); ax.grid(False)
    ax.set_title(lab, fontsize=6.2, color=INK2, pad=2)
fig.text(0.5, 0.955, '振荡器相位自发同步为三簇', ha='center', va='top', fontsize=6.8, color=INK)

# ---------------- C: attention matrices ----------------
order = np.lexsort((F, L))
A_osc = phase_attention(theta)[np.ix_(order, order)]
A_loc = K / K.sum(1, keepdims=True).clip(1e-9)
A_loc = (A_loc / A_loc.max(1, keepdims=True))[np.ix_(order, order)]
for k, (M, lab) in enumerate([(A_loc, '单层局部注意力'), (A_osc, '相位差读出')]):
    ax = fig.add_axes([0.70 + k * 0.145, 0.60, 0.13, 0.30])
    ax.imshow(M, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASE)
    ax.set_title(lab, fontsize=6.2, color=INK2, pad=2)
    n = len(L) // 3
    for o in range(3):
        ax.add_patch(Rectangle((-2.5, o * n - 0.5), 1.6, n, fc=OBJ[o], ec='none', clip_on=False))
fig.text(0.845, 0.955, r'$A_{ij}=(1+\cos(\theta_i-\theta_j))/2$', ha='center', va='top', fontsize=6.5, color=INK)

# ---------------- D: one oscillator layer vs stacked local attention ----------------
accs_osc, curves = [], []
for seed in range(8):
    P2, V2, L2, F2 = make_task(seed=seed)
    K2 = continuity_kernel(P2, V2, F2, window=3)
    th2, _ = run_oscillators(K2, steps=1200, seed=seed)
    accs_osc.append(pairwise_binding_accuracy(phase_attention(th2) > 0.75, L2))
    curves.append(stacked_local_attention_accuracy(K2, L2, 14))
curves = np.array(curves)
axD = fig.add_axes([0.07, 0.10, 0.22, 0.32])
lay = np.arange(1, 15)
axD.plot(lay, curves.mean(0), color=MUTED, lw=1.4, marker='o', ms=2.5)
axD.axhline(np.mean(accs_osc), color=BLUE, lw=1.4)
axD.text(14, np.mean(accs_osc) - 0.03, '一个振荡网络（共享耦合）', ha='right', va='top', fontsize=5.7, color=INK)
axD.text(4.6, 0.64, '堆叠局部注意力', ha='left', fontsize=5.7, color=INK)
axD.set_xlim(0.5, 14.5); axD.set_ylim(0.5, 1.05); axD.set_xticks([1, 5, 10, 14]); axD.set_yticks([0.5, 0.75, 1.0])
axD.set_xlabel('注意力层数', labelpad=1); axD.set_ylabel('绑定准确率', labelpad=1)
spines(axD)

# ---------------- E: travelling wave = order code ----------------
th_w, _ = run_oscillators(K, alpha_dir=0.1, F=F, steps=1500)
axE = fig.add_axes([0.40, 0.10, 0.22, 0.32])
for o in range(3):
    m = L == o
    idx = np.argsort(F[m])
    ph = np.unwrap(th_w[m][idx])
    ph = ph - ph[0]
    axE.plot(F[m][idx], ph, 'o-', color=OBJ[o], lw=1.0, ms=2.2)
axE.set_xlabel('时间帧', labelpad=1); axE.set_ylabel('相位（rad）', labelpad=1)
axE.set_xticks([0, 10, 20]); axE.set_title('相位滞后 → 物体内形成行波', fontsize=6.3, color=INK2, pad=2)
axE.text(0.97, 0.06, '相位梯度编码时间顺序', transform=axE.transAxes, ha='right', fontsize=5.7, color=INK)
spines(axE)

# ---------------- F: capacity vs number of heads ----------------
cap = np.load('capacity.npy')
axF = fig.add_axes([0.74, 0.10, 0.22, 0.32])
for h, col in zip([1, 2, 4], ['#9ec5f4', '#3987e5', '#104281']):
    r = cap[cap[:, 1] == h]
    axF.errorbar(r[:, 0], r[:, 2], yerr=r[:, 3], color=col, lw=1.3, ms=3, marker='o', capsize=1.5, elinewidth=0.6)
    axF.text(9.35, r[-1, 2], '%d 头' % h, fontsize=5.8, color=INK, va='center')
axF.set_xlim(2.5, 10.5); axF.set_ylim(0.75, 1.02); axF.set_xticks([3, 5, 7, 9]); axF.set_yticks([0.8, 0.9, 1.0])
axF.set_xlabel('物体数量', labelpad=1); axF.set_ylabel('绑定准确率', labelpad=1)
axF.set_title('多头振荡提高绑定容量', fontsize=6.3, color=INK2, pad=2)
spines(axF)

for x, y, Lt in [(0.005, 0.985, 'A'), (0.31, 0.985, 'B'), (0.66, 0.985, 'C'), (0.005, 0.47, 'D'),
                 (0.34, 0.47, 'E'), (0.675, 0.47, 'F')]:
    letter(fig, x, y, Lt)
save(fig, OUT)
print('osc acc', np.mean(accs_osc), 'stack', np.round(curves.mean(0), 3))
