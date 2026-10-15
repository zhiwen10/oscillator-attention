"""Shared style for the figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge, Circle

plt.rcParams.update({
    'font.family': ['Arial', 'Microsoft YaHei'], 'font.size': 7,
    'axes.linewidth': 0.6, 'axes.edgecolor': '#898781',
    'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.labelsize': 6.5,
    'axes.labelcolor': '#0b0b0b', 'axes.unicode_minus': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
})
INK, INK2, MUTED, BASE, GRID = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                                           '#e87ba4', '#008300', '#4a3aa7', '#e34948')
LTP, LTD = BLUE, RED


def tint(c, a):
    r = np.array(to_rgb(c))
    return tuple(1 - a * (1 - r))


def canvas(ax, w, h):
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off'); ax.set_aspect('equal')


def arrow(ax, p0, p1, color=INK2, lw=0.8, style='-|>', rad=0.0, ms=6, ls='-', z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, ls=ls,
                                 connectionstyle='arc3,rad=%g' % rad, shrinkA=0, shrinkB=0, zorder=z))


def box(ax, x, y, w, h, fc='white', ec=INK2, lw=0.7, r=0.25, ls='-', z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(path), interpolation='lanczos')
    ax.axis('off')
    return ax


def letter(fig, x, y, L):
    fig.text(x, y, L, fontsize=10, fontweight='bold', color=INK, va='top')


def spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def save(fig, out):
    fig.savefig(out + '.png', dpi=600, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    fig.savefig(out + '.svg', facecolor='white')
