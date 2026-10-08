"""Fig. 1: system architecture (grayscale, no overlapping elements, orthogonal routing without crossings)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys
import figstyle  # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else (_HERE + '/fig1_architecture.png')
fig = plt.figure(figsize=(7.0, 3.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-3, 47); ax.axis('off')


def box(x0, y0, x1, y1, title, body='', fc='white', lw=0.8, ts=7.2, bs=6.4, ls='-'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.0,rounding_size=0.8',
                                fc=fc, ec='black', lw=lw, ls=ls))
    if body:
        ax.text((x0 + x1) / 2, y1 - 1.4, title, ha='center', va='top', fontsize=ts, fontweight='bold')
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 1.3, body, ha='center', va='center', fontsize=bs, linespacing=1.25)
    else:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, title, ha='center', va='center', fontsize=ts, fontweight='bold')


def seg(pts, dashed=False, head=True, lw=0.9):
    xs, ys = zip(*pts)
    ls = (0, (3.2, 2.0)) if dashed else '-'
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color='black', lw=lw, ls=ls, solid_capstyle='butt')
    if head:
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        L = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
        f = min(1.0, 1.3 / L)
        ax.annotate('', xy=pts[-1], xytext=(xb - (xb - xa) * f, yb - (yb - ya) * f),
                    arrowprops=dict(arrowstyle='-|>,head_length=0.45,head_width=0.22', color='black', lw=lw,
                                    shrinkA=0, shrinkB=0))


def lab(x, y, s, ha='left', va='bottom', size=6.2, rot=0):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, rotation=rot,
            bbox=dict(fc='white', ec='none', pad=0.6))


# ---------------- groups
ax.add_patch(FancyBboxPatch((1, 2), 23, 43.5, boxstyle='round,pad=0,rounding_size=1.2', fc='0.93', ec='0.3', lw=0.7))
ax.text(12.5, 44.6, 'Solar-powered node $i$ (36 nodes,\n12 areas, 3 nodes per area)', ha='center', va='top', fontsize=6.6, fontweight='bold')
ax.add_patch(FancyBboxPatch((28.5, 2), 70, 43.5, boxstyle='round,pad=0,rounding_size=1.2', fc='0.93', ec='0.3', lw=0.7))
ax.text(63.5, 44.6, 'Gateway: risk-constrained readiness controller', ha='center', va='top', fontsize=7.2, fontweight='bold')

# ---------------- node internals
box(3, 27.5, 22, 36, 'PV panel + Li-ion cell', 'harvest $P^{\\mathrm{h}}_i(t)$, battery $B_i(t)$')
box(3, 15.5, 22, 25.5, 'Heated MOX sensor', 'conditioning state $\\theta_i(t)$\nS $\\rightarrow$ W $\\rightarrow$ R $\\rightarrow$ A')
box(3, 4, 22, 13.5, 'MCU + LPWAN radio', 'readings labelled M or U,\nsettling-time reports')

# ---------------- gateway tier 1 (estimators)
box(31, 26.5, 46, 35.5, 'Harvest forecaster', 'one-sided conformal\nlower bound $\\hat{P}^{-}_i$')
box(49, 26.5, 64, 35.5, 'Hazard model', 'GLM + audit Gamma\nrecalibration $\\lambda^{+}_a$')
box(67, 26.5, 82, 35.5, 'Readiness learner', 'Bayesian $\\tau^{+}_{\\mathrm{on}}(T)$,\nwarm-up $w^{+}(\\theta,T)$')
# tier 2 (decision)
box(31, 15, 54, 22.5, 'Strategic allocator', 'knapsack to next harvest\n$\\rightarrow\\nu_a,\\ \\bar{E}_a$')
box(58, 15, 89, 22.5, 'Tactical receding-horizon planner', 'min $R_a(\\pi)+\\nu_a E(\\pi)$ s.t. $R_a\\leq\\epsilon_a\\Lambda_a$')
# tier 3
box(31, 3.5, 54, 10.5, 'Independent audit scheduler', 'pre-committed Bernoulli($\\pi_{\\mathrm{A}}$)\nno input from controller', fc='0.98')
box(58, 3.5, 89, 10.5, 'Data product and audit analytics', 'M/P/U labels, HT and model-assisted\nexposure, CUSUM drift alarms')

# ---------------- uplink bus (node -> gateway)
seg([(24, 38.5), (26.3, 38.5), (26.3, 39.6), (95, 39.6), (95, 7), (89, 7)])
lab(28.5, 39.9, 'uplink: readings, battery, settling times', size=6.0)
seg([(38.5, 39.6), (38.5, 35.5)])
seg([(56.5, 39.6), (56.5, 35.5)], dashed=True)
lab(57.2, 36.3, 'audit outcomes', size=5.6)
seg([(74.5, 39.6), (74.5, 35.5)], dashed=True)
lab(75.2, 36.3, 'settling times', size=5.6)
lab(95.6, 22, 'readings', size=5.8, rot=90, ha='left', va='center')

# ---------------- estimator -> decision
seg([(38.5, 26.5), (38.5, 22.5)])
lab(39.1, 23.6, '$\\hat{P}^{-}_i$', size=6.2)
seg([(51.5, 26.5), (51.5, 22.5)])
lab(52.1, 23.6, '$\\lambda^{+}_a$', size=6.2)
seg([(61.5, 26.5), (61.5, 22.5)])
lab(62.1, 23.6, '$\\lambda^{+}_a$', size=6.2)
seg([(74.5, 26.5), (74.5, 22.5)])
lab(75.1, 23.6, '$w^{+}$', size=6.2)
seg([(54, 18.75), (58, 18.75)])

# ---------------- commands to nodes
seg([(66, 15), (66, 12.75), (24, 12.75)])
lab(27.0, 13.1, 'heater commands $u_i(t)$', size=6.0)
seg([(31, 7), (24, 7)])
lab(24.4, 7.4, 'audit', size=5.8)

# ---------------- legend
ax.plot([3, 8], [-1.3, -1.3], color='black', lw=0.9)
ax.text(9, -1.3, 'data and control flow', va='center', fontsize=6.2)
ax.plot([33, 38], [-1.3, -1.3], color='black', lw=0.9, ls=(0, (3.2, 2.0)))
ax.text(39, -1.3, 'learning and recalibration feedback', va='center', fontsize=6.2)
ax.text(70, -1.3, 'S sleep, W warming, R ready, A acquisition', va='center', fontsize=6.2)
fig.savefig(OUT, dpi=400)
print('saved', OUT)
