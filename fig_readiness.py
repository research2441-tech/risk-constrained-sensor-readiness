"""Fig. 2: readiness state machine and conditioning model (plant parameters of the emulator)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np
import figstyle as FS
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C
from sim import tau_on, warmup_slots

OUT = sys.argv[1] if len(sys.argv) > 1 else (_HERE + '/fig2_readiness.png')
fig = plt.figure(figsize=(3.5, 4.55))
# ---------------- (a) state machine
ax = fig.add_axes([0.0, 0.70, 1.0, 0.30]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off')
states = [('S', 'sleep', 10), ('W', 'warming', 37), ('R', 'ready', 64), ('A', 'acquire', 90)]
for k, (s, name, x) in enumerate(states):
    ax.add_patch(FancyBboxPatch((x - 7.5, 9), 15, 10, boxstyle='round,pad=0,rounding_size=2.5',
                                fc='0.88' if s in ('R', 'A') else 'white', ec='black', lw=0.8))
    ax.text(x, 15.6, s, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x, 11.6, name, ha='center', va='center', fontsize=6.3)


def arr(x0, y0, x1, y1, rad=0.0, dashed=False):
    a = FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>,head_length=3.2,head_width=1.6',
                        color='black', lw=0.8, ls=(0, (3, 2)) if dashed else '-', shrinkA=0, shrinkB=0)
    ax.add_patch(a)


arr(17.5, 14, 29.5, 14); ax.text(23.5, 15.0, '$u_i{=}1$', ha='center', va='bottom', fontsize=6.0)
arr(44.5, 14, 56.5, 14); ax.text(50.5, 15.0, '$\\theta_i{\\geq}\\theta^{\\star}$', ha='center', va='bottom', fontsize=6.0)
arr(71.5, 14, 82.5, 14); ax.text(77, 15.0, 'exceed', ha='center', va='bottom', fontsize=6.0)
arr(90, 9, 64, 9, rad=-0.45); ax.text(77, 1.6, 'event ends', ha='center', va='center', fontsize=6.0)
arr(64, 19, 10, 19, rad=0.28, dashed=True); ax.text(37, 27.8, '$u_i{=}0$: conditioning decays with $\\tau_{\\mathrm{off}}$', ha='center', va='center', fontsize=6.0)
arr(37, 9, 10, 9, rad=-0.35, dashed=True)
ax.text(1, 27.5, '(a)', fontsize=7.5, fontweight='bold', va='center')

# ---------------- (b) conditioning trajectories
ax1 = fig.add_axes([0.14, 0.39, 0.83, 0.23])
t = np.arange(0, 12.01, 0.05)
for k, T in enumerate([-10, 5, 20, 35]):
    th = 1 - np.exp(-t * 60 / tau_on(T))
    ax1.plot(t, th, color=FS.GRAYS[k], ls=FS.LINES[k], label=f'cold start, {T} °C')
th0 = 0.95 * np.exp(-600 / C.TAU_OFF)
ax1.plot(t, 1 - (1 - th0) * np.exp(-t * 60 / tau_on(20)), color='0.0', lw=1.6, ls=':', label='restart after 10 min off, 20 °C')
ax1.axhline(C.THETA_STAR, color='0.4', lw=0.6)
ax1.text(11.9, C.THETA_STAR - 0.06, '$\\theta^{\\star}$', ha='right', va='top', fontsize=7)
ax1.axvline(2, color='0.4', lw=0.6, ls=(0, (1, 1.5)))
ax1.text(2.15, 0.06, 'nominal 2-min\nwarm-up', fontsize=5.8, va='bottom')
ax1.set_xlim(0, 12); ax1.set_ylim(0, 1.04)
ax1.set_xlabel('time since heater on (min)'); ax1.set_ylabel('conditioning $\\theta$')
ax1.legend(loc='lower right', fontsize=5.6, handlelength=2.6, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.08))
ax1.text(-0.16, 1.02, '(b)', transform=ax1.transAxes, fontsize=7.5, fontweight='bold', va='bottom')

# ---------------- (c) warm-up time vs temperature and off duration
ax2 = fig.add_axes([0.14, 0.085, 0.83, 0.205])
Ts = np.linspace(-15, 40, 111)
for k, off in enumerate([5, 15, 30, 60, 1e9]):
    th0 = C.THETA_STAR * np.exp(-off * 60 / C.TAU_OFF)
    w = tau_on(Ts) * np.log((1 - th0) / (1 - C.THETA_STAR)) / 60.0
    ax2.plot(Ts, w, color=FS.GRAYS[min(k, 4)] if k < 4 else '0.0', ls=FS.LINES[k % 6],
             lw=1.5 if off > 1e8 else 1.0, label='cold' if off > 1e8 else f'off {off} min')
ax2.axhline(2, color='0.4', lw=0.6, ls=(0, (1, 1.5)))
ax2.set_xlim(-15, 40); ax2.set_ylim(0, 11)
ax2.set_xlabel('ambient temperature (°C)'); ax2.set_ylabel('warm-up time (min)')
ax2.legend(loc='upper right', ncol=3, fontsize=5.6, handlelength=2.2, columnspacing=0.9, borderaxespad=0.2)
ax2.text(-0.16, 1.02, '(c)', transform=ax2.transAxes, fontsize=7.5, fontweight='bold', va='bottom')
fig.savefig(OUT, dpi=400)
print('saved')
