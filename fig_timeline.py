"""Fig. 3: one winter day in one area under the proposed controller and the Lyapunov baseline."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys, json, numpy as np
import figstyle as FS
import matplotlib.pyplot as plt
import episode as E, controllers as K, config as C, runner as R
from sim import Sim
glm = R.GLM
ep_spec = R.test_episodes()[0]          # 2014-01-08, irradiance 2017
ep = E.build(ep_spec['start'], days=14, irr_year=ep_spec['irr'], seed=ep_spec['seed'])
kn = json.load(open(R.RES + '/knobs.json'))['knobs']
s1 = Sim(ep, K.RCSR(glm), audit=True).run()
s2 = Sim(ep, K.Lyapunov(glm, V=kn['LDP']), audit=False).run()
thr_all = (1 + C.DETECT_KAPPA) * ep['bg_pred']


def char_flags(sim, a, day):
    D = np.array(sim.deliv, dtype=float)
    lab = D[(D[:, 6] > 0) & ((D[:, 1] // 3) == a)]
    th = thr_all[:, a]
    out = []
    for (t0, aa, dur, M) in ep['events']:
        if int(aa) != a or not (day * 1440 <= t0 < (day + 1) * 1440):
            continue
        m = (lab[:, 0] >= t0) & (lab[:, 0] < t0 + dur) & (lab[:, 3] > 0) & (lab[:, 2] > th[lab[:, 0].astype(int)])
        out.append((t0 - day * 1440, dur, len(np.unique(lab[m, 0])) >= C.M_CHAR))
    return out


# representative area-day: median recall difference among area-days with at least 10 events (days 3-12)
cands = []
for aa in range(12):
    for dd in range(3, 13):
        f1, f2 = char_flags(s1, aa, dd), char_flags(s2, aa, dd)
        if len(f1) >= 10:
            cands.append((np.mean([o for *_, o in f1]) - np.mean([o for *_, o in f2]), aa, dd))
cands.sort()
diff, a, day = cands[len(cands) // 2]
print('selected area', a, 'day', day, 'diff', round(diff, 3), 'of', len(cands), 'candidates; mean diff', round(np.mean([c[0] for c in cands]), 3))
sl = slice(day * 1440, (day + 1) * 1440)
hrs = np.arange(1440) / 60.0
thr = thr_all[:, a]


def detected(sim):
    D = np.array(sim.deliv, dtype=float)
    lab = D[(D[:, 6] > 0) & ((D[:, 1] // 3) == a)]
    out = []
    for (t0, aa, dur, M) in ep['events']:
        if int(aa) != a or not (day * 1440 <= t0 < (day + 1) * 1440):
            continue
        m = (lab[:, 0] >= t0) & (lab[:, 0] < t0 + dur) & (lab[:, 3] > 0) & (lab[:, 2] > thr[lab[:, 0].astype(int)])
        out.append((t0 - day * 1440, dur, len(np.unique(lab[m, 0])) >= C.M_CHAR))
    return out


fig, axs = plt.subplots(5, 1, figsize=(3.5, 4.3), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 0.8, 0.8, 0.8, 1.1], hspace=0.28))
ax = axs[0]
ax.plot(hrs, ep['lam_true'][sl, a] * 60, color='0.55', lw=0.9, label='true hazard')
ax.plot(hrs, s1.ctl.lam_log[sl, a] * 60, color='0.0', lw=0.9, ls='--', label='planning hazard $\\lambda^{+}_a$')
ax.set_ylabel('events h$^{-1}$')
ax.set_ylim(0, max(ep['lam_true'][sl, a].max(), s1.ctl.lam_log[sl, a].max()) * 60 * 1.45)
ax.legend(loc='upper left', ncol=2, fontsize=5.8, handlelength=2.0, columnspacing=1.0, borderaxespad=0.1)
for k, (sim, nm) in enumerate([(s1, 'proposed'), (s2, 'LDP')]):
    ax = axs[1 + k]
    for j in range(3):
        i = a * 3 + j
        h = sim.log_heat[sl, i]; r = sim.log_ready[sl, i]
        ax.fill_between(hrs, j + 0.15, j + 0.85, where=h & ~r, color='0.72', step='post', lw=0)
        ax.fill_between(hrs, j + 0.15, j + 0.85, where=r, color='0.15', step='post', lw=0)
    if sim.audit:
        ta = np.nonzero(sim.audit_fire[sl, a])[0] / 60.0
        ax.plot(ta, np.full(len(ta), 3.25), marker='v', ls='none', color='0.0', ms=3.0)
    ax.set_ylim(0, 3.55); ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(['n1', 'n2', 'n3'], fontsize=6)
    ax.set_ylabel(nm, fontsize=7)
ax = axs[3]
for k, (sim, nm, mk) in enumerate([(s1, 'proposed', 'o'), (s2, 'LDP', 's')]):
    for (t0, dur, ok) in detected(sim):
        y = 1.0 - 0.5 * k
        ax.plot([t0 / 60, (t0 + dur) / 60], [y, y], color='0.0', lw=1.0)
        ax.plot(t0 / 60, y, marker=mk, ms=3.6, mfc='0.0' if ok else 'white', mec='0.0', mew=0.7, ls='none')
ax.set_ylim(0.2, 1.3); ax.set_yticks([1.0, 0.5]); ax.set_yticklabels(['proposed', 'LDP'], fontsize=6)
ax.set_ylabel('events', fontsize=7)
ax = axs[4]
for k, (sim, nm, ls) in enumerate([(s1, 'proposed', '-'), (s2, 'LDP', '--')]):
    ax.plot(hrs, sim.log_B[sl, a * 3:(a + 1) * 3].mean(1) / C.B_MAX, color='0.0' if k == 0 else '0.5', ls=ls, lw=1.0, label=nm)
ax.axhline(C.B_RES_FRAC, color='0.3', lw=0.6, ls=':')
ax.text(23.8, C.B_RES_FRAC + 0.01, 'reserve', ha='right', va='bottom', fontsize=6)
ax.set_ylabel('mean SoC'); ax.set_xlabel('local time (h)'); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
ax.legend(loc='upper left', fontsize=6, ncol=2, borderaxespad=0.1)
for k, ax in enumerate(axs):
    ax.text(-0.17, 1.0, '(%s)' % 'abcde'[k], transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='top')
fig.subplots_adjust(left=0.17, right=0.98, top=0.985, bottom=0.095)
fig.savefig((_HERE + '/fig3_timeline.png'), dpi=400)
st = {nm: dict(rec=float(np.mean([o for _, _, o in detected(sim)])) if detected(sim) else None, n=len(detected(sim)))
      for sim, nm in [(s1, 'proposed'), (s2, 'LDP')]}
print(st)
