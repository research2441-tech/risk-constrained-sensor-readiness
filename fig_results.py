"""Result figures 4-7 (grayscale; legends placed outside the data region)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, sys, numpy as np, pandas as pd
import figstyle as FS
import matplotlib.pyplot as plt
from analysis import load, label, boot_ci

RES = _HERE; FIG = _HERE
ORDER = ['PDC', 'ENO', 'EAAS', 'UGS', 'LDP', 'RCSR']
NAME = {'PDC': 'PDC', 'ENO': 'ENO', 'EAAS': 'EAAS', 'UGS': 'UGS', 'LDP': 'LDP', 'RCSR': 'Proposed'}
STY = {'PDC': ('0.55', 's', '--'), 'ENO': ('0.35', '^', '-.'), 'EAAS': ('0.7', 'v', ':'), 'UGS': ('0.45', 'D', (0, (5, 1))),
       'LDP': ('0.2', 'P', (0, (3, 1, 1, 1))), 'RCSR': ('0.0', 'o', '-')}
KEY = ['start', 'irr']


def pareto():
    p = load('pareto'); m = load('main'); sub = set(map(tuple, p[KEY].drop_duplicates().values))
    m = m[[tuple(x) in sub for x in m[KEY].values]]
    m = m[m.method == 'RCSR'].assign(budget=1.0)
    p['budget'] = p.kw.apply(lambda s: json.loads(s).get('budget_scale', np.nan))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45))
    for meth in ORDER:
        d = pd.concat([p[p.method == 'RCSR'], m]) if meth == 'RCSR' else p[p.method == meth]
        g = d.groupby('budget' if meth == 'RCSR' else 'knob')
        E = g.energy_Wh_node_day.mean(); Rc = g.recall_char.mean(); V = g.reserve_violation.mean() * 100
        ci = g.recall_char.apply(lambda x: boot_ci(x, 2000)).values
        o = np.argsort(E.values)
        c, mk, ls = STY[meth]
        axs[0].plot(E.values[o], Rc.values[o], color=c, ls=ls, marker=mk, ms=3.6, lw=1.0, label=NAME[meth],
                    mfc='white' if meth != 'RCSR' else '0.0', mec=c, mew=0.8)
        lo = np.array([x[0] for x in ci])[o]; hi = np.array([x[1] for x in ci])[o]
        axs[1].plot(E.values[o], V.values[o], color=c, ls=ls, marker=mk, ms=3.6, lw=1.0,
                    mfc='white' if meth != 'RCSR' else '0.0', mec=c, mew=0.8)
    axs[0].set_xlabel('energy (Wh per node-day)'); axs[0].set_ylabel('characterisation recall')
    axs[1].set_xlabel('energy (Wh per node-day)'); axs[1].set_ylabel('reserve violation (% of node-time)')
    for k, ax in enumerate(axs):
        ax.text(-0.15, 1.03, '(%s)' % 'ab'[k], transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='bottom')
        ax.grid(True, color='0.9', lw=0.4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, fontsize=6.6, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.86, wspace=0.27)
    fig.savefig(f'{FIG}/fig4_pareto.png', dpi=400)


def robust():
    r = load('robust'); m = load('main')
    sub = set(map(tuple, r[KEY].drop_duplicates().values))
    m = m[[tuple(x) in sub for x in m[KEY].values]].assign(tag='nominal')
    d = pd.concat([m, r]); d['sc'] = d.tag.str.replace('rob_', '')
    scen = ['nominal', 'link', 'lowsun', 'cold', 'slowwarm', 'fastdecay', 'drift']
    names = ['nominal', 'link\nimpairment', 'low\nirradiance', 'cold\nshift', 'slower\nwarm-up', 'faster\ndecay', 'severe\ndrift']
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.45), sharex=True)
    x0 = np.arange(len(scen)); w = 0.12
    for k, meth in enumerate(ORDER):
        c, mk, ls = STY[meth]
        g = d[d.method == meth].groupby('sc')
        y1 = [g.recall_char.mean().get(s, np.nan) for s in scen]
        y2 = [g.contamination.mean().get(s, np.nan) * 100 for s in scen]
        xs = x0 + (k - 2.5) * w
        axs[0].plot(xs, y1, ls='none', marker=mk, ms=3.6, color=c, mfc='white' if meth != 'RCSR' else '0.0', mec=c, mew=0.8, label=NAME[meth])
        axs[1].plot(xs, y2, ls='none', marker=mk, ms=3.6, color=c, mfc='white' if meth != 'RCSR' else '0.0', mec=c, mew=0.8)
    axs[0].set_ylabel('characterisation recall'); axs[1].set_ylabel('contaminated M records (%)')
    for ax in axs:
        ax.grid(True, axis='y', color='0.9', lw=0.4)
        for xv in x0[:-1] + 0.5:
            ax.axvline(xv, color='0.85', lw=0.5)
    axs[1].set_xticks(x0); axs[1].set_xticklabels(names, fontsize=6.0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=6.4, bbox_to_anchor=(0.55, 1.0), columnspacing=1.2, handletextpad=0.3)
    for k, ax in enumerate(axs):
        ax.text(-0.17, 1.0, '(%s)' % 'ab'[k], transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='top')
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.885, hspace=0.12)
    fig.savefig(f'{FIG}/fig5_robust.png', dpi=400)


def audit():
    m = load('main'); a = load('ablation')
    a['lab'] = a.apply(label, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    data, labels = [], []
    for meth in ORDER:
        data.append(m[m.method == meth].exp_bias_naive.values * 100); labels.append(NAME[meth] + '\nall records')
    rc = m[m.method == 'RCSR']
    data.append(rc.exp_bias_audit_ht.values * 100); labels.append('Proposed\naudit Hájek')
    data.append(rc.exp_bias_model_assisted.values * 100); labels.append('Proposed\nmodel-assist.')
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color='0.0', lw=1.0),
                    flierprops=dict(marker='o', ms=2.2, mfc='white', mec='0.3'), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for k, b in enumerate(bp['boxes']):
        b.set_facecolor('0.85' if k < len(ORDER) else '0.45'); b.set_edgecolor('0.0'); b.set_linewidth(0.7)
    ax.axhline(0, color='0.0', lw=0.6, ls=':')
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_ylabel('daily exposure bias (%)')
    ax = axs[1]
    rows = [('Proposed, audits', rc, 'audit'), ('Proposed, controller\nrecords', rc, 'opportunistic'),
            ('No audits (ablation)', a[a.lab == 'RCSR:audit=False'], 'opportunistic'), ('LDP', m[m.method == 'LDP'], 'opportunistic'),
            ('ENO', m[m.method == 'ENO'], 'opportunistic'), ('PDC', m[m.method == 'PDC'], 'opportunistic')]
    y = np.arange(len(rows))[::-1]
    for k, (nm, d, mode) in enumerate(rows):
        det = d[f'drift_detect_frac_{mode}'].mean() * 100
        dl = d[f'drift_delay_days_{mode}'].mean()
        fa = d[f'drift_fa_per_node_week_{mode}'].mean()
        ax.barh(y[k], det, color='0.25' if k == 0 else '0.75', edgecolor='0.0', lw=0.6, height=0.6)
        ax.text(det + 2, y[k], f'{dl:.1f} d, FA {fa:.2f}', va='center', fontsize=5.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.0)
    ax.set_xlim(0, 135); ax.set_xlabel('drifting nodes detected (%)')
    for k, ax in enumerate(axs):
        ax.text(-0.12 if k == 0 else -0.42, 1.03, '(%s)' % 'ab'[k], transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='bottom')
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.9, wspace=0.55)
    fig.savefig(f'{FIG}/fig7_audit.png', dpi=400)


def sens():
    s = load('sens'); m = load('main')
    sub = set(map(tuple, s[KEY].drop_duplicates().values))
    m = m[np.array([tuple(x) in sub for x in m[KEY].values]) & (m.method == 'RCSR').values]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
    # audit rate
    A = s[s.tag == 'sens_audit'].copy(); A['rate'] = A.knob.astype(float)
    A = pd.concat([A, m.assign(rate=1 / 240)])
    g = A.groupby('rate')
    per_day = np.array(g.recall_char.mean().index) * 1440
    ax = axs[0]
    ax.plot(per_day, g.recall_char.mean().values, color='0.0', marker='o', ms=3.4, label='recall')
    ax.set_xlabel('audits per area-day'); ax.set_ylabel('characterisation recall')
    ax2 = ax.twinx()
    ax2.plot(per_day[1:], g.exp_rmse_model_assisted.mean().values[1:] * 100, color='0.5', marker='s', ms=3.4, ls='--', mfc='white', label='exposure RMSE')
    ax2.set_ylabel('model-assisted RMSE (%)', color='0.35')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2, fontsize=5.8, handlelength=2.2)
    # harvest quantile
    Q = s[s.tag == 'sens_qh'].copy(); Q['q'] = Q.knob.astype(float); Q = pd.concat([Q, m.assign(q=0.10)])
    g = Q.groupby('q'); ax = axs[1]
    ax.plot(g.recall_char.mean().index, g.recall_char.mean().values, color='0.0', marker='o', ms=3.4, label='recall')
    ax.set_xlabel('harvest quantile $\\alpha_{\\mathrm{h}}$'); ax.set_ylabel('characterisation recall')
    ax2 = ax.twinx(); ax2.plot(g.reserve_violation.mean().index, g.reserve_violation.mean().values * 100, color='0.5', marker='s', ms=3.4, ls='--', mfc='white', label='reserve violation')
    ax2.set_ylabel('reserve violation (%)', color='0.35'); ax2.set_ylim(0, 1.0); ax.set_ylim(0.65, 0.71)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2, fontsize=5.8, handlelength=2.2)
    # risk target / hazard quantile
    ax = axs[2]
    Z = s[s.tag == 'sens_qz'].copy(); Z['q'] = Z.knob.astype(float); Z = pd.concat([Z, m.assign(q=0.80)])
    g = Z.groupby('q')
    ax.plot(g.recall_char.mean().index, g.recall_char.mean().values, color='0.0', marker='o', ms=3.4, label='recall')
    ax.set_xlabel('hazard quantile $q$'); ax.set_ylabel('characterisation recall')
    ax2 = ax.twinx(); ax2.plot(g.energy_Wh_node_day.mean().index, g.energy_Wh_node_day.mean().values, color='0.5', marker='s', ms=3.4, ls='--', mfc='white', label='energy')
    ax2.set_ylabel('Wh per node-day', color='0.35'); ax2.set_ylim(1.70, 1.78); ax.set_ylim(0.65, 0.71)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2, fontsize=5.8, handlelength=2.2)
    for k, ax in enumerate(axs):
        ax.text(-0.3, 1.27, '(%s)' % 'abc'[k], transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='center')
    fig.subplots_adjust(left=0.075, right=0.915, bottom=0.2, top=0.8, wspace=0.85)
    fig.savefig(f'{FIG}/fig6_sens.png', dpi=400)


if __name__ == '__main__':
    for f in sys.argv[1:]:
        globals()[f](); print('done', f)
