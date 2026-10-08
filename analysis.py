"""Statistics for all result tables: bootstrap CIs, paired Wilcoxon with Holm correction, rank-biserial r,
Friedman test."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

RES = _HERE
rng = np.random.default_rng(0)


def load(name):
    return pd.DataFrame([json.loads(l) for l in open(f'{RES}/{name}.jsonl')])


def label(r):
    kw = json.loads(r['kw'])
    if r['method'] != 'RCSR':
        return r['method']
    if not kw:
        return 'RCSR'
    return 'RCSR:' + ','.join(f'{k}={v}' for k, v in sorted(kw.items()))


def boot_ci(x, B=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def rank_biserial(d):
    d = np.asarray(d, float); d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


def compare(df, metrics, ref='RCSR', key=('start', 'irr')):
    df = df.copy(); df['lab'] = df.apply(label, axis=1)
    labs = [l for l in df.lab.unique()]
    out = {'summary': {}, 'tests': {}}
    for m in metrics:
        piv = df.pivot_table(index=list(key), columns='lab', values=m)
        out['summary'][m] = {l: dict(mean=float(piv[l].mean()), ci=boot_ci(piv[l]), sd=float(piv[l].std())) for l in piv.columns}
        others = [l for l in piv.columns if l != ref]
        ps, rows = [], []
        for l in others:
            d = (piv[ref] - piv[l]).dropna()
            if len(d) < 5 or np.allclose(d, 0):
                p = 1.0
            else:
                p = float(wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)
            ps.append(p); rows.append(dict(other=l, diff=float(d.mean()), diff_ci=boot_ci(d), r_rb=rank_biserial(d), p=p, n=int(len(d))))
        adj = holm(ps) if ps else []
        for r_, a in zip(rows, adj):
            r_['p_holm'] = float(a)
        out['tests'][m] = rows
        try:
            cols = piv.dropna()
            out['tests'][m + '__friedman'] = float(friedmanchisquare(*[cols[c] for c in cols.columns]).pvalue) if cols.shape[1] >= 3 else None
        except Exception:
            out['tests'][m + '__friedman'] = None
    return out


MAIN_METRICS = ['recall_char', 'recall_char_short', 'recall_detect', 'recall_char_designated', 'recall_char_night', 'delay_mean_min',
                'availability', 'energy_Wh_node_day', 'reserve_violation', 'brownout_node_h_day', 'false_alarms_per_area_day',
                'contamination', 'exp_bias_naive', 'exp_rmse_naive', 'valid_records_per_area_day', 'final_soc']

if __name__ == '__main__':
    name = sys.argv[1]
    df = load(name)
    if name == 'main':
        res = compare(df, MAIN_METRICS + [c for c in df.columns if c.startswith(('drift_', 'exp_', 'heat_share', 'infeasible', 'solve'))])
        # per-season breakdown
        df['season'] = pd.to_datetime(df.start).dt.month
        df['lab'] = df.apply(label, axis=1)
        res['season'] = {m: df.groupby(['lab', 'season'])[m].mean().unstack().to_dict() for m in ['recall_char', 'energy_Wh_node_day', 'reserve_violation', 'contamination']}
        res['year'] = df.groupby(['lab', pd.to_datetime(df.start).dt.year])['recall_char'].mean().unstack().to_dict()
        res['irr'] = df.groupby(['lab', 'irr'])['recall_char'].mean().unstack().to_dict()
    else:
        res = compare(df, MAIN_METRICS + [c for c in df.columns if c.startswith(('drift_', 'exp_', 'heat_share', 'infeasible', 'solve'))])
    json.dump(res, open(f'{RES}/stats_{name}.json', 'w'), indent=1, default=float)
    s = res['summary']
    for m in ['recall_char', 'recall_char_short', 'energy_Wh_node_day', 'reserve_violation', 'contamination', 'false_alarms_per_area_day', 'availability', 'exp_bias_naive']:
        print(m, {k: round(v['mean'], 3) for k, v in s[m].items()})
    for m in ['recall_char', 'reserve_violation', 'contamination']:
        print('TEST', m, [(r['other'], round(r['diff'], 3), round(r['r_rb'], 2), '%.2g' % r['p_holm']) for r in res['tests'][m]])
