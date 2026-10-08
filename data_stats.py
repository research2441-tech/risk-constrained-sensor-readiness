import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, numpy as np, pandas as pd
import episode as E, runner as R, config as C
rows = []
for ep in R.test_episodes():
    e = E.build(ep['start'], days=14, irr_year=ep['irr'], seed=ep['seed'])
    hv = e['harvest'].sum(0) * 60 / 3600 / 14
    ev = e['events']
    rows.append(dict(start=ep['start'], irr=ep['irr'], season=pd.Timestamp(ep['start']).month,
                     events_area_day=len(ev) / 12 / 14, short_frac=float((ev[:, 2] <= 5).mean()), dur_med=float(np.median(ev[:, 2])),
                     harvest_Wh=float(hv.mean()), harvest_min=float(hv.min()), temp_min=float(e['temp'].min()), temp_max=float(e['temp'].max()),
                     temp_mean=float(e['temp'].mean()), co_mean=float(e['co_bg'].mean()), drift_nodes=int(np.isfinite(e['drift_on']).sum()),
                     bad_frac=float((e['per'] > 0.5).mean()), kc_mean=float(e['kc_area'][e['cs'] > 50].mean()),
                     night_share=float(((e['hour'][ev[:, 0].astype(int)] >= 18)).mean())))
d = pd.DataFrame(rows)
d.to_csv((_HERE + '/data_stats.csv'), index=False)
print(d.groupby('season').mean(numeric_only=True).round(3).to_string())
print(d.mean(numeric_only=True).round(3).to_string())
m = pd.read_csv((_HERE + '/beijing_missing.csv'), index_col=0)
print('missing CO mean %.3f max %.3f; TEMP %.4f; WSPM %.4f' % (m['CO'].mean(), m['CO'].max(), m['TEMP'].mean(), m['WSPM'].mean()))
