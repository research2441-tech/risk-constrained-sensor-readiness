"""Pick, for each baseline, the knob with the best validation recall among settings whose mean energy does not exceed
the proposed controller's validation energy by more than 3 %."""
import json, pandas as pd, runner as R
d = pd.DataFrame([json.loads(l) for l in open(R.RES + '/val.jsonl')])
g = d.groupby(['method', 'knob'], dropna=False)[['energy_Wh_node_day', 'recall_char', 'reserve_violation']].mean()
e_ref = g.loc['RCSR'].energy_Wh_node_day.mean()
sel = {}
for m in ['PDC', 'ENO', 'EAAS', 'UGS', 'LDP']:
    gm = g.loc[m]
    ok = gm[gm.energy_Wh_node_day <= 1.03 * e_ref]
    k = ok.recall_char.idxmax() if len(ok) else gm.energy_Wh_node_day.idxmax()
    sel[m] = float(k)
print('reference energy', round(e_ref, 3), sel)
print(g.round(3).to_string())
json.dump(dict(e_ref=e_ref, knobs=sel), open(R.RES + '/knobs.json', 'w'), indent=1)
