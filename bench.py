"""Timing of the gateway planner as a function of horizon and grid (single core, warm caches)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, time, numpy as np
import config as C, episode as E, controllers as K, runner as R
from sim import Sim

ep = E.build('2014-01-08', days=2, irr_year='y2017', seed=1000)
out = []
for H, dlt in [(30, 5), (60, 5), (120, 5), (60, 2), (60, 10)]:
    C.H = H
    ctl = K.RCSR(R.GLM)
    s = Sim(ep, ctl, audit=True)
    ctl.init(s)
    st = np.arange(0, H, dlt); en = np.arange(dlt, H + 1, dlt)
    S_, E_ = np.meshgrid(st, en, indexing='ij'); m = E_ > S_
    ctl._SE = (S_[m], E_[m])
    for t in range(0, 120):           # warm up forecaster state
        ctl.decide(s, t)
    lam = ctl.hz_forecast(s, 200, H)
    temps = ep['temp'][200]
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        ctl._plan_all(s, 200, lam, temps)
    tp = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(20):
        ctl._strategic(s, 200)
    ts = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    for _ in range(50):
        ctl.hz_forecast(s, 200, H)
    th = (time.perf_counter() - t0) / 50 * 1000
    nplans = len(ctl._SE[0]) + 1
    mem = 12 * nplans * H * (8 + 1 + 1 + 8 + 8) / 1e6
    out.append(dict(H=H, delta=dlt, plans=nplans, tactical_ms=tp, strategic_ms=ts, hazard_ms=th, tensor_MB=mem))
    print(out[-1], flush=True)
C.H = 60
json.dump(out, open((_HERE + '/bench.json'), 'w'), indent=1)
