"""Experiment runner: episode definitions, method factory, parallel execution with JSON-lines output."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, os, sys, time, itertools
import numpy as np
from multiprocessing import Pool
import config as C
import episode as E
import controllers as K
import metrics as M
from sim import Sim

RES = os.environ.get('RCSR_OUT', _HERE)
GLM = json.load(open(os.path.join(RES, 'glm.json')))

TEST_WINDOWS = [f'{y}-{m}-08' for y in (2014, 2015, 2016) for m in ('01', '04', '07', '10')]
VAL_WINDOWS = ['2013-04-20', '2013-07-20', '2013-10-20', '2013-12-20']
IRR = ['y2017', 'y2023']


def test_episodes(n=None):
    eps = [dict(start=s, irr=i, seed=1000 + 17 * k + (0 if i == 'y2017' else 7))
           for k, s in enumerate(TEST_WINDOWS) for i in IRR]
    return eps if n is None else eps[:n]


def subset12():
    """one irradiance year per window, alternating, for the costlier sweeps."""
    return [e for k, e in enumerate(test_episodes()) if (k // 2 + k) % 2 == 0]


def val_episodes():
    return [dict(start=s, irr='y2023', seed=500 + k) for k, s in enumerate(VAL_WINDOWS)]


def make(method, knob=None, **kw):
    if method == 'RCSR':
        return K.RCSR(GLM, **kw), kw.get('audit', True)
    if method == 'PDC':
        return K.Periodic(knob or 15), False
    if method == 'ENO':
        return K.EnergyNeutral(knob or 1.0), False
    if method == 'EAAS':
        return K.AdaptiveSampling(knob or 6), False
    if method == 'UGS':
        return K.UncertaintyGuided(GLM, p_th=knob or 0.25), False
    if method == 'LDP':
        return K.Lyapunov(GLM, V=knob or 3e3), False
    raise ValueError(method)


def run_one(job):
    ep_spec, method, knob, kw, scen, tag = job
    t0 = time.time()
    scen = dict(scen or {})
    plant = scen.pop('plant', None)
    audit_rate = scen.pop('audit_rate', C.AUDIT_RATE)
    ep = E.build(ep_spec['start'], days=14, irr_year=ep_spec['irr'], seed=ep_spec['seed'], **scen)
    ctl, aud = make(method, knob, **(kw or {}))
    s = Sim(ep, ctl, plant=plant, audit=aud, audit_rate=audit_rate).run()
    r = M.evaluate(s)
    r.update(dict(method=method, knob=knob, kw=json.dumps(kw or {}), tag=tag, start=ep_spec['start'], irr=ep_spec['irr'],
                  seed=ep_spec['seed'], runtime_s=time.time() - t0))
    return r


def run_jobs(jobs, out, procs=2):
    done = set()
    if os.path.exists(out):
        for l in open(out):
            d = json.loads(l)
            done.add((d['tag'], d['method'], str(d['knob']), d['kw'], d['start'], d['irr']))
    todo = [j for j in jobs if (j[5], j[1], str(j[2]), json.dumps(j[3] or {}), j[0]['start'], j[0]['irr']) not in done]
    print(f'{len(todo)} jobs to run ({len(jobs) - len(todo)} cached) -> {out}', flush=True)
    with Pool(procs) as p, open(out, 'a') as f:
        for k, r in enumerate(p.imap_unordered(run_one, todo)):
            f.write(json.dumps(r) + '\n'); f.flush()
            if k % 10 == 0:
                print(k, r['tag'], r['method'], r['knob'], round(r['runtime_s'], 1), flush=True)
