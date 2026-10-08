"""Optimality gap of the single-segment plan family against exhaustive enumeration of all heater sequences
(H = 16 slots, 65 536 sequences) under the planner's own readiness, risk and energy model."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, sys, numpy as np
import config as C, episode as E
from hazard import HazardModel

rng = np.random.default_rng(7)
Hh = 16
Fd = E.duration_cdf_slots(); F0 = Fd[C.M_CHAR]
glm = json.load(open((_HERE + '/glm.json')))
hz = HazardModel(glm)
tau_ref, beta = 85.0, 0.025


def tau(T):
    return tau_ref * np.exp(beta * (C.T_REF - T))


def wslots(th, T):
    th = np.minimum(th, C.THETA_STAR)
    return np.ceil(tau(T) * np.log((1 - th) / (1 - C.THETA_STAR)) / C.DT - 1e-9).astype(int)


def evaluate(heat, th0, T, lam, cmd_now):
    """heat (P,H) bool -> risk R (P,), energy E (P,)"""
    P = heat.shape[0]
    th = np.full(P, th0); ready = np.zeros((P, Hh), bool)
    a_on = np.exp(-C.DT / tau(T)); a_off = np.exp(-C.DT / C.TAU_OFF)
    for k in range(Hh):
        ready[:, k] = heat[:, k] & (th >= C.THETA_STAR)
        th = np.where(heat[:, k], 1 - (1 - th) * a_on, th * a_off)
    kk = np.arange(Hh)
    big = 10 ** 6
    nxt = np.where(ready, kk[None, :], big)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    w_end = wslots(th, T)
    g = np.where(nxt < big, nxt - kk[None, :], (Hh - kk[None, :]) + w_end[:, None])
    g = np.clip(g, 0, len(Fd) - C.M_CHAR - 1)
    R = ((Fd[g + C.M_CHAR] - F0) * lam[None, :]).sum(1)
    En = heat.sum(1) * C.P_HEATER * C.DT
    return R, En


def select(R, En, nu, cap, target):
    feas = En <= cap + 1e-9
    sat = feas & (R <= target)
    J = R + nu * En
    if sat.any():
        i = np.argmin(np.where(sat, J, np.inf))
    else:
        i = np.argmin(np.where(feas, J, np.inf))
    return J[i], sat.any()


allseq = ((np.arange(2 ** Hh)[:, None] >> np.arange(Hh)[None, :]) & 1).astype(bool)
segs = [np.zeros(Hh, bool)]
for s in range(Hh):
    for e in range(s + 1, Hh + 1):
        m = np.zeros(Hh, bool); m[s:e] = True; segs.append(m)
segs = np.array(segs)
step = [np.zeros(Hh, bool)]
for s in range(0, Hh, 4):
    for e in range(s + 4, Hh + 1, 4):
        m = np.zeros(Hh, bool); m[s:e] = True; step.append(m)
step = np.array(step)
gaps1, gaps4, same = [], [], 0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for n in range(N):
    th0 = rng.choice([0.0, rng.uniform(0, 1), rng.uniform(0.9, 1.0)])
    T = rng.uniform(-10, 35)
    hour = rng.uniform(0, 24)
    lam = hz.predict((hour + np.arange(Hh) / 60) % 24, np.full((Hh, C.N_AREAS), rng.uniform(0.3, 4)),
                     np.full((Hh, C.N_AREAS), rng.uniform(400, 3000)))[:, rng.integers(C.N_AREAS)] * rng.uniform(0.5, 3)
    nu = rng.choice([0.0, 10 ** rng.uniform(-6, -3)])
    cap = rng.choice([np.inf, rng.uniform(0, Hh) * C.P_HEATER * C.DT])
    target = C.EPS_RISK * lam.sum()
    Rx, Ex = evaluate(allseq, th0, T, lam, False)
    Jx, sx = select(Rx, Ex, nu, cap, target)
    R1, E1 = evaluate(segs, th0, T, lam, False)
    J1, s1 = select(R1, E1, nu, cap, target)
    R4, E4 = evaluate(step, th0, T, lam, False)
    J4, s4 = select(R4, E4, nu, cap, target)
    sc = max(Jx, 1e-9)
    gaps1.append((J1 - Jx) / sc if s1 == sx else np.nan); gaps4.append((J4 - Jx) / sc if s4 == sx else np.nan)
    same += (s1 == sx)
g1, g4 = np.array(gaps1), np.array(gaps4)
res = dict(n=N, same_feasibility_frac=same / N,
           seg1_mean_gap=float(np.nanmean(g1)), seg1_p95_gap=float(np.nanpercentile(g1, 95)), seg1_zero_frac=float(np.nanmean(g1 < 1e-9)),
           seg4_mean_gap=float(np.nanmean(g4)), seg4_p95_gap=float(np.nanpercentile(g4, 95)), seg4_zero_frac=float(np.nanmean(g4 < 1e-9)),
           n_exhaustive=int(2 ** Hh), n_seg1=int(len(segs)), n_seg4=int(len(step)))
print(res)
json.dump(res, open((_HERE + '/optgap.json'), 'w'), indent=1)
