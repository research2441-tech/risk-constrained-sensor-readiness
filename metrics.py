"""Episode-level evaluation metrics computed from emulator logs and ground truth."""
import numpy as np
import config as C
from sim import NA, NPA, NN


def _deliv_array(sim):
    if not sim.deliv:
        return np.zeros((0, 7))
    return np.array(sim.deliv, dtype=float)


DRIFT_TOL = 0.10   # smallest log-gain change of practical interest (~10 %)


def cusum_alarms(times, r, k=0.06, h=0.6):
    sp = sm = 0.0; out = []
    for t, x in zip(times, r):
        sp = max(0.0, sp + x - k); sm = max(0.0, sm - x - k)
        if sp > h or sm > h:
            out.append(t); sp = sm = 0.0
    return out


def drift_eval(sim, D, mode):
    """mode 'audit': only audit records; 'opportunistic': any co-delivered labelled-valid records."""
    ep = sim.ep
    rec = D[D[:, 6] > 0]
    thr = (1 + C.DETECT_KAPPA) * ep['bg_pred']
    if mode == 'audit':
        rec = rec[rec[:, 4] > 0]; gap = 25
    else:
        rec = rec[rec[:, 4] == 0]; gap = 2
    res = {i: ([], []) for i in range(NN)}
    for a in range(NA):
        ra = rec[(rec[:, 1] // NPA) == a]
        if len(ra) < 2:
            continue
        ra = ra[np.argsort(ra[:, 0])]
        cl = np.concatenate([[0], np.cumsum(np.diff(ra[:, 0]) > gap)])
        for c in np.unique(cl):
            g = ra[cl == c]
            nodes = np.unique(g[:, 1]).astype(int)
            if len(nodes) < NPA:
                continue
            if mode == 'audit' and (g[:, 2] > thr[g[:, 0].astype(int), a]).any():
                continue          # audit coincided with a transient: not a calibration opportunity
            ly = {n: np.log(max(np.mean(g[g[:, 1] == n, 2]), 1.0)) for n in nodes}
            med = np.median(list(ly.values()))
            for n in nodes:
                res[n][0].append(g[0, 0]); res[n][1].append(ly[n] - med)
    det, delays, fa, n_obs = 0, [], 0, []
    T = sim.T
    ndrift = 0
    t_ref = 3 * C.SLOTS_PER_DAY          # drift never starts before day 3: robust noise scale from that period
    early = np.concatenate([np.array(res[i][1])[np.array(res[i][0]) < t_ref] for i in range(NN) if res[i][0]] or [np.zeros(0)])
    sigma = 1.4826 * np.median(np.abs(early - np.median(early))) if len(early) >= 10 else 0.05
    sigma = max(sigma, 0.01)
    for i in range(NN):
        times, r = np.array(res[i][0]), np.array(res[i][1])
        n_obs.append(len(times))
        if len(times):
            base = np.median(r[times < t_ref]) if (times < t_ref).sum() >= 3 else 0.0
            al = cusum_alarms(times, (r - base) / sigma, k=max(0.5, DRIFT_TOL / 2 / sigma), h=5.0)
        else:
            al = []
        on = ep['drift_on'][i]
        if np.isfinite(on):
            ndrift += 1
            post = [x for x in al if x >= on]
            fa += len([x for x in al if x < on])
            if post:
                det += 1; delays.append((post[0] - on) / C.SLOTS_PER_DAY)
        else:
            fa += len(al)
    weeks = T / C.SLOTS_PER_DAY / 7.0
    return dict(drift_detect_frac=det / max(ndrift, 1), drift_delay_days=float(np.median(delays)) if delays else np.nan,
                drift_fa_per_node_week=fa / (NN * weeks), pair_obs_per_node_day=float(np.mean(n_obs)) / (T / C.SLOTS_PER_DAY),
                resid_sigma=float(sigma))


def evaluate(sim, extra=None):
    ep = sim.ep; T = sim.T; days = T / C.SLOTS_PER_DAY
    D = _deliv_array(sim)
    out = {}
    thr_area = (1 + C.DETECT_KAPPA) * ep['bg_pred']             # (T, A)
    # ---------------- events
    ev = ep['events']
    det_t, nchar = np.full(len(ev), np.nan), np.zeros(len(ev))
    if len(D):
        lab = D[D[:, 6] > 0]
        la_area = (lab[:, 1] // NPA).astype(int); la_t = lab[:, 0].astype(int)
        exc = lab[:, 2] > thr_area[la_t, la_area]
        good = exc & (lab[:, 3] > 0)
        for k, (t0, a, dur, M) in enumerate(ev):
            m = good & (la_area == int(a)) & (la_t >= t0) & (la_t < t0 + dur)
            if m.any():
                tt = np.unique(la_t[m])
                det_t[k] = tt[0] - t0
                nchar[k] = len(tt)
        # false alarms: labelled exceedances outside event windows
        inwin = np.zeros((T + 50, NA), bool)
        for (t0, a, dur, M) in ev:
            inwin[int(t0) - 1:int(t0 + dur) + 3, int(a)] = True
        fa_m = exc & ~inwin[la_t, la_area]
        fa_eps = 0
        for a in range(NA):
            ts = np.sort(la_t[fa_m & (la_area == a)])
            if len(ts):
                fa_eps += 1 + int((np.diff(ts) > 30).sum())
        out['false_alarms_per_area_day'] = fa_eps / (NA * days)
        out['contamination'] = float((lab[:, 3] == 0).mean()) if len(lab) else 0.0
        out['valid_records_per_area_day'] = float((lab[:, 3] > 0).sum()) / (NA * days)
    else:
        out.update(false_alarms_per_area_day=0.0, contamination=0.0, valid_records_per_area_day=0.0)
    detected = np.isfinite(det_t)
    out['n_events'] = len(ev)
    out['recall_detect'] = float(detected.mean())
    out['recall_char'] = float((nchar >= C.M_CHAR).mean())
    short = ev[:, 2] <= 5
    out['recall_char_short'] = float((nchar[short] >= C.M_CHAR).mean()) if short.any() else np.nan
    out['recall_char_designated'] = float((nchar[np.isin(ev[:, 1], C.DESIGNATED)] >= C.M_CHAR).mean())
    hr = ep['hour'][ev[:, 0].astype(int)]
    for nm, (lo, hi) in dict(night=(18, 24), early=(0, 6), day=(6, 18)).items():
        m = (hr >= lo) & (hr < hi)
        out[f'recall_char_{nm}'] = float((nchar[m] >= C.M_CHAR).mean()) if m.any() else np.nan
    out['delay_median_min'] = float(np.median(det_t[detected])) if detected.any() else np.nan
    out['delay_mean_min'] = float(np.mean(det_t[detected])) if detected.any() else np.nan
    # ---------------- readiness availability
    ready_area = sim.log_ready.reshape(T, NA, NPA).any(2)
    out['availability'] = float(ready_area.mean())
    out['availability_designated'] = float(ready_area[:, C.DESIGNATED].mean())
    # ---------------- energy & battery
    Etot = sim.E_heat + sim.E_tx + sim.E_samp
    out['energy_Wh_node_day'] = float(Etot.mean() / 3600 / days)
    out['heater_share'] = float(sim.E_heat.sum() / max(Etot.sum(), 1e-9))
    tot = sim.slots_audit + sim.slots_alert + sim.slots_ctl
    for nm, v in (('audit', sim.slots_audit), ('alert', sim.slots_alert), ('ctl', sim.slots_ctl)):
        out[f'heat_share_{nm}'] = float(v / max(tot, 1))
    out['reserve_violation'] = float((sim.log_B < C.B_RES_FRAC * C.B_MAX).mean())
    out['brownout_node_h_day'] = float(sim.brown.sum() / 60 / days / NN)
    out['final_soc'] = float(sim.log_B[-1].mean() / C.B_MAX)
    # ---------------- exposure (daily mean concentration per area)
    if len(D):
        lab = D[D[:, 6] > 0]
        la_area = (lab[:, 1] // NPA).astype(int); la_t = lab[:, 0].astype(int)
        truth_area = ep['conc'].reshape(T, NA, NPA).mean(2)
        errs = {'naive': [], 'audit_ht': [], 'model_assisted': [], 'model_only': []}
        for d in range(int(days)):
            sl = slice(d * 1440, (d + 1) * 1440)
            for a in range(NA):
                tru = truth_area[sl, a].mean()
                m = (la_area == a) & (la_t >= d * 1440) & (la_t < (d + 1) * 1440)
                if m.sum() > 0:
                    errs['naive'].append(lab[m, 5].mean() / tru - 1)
                ma = m & (lab[:, 4] > 0)
                errs['model_only'].append(ep['bg_pred'][sl, a].mean() / tru - 1)
                if ma.sum() > 0:
                    # first audit record per audit cluster
                    errs['audit_ht'].append(lab[ma, 5].mean() / tru - 1)
                    resid = lab[ma, 5] - ep['bg_pred'][la_t[ma], a]
                    errs['model_assisted'].append((ep['bg_pred'][sl, a].mean() + resid.mean()) / tru - 1)
        for k, v in errs.items():
            v = np.array(v)
            out[f'exp_bias_{k}'] = float(v.mean()) if len(v) else np.nan
            out[f'exp_rmse_{k}'] = float(np.sqrt((v ** 2).mean())) if len(v) else np.nan
    # ---------------- drift auditing
    if len(D):
        for mode in ('audit', 'opportunistic'):
            r = drift_eval(sim, D, mode)
            out.update({f'{k}_{mode}': v for k, v in r.items()})
    ctl = sim.ctl
    if hasattr(ctl, 'n_plans'):
        out['infeasible_frac'] = ctl.n_infeasible / max(ctl.n_plans, 1)
        out['solve_ms_per_replan'] = 1000 * ctl.solve_time / max(ctl.n_plans / NA, 1)
    if hasattr(ctl, 'm') and getattr(ctl, 'readiness', '') == 'model':
        out['tau0_hat'] = float(np.exp(ctl.m[0])); out['beta_hat'] = float(ctl.m[1])
        out['tau0_sd_log'] = float(np.sqrt(ctl.S[0, 0])); out['n_settle'] = int(ctl.n_settle)
    if extra:
        out.update(extra)
    return out
