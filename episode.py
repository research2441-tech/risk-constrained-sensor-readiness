"""Construction of one trace-driven episode: real meteorology, real pollutant background, real irradiance
variability, injected transient events with known ground truth, link processes and sensor drift."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import os
import numpy as np, pandas as pd, pvlib
from scipy.stats import lognorm
import config as C

DATA = os.environ.get('RCSR_DATA', _HERE)
_cache = {}


def _load():
    if not _cache:
        b = np.load(os.path.join(DATA, 'beijing.npz'))
        _cache['bj'] = b['data']                         # (T_hours, 12, 6)
        _cache['bj_t0'] = pd.Timestamp('2013-03-01 00:00')
        k = np.load(os.path.join(DATA, 'kc30.npz'))
        _cache['kc'] = {y: k[y] for y in k.files}
        a = np.load(os.path.join(DATA, 'bsrn_analogues.npz'))
        _cache['an_means'], _cache['an_devs'] = a['means'], a['devs']
    return _cache


# ------------------------------------------------------------------ hazard structure
def rush(hour):
    return np.exp(-0.5 * ((hour - 8.0) / 1.2) ** 2) + np.exp(-0.5 * ((hour - 18.5) / 1.5) ** 2)


def stagnation(w):
    return np.exp(-w / 1.5)


def night_heating(hour, temp):
    h = (hour - 22.0 + 12) % 24 - 12        # centred at 22 h
    return np.exp(-0.5 * (h / 2.5) ** 2) * (temp < 5.0)


def glm_features(hour, wind, co_bg):
    """Covariates available to the gateway hazard model (night heating and weekday omitted)."""
    return np.stack([np.ones_like(hour), rush(hour), stagnation(wind), np.log(np.maximum(co_bg, 100) / 1000.0)], -1)


TRUE_COEF = dict(rush=1.0, stag=0.8, co=0.35, night=1.6, weekday=0.35)


def area_multipliers(year, seed=7):
    r = np.random.default_rng(seed * 1000 + year)
    return np.exp(r.normal(0, 0.35, C.N_AREAS))


def true_hazard_per_slot(hour, wind, co_bg, temp, weekday, mult):
    eta = (TRUE_COEF['rush'] * rush(hour) + TRUE_COEF['stag'] * stagnation(wind)
           + TRUE_COEF['co'] * np.log(np.maximum(co_bg, 100) / 1000.0)
           + TRUE_COEF['night'] * night_heating(hour, temp) + TRUE_COEF['weekday'] * weekday[:, None])
    lam_day = C.LAMBDA0_PER_DAY * 0.45 * mult[None, :] * np.exp(eta)
    return lam_day / C.SLOTS_PER_DAY


def duration_cdf_slots(max_slots=400):
    s = np.arange(max_slots + 1, dtype=float)
    F = lognorm.cdf(s, C.EVENT_SIGMA, scale=C.EVENT_MEDIAN_MIN)
    F[s < C.EVENT_MIN] = 0.0
    F[s >= C.EVENT_MAX] = 1.0
    return F


# ------------------------------------------------------------------ episode
def build(start, days=14, irr_year='y2023', seed=0, temp_offset=0.0, irr_scale=1.0, link_bad=1.0,
          drift_rate=0.03, drift_prob=0.5, hidden_scale=1.0):
    cache = _load()
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    T = days * C.SLOTS_PER_DAY
    h0 = int((start - cache['bj_t0']) / pd.Timedelta(hours=1))
    hourly = cache['bj'][h0: h0 + days * 24 + 2]           # (hours, 12, 6)
    th = np.arange(hourly.shape[0]) * 60.0
    tm = np.arange(T, dtype=float)
    interp = lambda v: np.stack([np.interp(tm, th, hourly[:, a, v]) for a in range(C.N_AREAS)], 1)
    co_h, temp, dew, wind = interp(0), interp(2) + temp_offset, interp(3), interp(4)
    # sub-hourly background fluctuation (AR(1) in log space)
    ar = np.zeros((T, C.N_AREAS)); e = rng.normal(0, 0.012, (T, C.N_AREAS))
    for t in range(1, T):
        ar[t] = 0.985 * ar[t - 1] + e[t]
    co_bg = np.maximum(co_h, 100.0) * np.exp(ar)
    times = start + pd.to_timedelta(tm, unit='min')
    hour = (times.hour + times.minute / 60.0).values.astype(float)
    weekday = (times.dayofweek < 5).astype(float)

    # ---- irradiance: clear-sky at Beijing x transposed clear-sky index with 1-min analogues
    loc = pvlib.location.Location(39.93, 116.40, tz='Asia/Shanghai', altitude=50)
    cs = loc.get_clearsky(times.tz_localize('Asia/Shanghai'), model='ineichen')['ghi'].values
    kc30 = cache['kc'][irr_year]
    doy = (times.dayofyear.values - 1) % 365
    blk = doy * 48 + (times.hour.values * 2 + times.minute.values // 30)
    kc_b = np.nan_to_num(kc30[blk], nan=0.6)
    means, devs = cache['an_means'], cache['an_devs']
    order = np.argsort(means)
    kc_area = np.zeros((T, C.N_AREAS))
    nblk = T // 30
    for a in range(C.N_AREAS):
        tgt = kc_b[::30][:nblk]
        pos = np.searchsorted(means[order], tgt)
        pos = np.clip(pos + rng.integers(-6, 7, nblk), 0, len(order) - 1)
        d = devs[order[pos]]                                    # (nblk, 30)
        kc_area[:, a] = np.clip(np.repeat(tgt, 30) + d.reshape(-1), 0.0, 1.35)
    ghi = cs[:, None] * kc_area * irr_scale                     # (T, A)
    node_area = np.repeat(np.arange(C.N_AREAS), C.NODES_PER_AREA)
    derate = rng.uniform(0.55, 1.0, C.N_NODES)
    harvest = ghi[:, node_area] / 1000.0 * C.P_PEAK * derate[None, :] * C.ETA_CHG   # [W]

    # ---- events (true hazard includes structure the gateway model omits)
    mult = area_multipliers(start.year) ** hidden_scale
    lam = true_hazard_per_slot(hour[:, None], wind, co_bg, temp, weekday, mult)
    if hidden_scale != 1.0:
        lam = lam  # hidden structure strength is controlled through mult and night term below
    onset = rng.random((T, C.N_AREAS)) < lam
    events = []
    conc_ev = np.zeros((T, C.N_NODES))
    for t, a in zip(*np.nonzero(onset)):
        D = int(np.clip(np.round(rng.lognormal(np.log(C.EVENT_MEDIAN_MIN), C.EVENT_SIGMA)), C.EVENT_MIN, C.EVENT_MAX))
        if t + D >= T:
            continue
        M = 1.0 + rng.lognormal(np.log(1.2), 0.4)
        att = rng.uniform(0.6, 1.0, C.NODES_PER_AREA)
        u = (np.arange(D) + 0.5) / D
        shape = np.minimum(1.0, np.minimum(u / 0.2, (1 - u) / 0.25))
        prof = co_bg[t:t + D, a] * (M - 1.0) * shape
        for j in range(C.NODES_PER_AREA):
            conc_ev[t:t + D, a * C.NODES_PER_AREA + j] += att[j] * prof
        events.append((t, a, D, M))
    events = np.array(events, dtype=float).reshape(-1, 4)
    conc = co_bg[:, node_area] + conc_ev

    # ---- calibration drift (sensitivity loss) in a subset of nodes
    drift_on = np.full(C.N_NODES, np.inf)
    gain = np.ones((T, C.N_NODES))
    for a in range(C.N_AREAS):
        if rng.random() < drift_prob:
            i = a * C.NODES_PER_AREA + rng.integers(C.NODES_PER_AREA)
            t0 = int(rng.uniform(3, 9) * C.SLOTS_PER_DAY)
            drift_on[i] = t0
            sgn = -1.0 if rng.random() < 0.7 else 1.0
            gain[t0:, i] = 1.0 + sgn * drift_rate * (np.arange(T - t0) / C.SLOTS_PER_DAY)
    gain = np.clip(gain, 0.3, 2.0)

    # ---- link processes: Gilbert-Elliott per node
    p_good = rng.uniform(0.02, 0.25, C.N_NODES)          # packet error in good state
    p_bad = 0.85
    p_gb = rng.uniform(0.002, 0.01, C.N_NODES) * link_bad
    p_bg = rng.uniform(0.05, 0.2, C.N_NODES)
    bad = np.zeros((T, C.N_NODES), bool)
    st = np.zeros(C.N_NODES, bool)
    u = rng.random((T, C.N_NODES))
    for t in range(T):
        st = np.where(st, u[t] >= p_bg, u[t] < p_gb)
        bad[t] = st
    per = np.where(bad, p_bad, p_good[None, :])

    # predicted background available to every scheduler (hourly persistence with error)
    bg_err = np.exp(np.cumsum(rng.normal(0, 0.004, (T, C.N_AREAS)), 0) * 0.0 + rng.normal(0, 0.05, (T, C.N_AREAS)))
    bg_pred = np.maximum(co_h, 100.0) * bg_err
    return dict(T=T, start=start, hour=hour, weekday=weekday, temp=temp, wind=wind, co_bg=co_bg, co_h=co_h,
                ghi=ghi, cs=cs, kc_area=kc_area, harvest=harvest, derate=derate, node_area=node_area,
                events=events, conc=conc, conc_ev=conc_ev, gain=gain, drift_on=drift_on, per=per,
                bg_pred=bg_pred, lam_true=lam, mult=mult, seed=seed, irr_year=irr_year,
                noise_rng=np.random.default_rng(seed + 99991))
