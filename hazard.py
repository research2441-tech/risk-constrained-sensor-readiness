"""Gateway event-hazard model: Poisson GLM fitted on a densely observed commissioning period, with
parameter uncertainty, plus an audit-based Gamma-Poisson recalibration of area x time-of-day multipliers."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import os, json
import numpy as np
from scipy.stats import gamma as gamma_dist, norm
import config as C
import episode as E

N_BINS = 4  # time-of-day bins of 6 h


def design(hour, wind, co_bg):
    """hour (T,), wind/co (T,A) -> X (T, A, p) with area intercepts + shared covariates."""
    T, A = wind.shape
    f = E.glm_features(np.repeat(hour[:, None], A, 1), wind, co_bg)[..., 1:]   # (T,A,3)
    onehot = np.broadcast_to(np.eye(A)[None], (T, A, A))
    return np.concatenate([onehot, f], -1)


def fit_glm(starts, days=14, agg=10, seed=500):
    """Fit Poisson GLM by IRLS on 10-min counts from fully observed commissioning episodes."""
    Xs, ys = [], []
    for k, s in enumerate(starts):
        ep = E.build(s, days=days, seed=seed + k)
        X = design(ep['hour'], ep['wind'], ep['co_bg'])
        cnt = np.zeros((ep['T'], C.N_AREAS))
        for t, a, D, M in ep['events']:
            cnt[int(t), int(a)] += 1
        T = ep['T'] // agg * agg
        Xa = X[:T].reshape(T // agg, agg, C.N_AREAS, -1).mean(1)
        ya = cnt[:T].reshape(T // agg, agg, C.N_AREAS).sum(1)
        Xs.append(Xa.reshape(-1, Xa.shape[-1])); ys.append(ya.reshape(-1))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    off = np.log(agg)
    b = np.zeros(X.shape[1]); b[:C.N_AREAS] = np.log(y.mean() / agg + 1e-9)
    for it in range(50):
        mu = np.exp(X @ b + off)
        g = X.T @ (y - mu)
        Hs = X.T @ (X * mu[:, None])
        step = np.linalg.solve(Hs + 1e-8 * np.eye(len(b)), g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.inv(Hs)
    return dict(beta=b.tolist(), cov=cov.tolist(), n=int(y.sum()))


class HazardModel:
    def __init__(self, glm, use_audit=True, prior_events=3.0, half_life_days=5.0, q=C.Q_HAZARD, upper=True):
        self.b = np.array(glm['beta']); self.cov = np.array(glm['cov'])
        self.use_audit, self.q, self.upper = use_audit, q, upper
        self.n0 = prior_events
        self.forget = 0.5 ** (1.0 / (half_life_days * C.SLOTS_PER_DAY))
        self.k = np.zeros((C.N_AREAS, N_BINS)); self.o = np.zeros((C.N_AREAS, N_BINS))
        self.zq = norm.ppf(q)

    def point(self, hour, wind, co):
        """per-slot hazard (T,A) from the GLM and the standard error of the linear predictor."""
        hour = np.atleast_1d(hour); wind = np.atleast_2d(wind); co = np.atleast_2d(co)
        A = C.N_AREAS
        f = E.glm_features(np.repeat(hour[:, None], A, 1), wind, co)[..., 1:]      # (T,A,3)
        ba, bf = self.b[:A], self.b[A:]
        eta = ba[None, :] + f @ bf
        S = self.cov
        Saa = np.diag(S)[:A]; Saf = S[:A, A:]; Sff = S[A:, A:]
        var = Saa[None, :] + 2 * np.einsum('tak,ak->ta', f, Saf) + np.einsum('tak,kl,tal->ta', f, Sff, f)
        return np.exp(eta), np.sqrt(np.maximum(var, 0))

    def multiplier(self):
        a = self.n0 + self.k; r = self.n0 + self.o
        mean = a / r
        up = gamma_dist.ppf(self.q, a, scale=1.0 / r)
        return mean, up

    def predict(self, hour, wind, co, upper=None):
        upper = self.upper if upper is None else upper
        lam, se = self.point(hour, wind, co)
        bins = (np.atleast_1d(hour) // 6).astype(int) % N_BINS
        mean, up = self.multiplier()
        if not upper:
            return lam * mean[:, bins].T
        return lam * np.exp(self.zq * se) * up[:, bins].T

    def decay(self, nslots=1):
        f = self.forget ** nslots
        self.k *= f; self.o *= f

    def update(self, area, hour, event_active, expected_occ):
        b = int(hour // 6) % N_BINS
        self.k[area, b] += float(event_active)
        self.o[area, b] += float(expected_occ)


def mean_duration_slots():
    F = E.duration_cdf_slots()
    return float(np.sum(1 - F))


if __name__ == '__main__':
    starts = ['2013-03-05', '2013-03-25', '2013-04-15', '2013-05-05', '2013-05-25', '2013-06-15', '2013-07-05',
              '2013-07-25', '2013-08-15']
    glm = fit_glm(starts)
    out = os.environ.get('RCSR_OUT', _HERE)
    json.dump(glm, open(os.path.join(out, 'glm.json'), 'w'))
    print('events used', glm['n'], 'coef (shared)', np.round(glm['beta'][-3:], 3))
