"""Schedulers: proposed risk-constrained readiness controller (RCSR) and baselines."""
import numpy as np
from scipy.stats import norm
import config as C
import episode as E
from hazard import HazardModel, mean_duration_slots
from sim import tau_on, warmup_slots, NA, NPA, NN

ED = mean_duration_slots()
Z90 = 1.2815515655446004


# ============================================================ common base (readiness-blind behaviour)
class Base:
    name = 'base'
    self_guarded = False
    uses_audit = False

    def init(self, sim):
        self.area = sim.area
        self.offset = (np.arange(NN) % NPA) / NPA + (np.arange(NN) // NPA) * 0.137

    def believed_valid(self, sim, t, heat):
        return sim.on_run > C.NOMINAL_WARMUP_SLOTS

    def routine_tx(self, sim, t, label):
        return np.zeros(NN, bool)

    def observe(self, sim, t, delivered, y, label, exceed, aud, heat):
        pass


# ------------------------------------------------------------ periodic duty cycling
class Periodic(Base):
    name = 'PDC'

    def __init__(self, period=15):
        self.P = int(period)

    def init(self, sim):
        super().init(sim)
        self.phase = (self.offset * self.P).astype(int)

    def decide(self, sim, t):
        c = (t + self.phase) % self.P
        return c <= C.NOMINAL_WARMUP_SLOTS

    def routine_tx(self, sim, t, label):
        return ((t + self.phase) % self.P) == C.NOMINAL_WARMUP_SLOTS


# ------------------------------------------------------------ energy-neutral adaptive duty cycling (Kansal-type)
class EnergyNeutral(Base):
    name = 'ENO'

    def __init__(self, util=1.0, cycle=20):
        self.u, self.cyc = util, cycle

    def init(self, sim):
        super().init(sim)
        self.phase = (self.offset * self.cyc).astype(int)
        self.h_ewma = np.full((24, NN), np.nan)
        self.duty = np.full(NN, 0.2)
        self.acc = np.zeros(NN)

    def decide(self, sim, t):
        hr = int(sim.ep['hour'][t])
        self.acc += sim.ep['harvest'][t]
        if (t + 1) % 60 == 0:
            obs = self.acc / 60.0; self.acc[:] = 0
            prev = self.h_ewma[hr]
            self.h_ewma[hr] = np.where(np.isnan(prev), obs, 0.5 * prev + 0.5 * obs)
            daily = np.nanmean(self.h_ewma, 0)            # predicted mean harvest power
            err = (sim.B - 0.6 * C.B_MAX) / (C.P_HEATER * 86400.0)
            self.duty = np.clip(self.u * (np.nan_to_num(daily) - C.P_BASE) / C.P_HEATER + 0.5 * err, 0.03, 0.95)
        on = np.maximum(C.NOMINAL_WARMUP_SLOTS + 1, np.round(self.duty * self.cyc)).astype(int)
        c = (t + self.phase) % self.cyc
        return c < on

    def routine_tx(self, sim, t, label):
        c = (t + self.phase) % self.cyc
        return (c == C.NOMINAL_WARMUP_SLOTS) | ((c % C.ROUTINE_REPORT == 0) & (c > 0))


# ------------------------------------------------------------ energy-aware adaptive sampling (Alippi-type)
class AdaptiveSampling(Base):
    name = 'EAAS'

    def __init__(self, tmin=6, tmax=60):
        self.tmin, self.tmax = tmin, tmax

    def init(self, sim):
        super().init(sim)
        self.Ts = np.full(NN, float(self.tmin * 3))
        self.next = (self.offset * self.Ts).astype(int)
        self.last = np.full(NN, np.nan)
        self.active = np.zeros(NN, int)

    def decide(self, sim, t):
        start = (t >= self.next) & (self.active == 0)
        self.active[start] = C.NOMINAL_WARMUP_SLOTS + 1
        on = self.active > 0
        self.active = np.maximum(self.active - 1, 0)
        return on

    def routine_tx(self, sim, t, label):
        return label & (sim.on_run == C.NOMINAL_WARMUP_SLOTS + 1)

    def observe(self, sim, t, delivered, y, label, exceed, aud, heat):
        done = label & (sim.on_run == C.NOMINAL_WARMUP_SLOTS + 1)
        for i in np.nonzero(done)[0]:
            if np.isfinite(self.last[i]):
                rel = abs(y[i] - self.last[i]) / max(self.last[i], 1.0)
                if rel > 0.15:
                    self.Ts[i] = max(self.tmin, self.Ts[i] / 2)
                elif rel < 0.05:
                    self.Ts[i] = min(self.tmax, self.Ts[i] * 1.5)
            self.last[i] = y[i]
            # energy-aware stretch when battery is below target
            lowb = sim.B[i] < 0.5 * C.B_MAX
            ts = self.Ts[i] * (1.0 + 2.0 * max(0.0, 0.5 - sim.B[i] / C.B_MAX) / 0.5) if lowb else self.Ts[i]
            self.next[i] = t + int(max(self.tmin, ts))


# ------------------------------------------------------------ hazard helpers shared by model-based schedulers
class _HazardMixin:
    def hz_init(self, sim, glm, use_audit, upper=True):
        self.hz = HazardModel(glm, use_audit=use_audit, upper=upper)

    def hz_forecast(self, sim, t, H, upper=True):
        ep = sim.ep
        hours = (ep['hour'][t] + np.arange(H) / 60.0) % 24
        wind = np.repeat(ep['wind'][t][None], H, 0)
        co = np.repeat(ep['bg_pred'][t][None], H, 0)
        return self.hz.predict(hours, wind, co, upper=upper)          # (H, A)


# ------------------------------------------------------------ uncertainty-guided scheduling
class UncertaintyGuided(Base, _HazardMixin):
    name = 'UGS'

    def __init__(self, glm, p_th=0.25, tmax=90, hold=3):
        self.glm, self.p_th, self.tmax, self.hold = glm, p_th, tmax, hold

    def init(self, sim):
        super().init(sim)
        self.hz_init(sim, self.glm, use_audit=False)
        self.last_valid = np.full(NA, -10 ** 6)
        self.busy = np.zeros(NN, int)

    def decide(self, sim, t):
        if t % C.REPLAN == 0:
            lam = self.hz_forecast(sim, t, 10)
            p_ev = 1 - np.exp(-lam.sum(0))
            stale = (t - self.last_valid) > self.tmax
            for a in np.nonzero((p_ev > self.p_th) | stale)[0]:
                sl = slice(a * NPA, (a + 1) * NPA)
                if (self.busy[sl] > 0).any():
                    continue
                j = a * NPA + int(np.argmax(sim.B[sl]))
                self.busy[j] = C.NOMINAL_WARMUP_SLOTS + 1 + self.hold
        on = self.busy > 0
        self.busy = np.maximum(self.busy - 1, 0)
        return on

    def routine_tx(self, sim, t, label):
        return label

    def observe(self, sim, t, delivered, y, label, exceed, aud, heat):
        for i in np.nonzero(delivered & label)[0]:
            self.last_valid[self.area[i]] = t


# ------------------------------------------------------------ Lyapunov drift-plus-penalty
class Lyapunov(Base, _HazardMixin):
    name = 'LDP'

    def __init__(self, glm, V=3.0e3):
        self.glm, self.V = glm, V

    def init(self, sim):
        super().init(sim)
        self.hz_init(sim, self.glm, use_audit=False)
        self.Fd = E.duration_cdf_slots()
        self.duty = np.arange(NA) * NPA
        self.on = np.zeros(NN, bool)

    def decide(self, sim, t):
        if t % C.REPLAN == 0:
            lam = self.hz_forecast(sim, t, C.REPLAN).mean(0)               # (A,)
            w0 = C.NOMINAL_WARMUP_SLOTS
            benefit = lam * (self.Fd[w0 + C.M_CHAR + 5] - self.Fd[C.M_CHAR])   # missed-event reduction per slot
            Q = np.maximum(0.0, 0.7 * C.B_MAX - sim.B) / C.B_MAX
            self.on[:] = False
            for a in range(NA):
                sl = slice(a * NPA, (a + 1) * NPA)
                j = a * NPA + int(np.argmin(Q[sl]))
                if self.V * benefit[a] >= Q[j] * C.P_HEATER * C.DT / C.B_MAX * 1e3 + 1e-3:
                    self.on[j] = True
        return self.on.copy()

    def routine_tx(self, sim, t, label):
        return label & ((sim.on_run % C.ROUTINE_REPORT) == C.NOMINAL_WARMUP_SLOTS + 1)


# ============================================================ proposed: risk-constrained sensor readiness
class RCSR(Base, _HazardMixin):
    """Risk-constrained receding-horizon readiness controller.

    flags: readiness ('model' | 'fixed2' | 'fixed10'), audit (bool), conservative_harvest (bool),
           upper_hazard (bool), link_aware (bool), strategic (bool)"""
    name = 'RCSR'
    self_guarded = True

    def __init__(self, glm, readiness='model', audit=True, conservative_harvest=True, upper_hazard=True,
                 link_aware=True, strategic=True, eps=C.EPS_RISK, eps_d=C.EPS_RISK_DESIGNATED, name=None,
                 q_harvest=C.Q_HARVEST, recal=None, budget_scale=1.0, q_hazard=C.Q_HAZARD, margin=0.05, n0=3.0):
        self.glm, self.readiness, self.uses_audit = glm, readiness, audit
        self.cons_h, self.upper_h, self.link_aware, self.strategic = conservative_harvest, upper_hazard, link_aware, strategic
        self.eps, self.eps_d, self.qh = eps, eps_d, q_harvest
        self.recal = recal if recal is not None else ('audit' if audit else 'controller')
        self.budget_scale, self.q_hazard, self.margin, self.n0 = budget_scale, q_hazard, margin, n0
        if name:
            self.name = name

    # ---------------- initialisation
    def init(self, sim):
        super().init(sim)
        self.hz = HazardModel(self.glm, use_audit=self.uses_audit, upper=self.upper_h, q=self.q_hazard, prior_events=self.n0)
        self.Fd = E.duration_cdf_slots()
        self.F0 = self.Fd[C.M_CHAR]
        self.eps_a = np.full(NA, self.eps); self.eps_a[C.DESIGNATED] = self.eps_d
        self.theta_hat = np.zeros(NN)
        self.theta_at_start = np.zeros(NN)
        self.cmd = np.zeros(NN, bool)
        self.duty = np.arange(NA) * NPA
        self.plan_mask = np.zeros((NA, C.H), bool)       # heater-on plan for duty node
        self.plan_t0 = 0
        # readiness-model posterior: log tau = w0 + w1 * (T_REF - T)
        self.m = np.array([np.log(60.0), 0.0]); self.S = np.diag([0.5 ** 2, 0.03 ** 2]); self.s2 = 0.15 ** 2
        self.n_settle = 0
        # harvest forecaster state (per area, 30-min resolution)
        self.kc_hist = []                     # list of (t, kc_obs (A,))
        self.fc_store = []                    # (t_issue, kc_f (L,A))
        self.resid = [[] for _ in range(48)]
        self.acc_h = np.zeros(NA); self.acc_cs = 0.0
        self.allow = np.full(NA, np.inf); self.nu = np.zeros(NA)
        self.cs = sim.ep['cs']; self.derate = sim.ep['derate']
        self.last_aud_area_t = np.full(NA, -1)
        self.n_infeasible = 0; self.n_plans = 0; self.solve_time = 0.0
        st = np.arange(0, C.H, 5); en = np.arange(5, C.H + 1, 5)
        S_, E_ = np.meshgrid(st, en, indexing='ij'); m = E_ > S_
        self._SE = (S_[m], E_[m])

    # ---------------- readiness model
    def tau_params(self):
        if self.readiness != 'model':
            return None
        z = norm.ppf(0.9)
        return self.m, self.S, self.s2, z

    def tau_upper(self, temp):
        x = C.T_REF - np.asarray(temp, float)
        mu = self.m[0] + self.m[1] * x
        var = self.S[0, 0] + 2 * self.S[0, 1] * x + self.S[1, 1] * x * x
        return np.exp(mu + Z90 * np.sqrt(var))

    def warm_slots(self, theta0, temp):
        if self.readiness == 'fixed2':
            return (np.where(np.asarray(theta0) >= 0.999, 0, C.NOMINAL_WARMUP_SLOTS + 1) * np.ones(np.shape(temp))).astype(int)
        if self.readiness == 'fixed10':
            return (np.where(np.asarray(theta0) >= 0.999, 0, 10) * np.ones(np.shape(temp))).astype(int)
        tu = self.tau_upper(temp)
        th = np.minimum(theta0, C.THETA_STAR)
        return np.ceil(tu * np.log((1 - th) / (1 - C.THETA_STAR)) / C.DT - 1e-9).astype(int)

    def believed_valid(self, sim, t, heat):
        if self.readiness == 'fixed2':
            return sim.on_run > C.NOMINAL_WARMUP_SLOTS
        if self.readiness == 'fixed10':
            return sim.on_run >= 10
        temp = sim.ep['temp'][t][self.area]
        tu = self.tau_upper(temp)
        th = np.where(heat, 1 - (1 - self.theta_hat) * np.exp(-C.DT / tu), self.theta_hat * np.exp(-C.DT / C.TAU_OFF))
        self._th_next = th
        return heat & (th >= C.THETA_STAR)

    def _update_tau(self, sim):
        while self.n_settle < len(sim.settle_obs):
            temp, i, el, t0 = sim.settle_obs[self.n_settle]; self.n_settle += 1
            th0 = self.theta_at_start[i]
            if th0 > 0.85 or el < 1:
                continue
            tau_obs = el * C.DT / np.log((1 - th0) / (1 - C.THETA_STAR))
            x = np.array([1.0, C.T_REF - temp])
            Sinv = np.linalg.inv(self.S)
            Sn = np.linalg.inv(Sinv + np.outer(x, x) / self.s2)
            self.m = Sn @ (Sinv @ self.m + x * np.log(tau_obs) / self.s2)
            self.S = Sn

    # ---------------- harvest forecasting with one-sided conformal lower bound
    def _harvest_update(self, sim, t):
        a_h = sim.ep['harvest'][t]
        self.acc_h += np.array([a_h[a * NPA:(a + 1) * NPA].sum() / (self.derate[a * NPA:(a + 1) * NPA].sum()) for a in range(NA)])
        self.acc_cs += self.cs[t]
        if (t + 1) % 30 == 0:
            cs_mean = self.acc_cs / 30.0
            if cs_mean > 30:
                kc = (self.acc_h / 30.0) / (cs_mean / 1000.0 * C.P_PEAK * C.ETA_CHG)
            else:
                kc = np.full(NA, np.nan)
            self.kc_hist.append(np.clip(kc, 0, 1.4))
            self.acc_h[:] = 0; self.acc_cs = 0.0
            k_now = len(self.kc_hist) - 1
            # score earlier forecasts against this observation
            for (k_issue, f) in self.fc_store[-48:]:
                L = k_now - k_issue - 1
                if 0 <= L < 48 and np.isfinite(kc).all():
                    self.resid[L].extend((f[L] - kc).tolist())
                    self.resid[L] = self.resid[L][-300:]
            # issue new forecast for next 48 half-hours
            hist = np.array(self.kc_hist)
            clim = np.nanmean(hist[-48 * 3::48][-3:], 0) if len(hist) >= 48 else np.full(NA, 0.65)
            clim_l = []
            for L in range(48):
                idx = len(hist) - 48 + L + 1
                past = [hist[j] for j in range(idx, -1, -48) if 0 <= j < len(hist)][:3]
                v = np.nanmean(past, 0) if past and np.isfinite(np.nanmean(past)) else np.full(NA, 0.65)
                clim_l.append(np.nan_to_num(v, nan=0.65))
            last = np.nan_to_num(kc, nan=np.nan)
            if not np.isfinite(last).all():
                lastvalid = [h for h in hist[::-1] if np.isfinite(h).all()]
                last = lastvalid[0] if lastvalid else np.full(NA, 0.65)
            f = np.array([np.exp(-(L + 1) / 3.0) * last + (1 - np.exp(-(L + 1) / 3.0)) * clim_l[L] for L in range(48)])
            self.fc_store.append((k_now, f))
            self.fc_store = self.fc_store[-60:]

    def kc_lower(self, L):
        if not self.fc_store:
            return np.full((L, NA), 0.3)
        key = len(self.kc_hist)
        if getattr(self, '_kcl_key', None) == key:
            return self._kcl
        k_issue, f = self.fc_store[-1]
        out = np.zeros((48, NA))
        for l in range(48):
            r = self.resid[l]
            if self.cons_h:
                q = np.quantile(r, 1 - self.qh) if len(r) >= 20 else 0.5
                out[l] = np.clip(f[l] - max(q, 0.0), 0, 1.4)
            else:
                out[l] = f[l]
        self._kcl_key = key; self._kcl = out
        return out

    # ---------------- strategic allocation: hazard-ranked fractional knapsack until harvest resumes
    def _strategic(self, sim, t):
        ep = sim.ep
        hour = ep['hour'][t]
        to10 = ((10.0 - hour) % 24) * 60
        Wn = int(max(60, to10))
        Wn = min(Wn, 1440)
        tt = np.arange(t, min(t + Wn, sim.T))
        Wn = len(tt)
        kcl = self.kc_lower(48)                                       # (48, A) half-hour steps
        steps = np.minimum((np.arange(Wn) // 30), 47)
        kc_slots = kcl[steps]                                         # (Wn, A)
        # conservative harvest energy per node over window
        csw = self.cs[tt] / 1000.0 * C.P_PEAK * C.ETA_CHG * C.DT
        Hn = (csw[:, None] * kc_slots[:, self.area] * self.derate[None, :]).sum(0)
        dur = Wn * C.DT
        audit_cost = (C.AUDIT_RATE * Wn * C.P_HEATER * C.DT * 6.0 * (C.AUDIT_FULL_FRAC + (1 - C.AUDIT_FULL_FRAC) / NPA)) if self.uses_audit else 0.0
        avail = sim.B - (C.B_RES_FRAC + self.margin) * C.B_MAX + Hn - C.P_BASE * dur - audit_cost
        self.avail = avail
        # hazard over window (hour-of-day profile, persistence covariates)
        hours = (hour + np.arange(Wn) / 60.0) % 24
        hc = hours[::10]
        lam = self.hz.predict(hc, np.repeat(ep['wind'][t][None], len(hc), 0), np.repeat(ep['bg_pred'][t][None], len(hc), 0),
                              upper=self.upper_h)
        lam = np.repeat(lam, 10, 0)[:Wn]                              # (Wn, A)
        c = C.P_HEATER * C.DT
        wc = int(np.median(self.warm_slots(np.zeros(NA), ep['temp'][t])))
        self.kappa = self.Fd[min(wc + C.M_CHAR, len(self.Fd) - 1)] - self.F0
        for a in range(NA):
            sl = slice(a * NPA, (a + 1) * NPA)
            budget = np.maximum(avail[sl], 0).sum() * self.budget_scale
            # expected alert/characterisation energy reserved first
            budget -= lam[:, a].sum() * 3 * 6 * c
            n_aff = int(max(budget, 0) // c)
            v = lam[:, a]
            if n_aff >= Wn:
                self.allow[a] = np.inf; self.nu[a] = 0.0
                continue
            if n_aff <= 0:
                self.allow[a] = 0.0; self.nu[a] = np.inf
                continue
            order = np.argsort(-v)
            chosen = np.zeros(Wn, bool); chosen[order[:n_aff]] = True
            self.allow[a] = c * (chosen[:C.H].sum() + 6)
            self.nu[a] = self.kappa * v[order[n_aff - 1]] / c

    # ---------------- tactical plans for all areas (vectorised over areas and candidate plans)
    def _plan_all(self, sim, t, lam, temps):
        Hh, A = C.H, NA
        avail = self.avail
        etx = self.etx(sim) if self.link_aware else np.ones(NN)
        ids = np.arange(NN)
        score = avail - (etx - 1.0) * 400.0 + 2000.0 * np.isin(ids, self.duty) \
            + 1500.0 * (self.theta_hat >= C.THETA_STAR) - 5000.0 * (~sim.alive)
        j = np.arange(A) * NPA + np.argmax(score.reshape(A, NPA), 1)
        self.duty = j
        th0 = self.theta_hat[j]
        hs = np.arange(Hh)
        th_h = th0[:, None] * np.exp(-hs[None, :] * C.DT / C.TAU_OFF)
        th_h[:, 0] = th0
        w_h = self.warm_slots(th_h, np.repeat(temps[:, None], Hh, 1))
        ready_at = hs[None, :] + w_h                                   # (A,H) increasing in h
        S_, E_ = self._SE
        hstar = (ready_at[:, None, :] <= S_[None, :, None]).sum(-1) - 1   # (A,P)
        now = ready_at[:, :1] > S_[None, :]
        hst = np.where(now, 0, np.maximum(hstar, 0))
        r0 = np.where(now, ready_at[:, :1], S_[None, :])
        valid = r0 < E_[None, :]
        k = hs[None, None, :]
        ready = (k >= r0[..., None]) & (k < E_[None, :, None]) & valid[..., None]
        heaton = (k >= hst[..., None]) & (k < E_[None, :, None]) & valid[..., None]
        big = 10 ** 6
        nxt = np.where(ready, k, big)
        nxt = np.minimum.accumulate(nxt[..., ::-1], axis=-1)[..., ::-1]
        w_end = self.warm_slots(th0 * np.exp(-Hh * C.DT / C.TAU_OFF), temps)       # (A,)
        tail = (Hh - hs[None, None, :]) + w_end[:, None, None]
        g = np.where(nxt < big, nxt - k, tail)
        g = np.clip(g, 0, len(self.Fd) - C.M_CHAR - 1)
        pmiss = self.Fd[g + C.M_CHAR] - self.F0
        R = (pmiss * lam.T[:, None, :]).sum(-1)                        # (A,P)
        En = heaton.sum(-1) * C.P_HEATER * C.DT
        g0 = np.clip(tail[:, 0, :], 0, len(self.Fd) - C.M_CHAR - 1)
        R0 = ((self.Fd[g0 + C.M_CHAR] - self.F0) * lam.T).sum(-1)
        R = np.concatenate([R, R0[:, None]], 1); En = np.concatenate([En, np.zeros((A, 1))], 1)
        En = np.where(np.concatenate([valid, np.ones((A, 1), bool)], 1), En, np.inf)
        heaton = np.concatenate([heaton, np.zeros((A, 1, Hh), bool)], 1)
        Lam = lam.sum(0)
        allow = 1.5 * self.allow if self.strategic else np.full(A, np.inf)
        cap = np.minimum(allow, np.maximum(avail[j], 0.0))
        feas = En <= cap[:, None] + 1e-9
        target = self.eps_a * Lam
        sat = feas & (R <= target[:, None])
        if self.strategic:
            nu = np.where(np.isfinite(self.nu), self.nu, 1e9)
        else:
            # myopic price: scarcity of the duty node's own energy above reserve
            frac = np.clip(avail[j] / C.B_MAX, 1e-3, None)
            nu = Lam.mean() / (C.H * C.P_HEATER * C.DT) * (0.2 / frac)
        J1 = np.where(sat, R + nu[:, None] * En + 1e-9 * En, np.inf)
        J2 = np.where(feas, R + nu[:, None] * En, np.inf)
        has = sat.any(1)
        best = np.where(has, np.argmin(J1, 1), np.argmin(J2, 1))
        self.n_plans += A; self.n_infeasible += int((~has).sum())
        self.plan_mask = heaton[np.arange(A), best]
        self.last_R = R[np.arange(A), best]; self.last_Lam = Lam

    def etx(self, sim):
        return sim.etx_hat

    # ---------------- per-slot decision
    def decide(self, sim, t):
        self._harvest_update(sim, t)
        if self.readiness == 'model':
            self._update_tau(sim)
        if t % 15 == 0 and self.strategic:
            self._strategic(sim, t)
        elif not hasattr(self, 'avail'):
            self.avail = sim.B - C.B_RES_FRAC * C.B_MAX
        if t % C.REPLAN == 0:
            import time as _t
            t0 = _t.perf_counter()
            if not self.strategic:
                # myopic availability: current battery above reserve plus conservative next-hour harvest
                self.avail = (sim.B - (C.B_RES_FRAC + self.margin) * C.B_MAX) * self.budget_scale
            lam = self.hz_forecast(sim, t, C.H, upper=self.upper_h)
            if not hasattr(self, 'lam_log'):
                self.lam_log = np.zeros((sim.T, NA))
            self.lam_log[t:t + C.REPLAN] = lam[:min(C.REPLAN, sim.T - t)]
            temps = sim.ep['temp'][t]
            self._plan_all(sim, t, lam, temps)
            self.plan_t0 = t
            self.solve_time += _t.perf_counter() - t0
        k = t - self.plan_t0
        cmd = np.zeros(NN, bool)
        for a in range(NA):
            if self.plan_mask[a, k]:
                cmd[self.duty[a]] = True
        return cmd

    def routine_tx(self, sim, t, label):
        return label & ((sim.on_run % C.ROUTINE_REPORT) == 0)

    def observe(self, sim, t, delivered, y, label, exceed, aud, heat):
        started = heat & (sim.on_run == 1)
        self.theta_at_start[started] = self.theta_hat[started]
        if self.readiness == 'model':
            self.theta_hat = self._th_next if hasattr(self, '_th_next') else self.theta_hat
        self.cmd = heat.copy()
        # hazard recalibration
        self.hz.decay(1)
        hour = sim.ep['hour'][t]
        if self.recal == 'none':
            pass
        elif self.recal == 'audit':
            for a in [a for a in range(NA) if aud[a * NPA:(a + 1) * NPA].any()]:
                if self.last_aud_area_t[a] >= 0 and t - self.last_aud_area_t[a] < 25:
                    continue
                self.last_aud_area_t[a] = t
                sl = slice(a * NPA, (a + 1) * NPA)
                act = bool((exceed[sl] & aud[sl]).any())
                lam_p, _ = self.hz.point(np.array([hour]), sim.ep['wind'][t][None], sim.ep['bg_pred'][t][None])
                self.hz.update(a, hour, act, lam_p[0, a] * ED)
        else:
            lam_p = None
            for a in range(NA):
                sl = slice(a * NPA, (a + 1) * NPA)
                if (label[sl]).any():
                    if lam_p is None:
                        lam_p, _ = self.hz.point(np.array([hour]), sim.ep['wind'][t][None], sim.ep['bg_pred'][t][None])
                    self.hz.update(a, hour, bool(exceed[sl].any()), lam_p[0, a] * ED)
