"""Slot-level emulator: true sensor conditioning physics, energy, links, audits, alerts and metrics.
Controllers only see what a gateway would see (commands issued, delivered packets, battery heartbeats,
public weather, forecasts)."""
import numpy as np
import config as C
import episode as E

NA, NPA, NN = C.N_AREAS, C.NODES_PER_AREA, C.N_NODES


def tau_on(temp, tau_ref=C.TAU_ON_REF, beta=C.BETA_T):
    return tau_ref * np.exp(beta * (C.T_REF - temp))


def warmup_slots(theta0, temp, tau_ref=C.TAU_ON_REF, beta=C.BETA_T):
    """slots of heating needed to move conditioning from theta0 to THETA_STAR."""
    th = np.minimum(theta0, C.THETA_STAR)
    w = tau_on(temp, tau_ref, beta) * np.log((1 - th) / (1 - C.THETA_STAR)) / C.DT
    return np.ceil(np.maximum(w, 0) - 1e-9).astype(int)


class Sim:
    def __init__(self, ep, controller, plant=None, audit=False, audit_rate=C.AUDIT_RATE, audit_seed=12345,
                 audit_full_frac=C.AUDIT_FULL_FRAC):
        self.ep, self.ctl = ep, controller
        self.plant = dict(tau_ref=C.TAU_ON_REF, beta=C.BETA_T, tau_off=C.TAU_OFF)
        if plant:
            self.plant.update(plant)
        T = ep['T']
        self.T = T
        self.area = ep['node_area']
        self.theta = np.full(NN, 0.0)
        self.B = np.full(NN, C.B_INIT_FRAC * C.B_MAX)
        self.alive = np.ones(NN, bool)
        self.on_run = np.zeros(NN, int)          # consecutive heater-on slots
        self.Fd = E.duration_cdf_slots()
        # audit schedule: pre-committed Bernoulli draws independent of controller and environment
        self.audit = audit
        ar = np.random.default_rng(audit_seed + ep['seed'])
        self.audit_fire = ar.random((T, NA)) < audit_rate if audit else np.zeros((T, NA), bool)
        self.audit_full = ar.random((T, NA)) < audit_full_frac
        self.audit_node = ar.integers(0, NPA, (T, NA))
        self.audit_open = np.full(NA, -1)        # slot at which current audit opened
        self.audit_done = np.zeros(NN, bool)
        self.alert_until = np.full(NA, -1)
        self.etx_hat = np.full(NN, 1.5)
        self.noise = np.random.default_rng(ep['seed'] + 99991)
        # logs
        self.log_heat = np.zeros((T, NN), bool)
        self.log_ready = np.zeros((T, NN), bool)        # truly stabilised and alive
        self.log_B = np.zeros((T, NN), np.float32)
        self.deliv = []      # (t, node, y, truly_valid, kind) kind: 0 routine/event, 1 audit
        self.E_heat = np.zeros(NN); self.E_tx = np.zeros(NN); self.E_samp = np.zeros(NN)
        self.brown = np.zeros(NN)
        self.slots_audit = self.slots_alert = self.slots_ctl = 0
        self.settle_obs = []  # (temp, theta0_true_hidden_not_used, elapsed_slots)
        self.heat_start_theta = np.zeros(NN)
        self.heat_start_t = np.full(NN, -1)
        self.settled_reported = np.ones(NN, bool)

    # ------------------------------------------------------------------ helpers visible to controllers
    def bg_pred(self, t):
        return self.ep['bg_pred'][t]

    def threshold(self, t):
        return (1.0 + C.DETECT_KAPPA) * self.ep['bg_pred'][t][self.area]

    # ------------------------------------------------------------------ main loop
    def run(self):
        ep, ctl = self.ep, self.ctl
        ctl.init(self)
        plant = self.plant
        for t in range(self.T):
            temp_n = ep['temp'][t][self.area]
            # ---- audits open (area-synchronised)
            if self.audit:
                for a in np.nonzero(ep_fire := self.audit_fire[t])[0]:
                    if self.audit_open[a] < 0:
                        self.audit_open[a] = t
                        if self.audit_full[t, a]:
                            self.audit_done[a * NPA:(a + 1) * NPA] = False
                        else:
                            self.audit_done[a * NPA:(a + 1) * NPA] = True
                            self.audit_done[a * NPA + self.audit_node[t, a]] = False
            # ---- commands
            heat = ctl.decide(self, t).copy()
            audit_need = np.zeros(NN, bool)
            for a in np.nonzero(self.audit_open >= 0)[0]:
                sl = slice(a * NPA, (a + 1) * NPA)
                audit_need[sl] = ~self.audit_done[sl]
                if t - self.audit_open[a] > 25:            # timeout
                    self.audit_open[a] = -1
                    audit_need[sl] = False
            alert = (self.alert_until[self.area] >= t)
            guard = self.B > C.B_RES_FRAC * C.B_MAX
            ctl_heat = heat & (guard | ctl.self_guarded)
            heat = ctl_heat | ((audit_need | alert) & (self.B > 0.05 * C.B_MAX))
            heat &= self.alive
            self.slots_audit += (heat & audit_need).sum()
            self.slots_alert += (heat & alert & ~audit_need).sum()
            self.slots_ctl += (heat & ~alert & ~audit_need).sum()
            # ---- physics: conditioning state
            started = heat & (self.on_run == 0)
            self.heat_start_theta[started] = self.theta[started]
            self.heat_start_t[started] = t
            self.settled_reported[started] = self.theta[started] >= C.THETA_STAR
            ton = tau_on(temp_n, plant['tau_ref'], plant['beta'])
            self.theta = np.where(heat, 1 - (1 - self.theta) * np.exp(-C.DT / ton),
                                  self.theta * np.exp(-C.DT / plant['tau_off']))
            self.on_run = np.where(heat, self.on_run + 1, 0)
            ready_true = heat & (self.theta >= C.THETA_STAR)
            # settling observations reported by nodes (detected from signal derivative)
            newly = ready_true & ~self.settled_reported
            for i in np.nonzero(newly)[0]:
                el = (t - self.heat_start_t[i] + 1) * (1 + C.SETTLE_NOISE * self.noise.standard_normal())
                self.settle_obs.append((temp_n[i], i, el, self.heat_start_t[i]))
            self.settled_reported |= newly
            # ---- readings
            conc = ep['conc'][t]
            bgn = ep['co_bg'][t][self.area]
            art = C.ARTIFACT_GAIN * bgn * np.maximum(0.0, (C.THETA_STAR - self.theta) / C.THETA_STAR)
            y = ep['gain'][t] * conc * (1 + 0.04 * self.noise.standard_normal(NN)) + art
            label = ctl.believed_valid(self, t, heat) & heat
            thr = self.threshold(t)
            exceed = label & (y > thr)
            # ---- transmissions
            tx = exceed | (ctl.routine_tx(self, t, label) & label)
            aud_tx = np.zeros(NN, bool)
            for a in np.nonzero(self.audit_open >= 0)[0]:
                sl = slice(a * NPA, (a + 1) * NPA)
                ok = label[sl] & ~self.audit_done[sl]
                aud_tx[sl] = ok
            tx |= aud_tx
            nsamp = heat.astype(float)
            self.E_samp += nsamp * C.E_SAMPLE
            per = ep['per'][t]
            delivered = np.zeros(NN, bool); attempts = np.zeros(NN)
            for k in range(C.MAX_ATTEMPTS):
                pending = tx & ~delivered
                if not pending.any():
                    break
                attempts += pending
                succ = pending & (self.noise.random(NN) >= per)
                delivered |= succ
            self.E_tx += attempts * C.E_TX
            # gateway link estimate (EWMA of attempts per delivered packet)
            used = attempts > 0
            self.etx_hat[used] = 0.9 * self.etx_hat[used] + 0.1 * np.where(delivered[used], attempts[used], C.MAX_ATTEMPTS + 1)
            for i in np.nonzero(delivered)[0]:
                self.deliv.append((t, i, y[i], ready_true[i], int(aud_tx[i]), conc[i], label[i]))
            # audit bookkeeping (audit sample counts once delivered or attempted)
            self.audit_done |= aud_tx
            for a in np.nonzero(self.audit_open >= 0)[0]:
                if self.audit_done[a * NPA:(a + 1) * NPA].all():
                    self.audit_open[a] = -1
            # alerts: delivered exceedance keeps whole area heated for characterisation
            for i in np.nonzero(delivered & exceed)[0]:
                self.alert_until[self.area[i]] = t + 3
            ctl.observe(self, t, delivered, y, label, exceed, aud_tx, heat)
            # ---- energy
            eh = heat * C.P_HEATER * C.DT
            self.E_heat += eh
            dB = (ep['harvest'][t] - C.P_BASE) * C.DT - eh - nsamp * C.E_SAMPLE - attempts * C.E_TX
            dB = np.where(self.alive, dB, ep['harvest'][t] * C.DT)
            self.B = np.clip(self.B + dB, 0, C.B_MAX)
            died = self.alive & (self.B <= 0)
            self.alive &= ~died
            self.alive |= (~self.alive) & (self.B >= C.B_RESTART_FRAC * C.B_MAX)
            self.brown += ~self.alive
            self.log_heat[t] = heat; self.log_ready[t] = ready_true & self.alive; self.log_B[t] = self.B
        return self
