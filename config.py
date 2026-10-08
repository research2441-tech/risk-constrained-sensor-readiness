"""Shared constants for the risk-constrained sensor-readiness emulator (SI units unless noted)."""
import numpy as np

DT = 60.0                 # control slot length [s]
SLOTS_PER_DAY = 1440
N_AREAS = 12              # one monitoring area per Beijing station
NODES_PER_AREA = 3
N_NODES = N_AREAS * NODES_PER_AREA
DESIGNATED = np.array([0, 3, 7, 9])   # areas with a stricter risk target (urban core stations)

# ---------------- energy model ----------------
P_HEATER = 0.25           # heater power drawn from battery incl. regulator losses [W] (210 mW heater, 85 % conversion)
P_BASE = 0.0004           # MCU + RTC + regulator quiescent [W]
E_SAMPLE = 0.005          # one screening read-out (ADC + MCU wake) [J]
E_TX = 0.06               # one LPWAN uplink attempt [J]
MAX_ATTEMPTS = 3
B_MAX = 3.7 * 3600.0      # 1000 mAh, 3.7 V cell [J]
B_RES_FRAC = 0.20         # reserve that discretionary heating must not consume
B_RESTART_FRAC = 0.10     # brown-out recovery level
B_INIT_FRAC = 0.60
P_PEAK = 0.75             # panel peak power at 1000 W/m2 [W]
ETA_CHG = 0.85            # charge-path efficiency

# ---------------- sensor readiness model (true plant) ----------------
TAU_ON_REF = 80.0         # conditioning time constant at 20 C with heater on [s]
BETA_T = 0.025            # temperature sensitivity of tau_on [1/K]
T_REF = 20.0
TAU_OFF = 1800.0          # decay constant of surface conditioning while unpowered [s]
THETA_STAR = 0.95         # stabilisation threshold on conditioning state
ARTIFACT_GAIN = 3.0       # initial-action artefact: apparent concentration multiplier at theta = 0
NOMINAL_WARMUP_SLOTS = 2  # datasheet-style fixed warm-up used by readiness-blind schedulers (2 min)
SETTLE_NOISE = 0.10       # relative noise of reported settling time observations

# ---------------- events ----------------
EVENT_MEDIAN_MIN = 6.0
EVENT_SIGMA = 0.6
EVENT_MIN, EVENT_MAX = 2, 40
DETECT_KAPPA = 0.35       # exceedance ratio over predicted background
M_CHAR = 3                # valid delivered samples needed to characterise an event
LAMBDA0_PER_DAY = 5.0

# ---------------- control ----------------
H = 60                    # receding-horizon length [slots]
REPLAN = 5                # replanning period [slots]
EPS_RISK = 0.20           # normalised risk target (expected miss fraction)
EPS_RISK_DESIGNATED = 0.08
AUDIT_RATE = 1.0 / 240.0  # Bernoulli audit probability per area and slot (6 audits per area-day)
AUDIT_FULL_FRAC = 1.0 / 3.0  # share of audits that interrogate every node of the area (calibration audits)
Q_HARVEST = 0.10          # lower quantile used for conservative harvest
Q_HAZARD = 0.80           # upper quantile used for hazard
ROUTINE_REPORT = 15       # routine report period while ready [slots]
