# Risk-constrained readiness scheduling of heated gas sensors

Code, prepared traces and per-episode results for the manuscript *Risk-Constrained Readiness Scheduling of Heated Gas Sensors in Solar-Powered Environmental Networks with Randomized Audits*.

Everything lives in this one folder: scripts read inputs from, and write outputs to, the folder they are in.

## What is here

| File | Role |
|---|---|
| `config.py` | Plant, energy, event and control constants (Table IV of the paper) |
| `prep_data.py` | Builds `beijing.npz`, `kc30.npz`, `bsrn_analogues.npz` from the public raw files |
| `episode.py` | Trace-driven episode: Beijing meteorology and CO, transposed irradiance, injected events, drift, Gilbert-Elliott links |
| `hazard.py` | Poisson GLM fitted on the 2013 commissioning windows (`python hazard.py` writes `glm.json`) and the audit Gamma recalibration |
| `sim.py` | Slot-level emulator: conditioning physics, initial-action artefact, energy, uplink, audits, alerts |
| `controllers.py` | Proposed controller (`RCSR`) and baselines `Periodic` (PDC), `EnergyNeutral` (ENO), `AdaptiveSampling` (EAAS), `UncertaintyGuided` (UGS), `Lyapunov` (LDP) |
| `metrics.py` | Recall, delay, contamination, false alarms, availability, reserve violation, exposure estimators, CUSUM drift detection |
| `runner.py` | Episode definitions (validation 2013; test 2014-2016 x two irradiance years) and the parallel job runner |
| `exp_val.py`, `select_knobs.py` | Validation sweeps and energy-matched knob selection (`knobs.json`) |
| `exp_main.py` | Experiments: `main`, `ablation`, `pareto`, `robust`, `sens` |
| `analysis.py` | Bootstrap CIs, Wilcoxon signed-rank with Holm correction, rank-biserial effect sizes, Friedman test |
| `optgap.py`, `bench.py` | Exhaustive-search optimality check and planner timing |
| `fig_*.py`, `figstyle.py` | Figures 1-7 (grayscale) |
| `*.jsonl`, `*.json` | Results used in the paper (one JSON object per episode and method) |

## Reproduce

```bash
pip install -r requirements.txt
python hazard.py                 # glm.json (already included)
python exp_val.py && python select_knobs.py
bash run_all.sh                  # main, ablation, pareto, robust, sens  (about 4-5 h on two cores)
python analysis.py main          # stats_main.json
python fig_results.py pareto robust audit sens
```

The prepared traces (`*.npz`) are included. To rebuild them, clone the raw sources and run `python prep_data.py <raw_dir> .`, where `<raw_dir>` contains
`submission_air_quality/data/PRSA_Data_*_20130301-20170228.csv` (Beijing Multi-Site Air-Quality data, UCI) and
`pvrepo/tests/data/` from the pvlib-python repository (NSRDB PSM files and BSRN Payerne June 2016).

## Notes

* Events, sensor drift and the conditioning physics are emulated with known ground truth; meteorology, CO background and irradiance variability are measured.
* All randomness is seeded per episode; audit times are drawn from a separate pre-committed generator.
