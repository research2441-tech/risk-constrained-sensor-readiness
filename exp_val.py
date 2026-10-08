"""Validation runs (2013 windows, disjoint from test) used only to set baseline knobs at matched energy."""
import runner as R
GRIDS = {'PDC': [8, 10, 12, 15, 20, 30, 45], 'ENO': [0.4, 0.6, 0.8, 1.0, 1.2], 'EAAS': [3, 4, 6, 10, 15],
         'UGS': [0.005, 0.01, 0.02, 0.05, 0.1], 'LDP': [10, 30, 60, 100, 200, 300]}
if __name__ == '__main__':
    jobs = []
    for ep in R.val_episodes():
        jobs.append((ep, 'RCSR', None, {}, None, 'val'))
        for m, g in GRIDS.items():
            for k in g:
                jobs.append((ep, m, k, {}, None, 'val'))
    R.run_jobs(jobs, R.RES + '/val.jsonl')
