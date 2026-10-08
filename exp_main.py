import json, sys, runner as R
K = json.load(open(R.RES + '/knobs.json'))['knobs']
which = sys.argv[1] if len(sys.argv) > 1 else 'main'
jobs = []
if which == 'main':
    for ep in R.test_episodes():
        jobs.append((ep, 'RCSR', None, {}, None, 'main'))
        for m, k in K.items():
            jobs.append((ep, m, k, {}, None, 'main'))
elif which == 'ablation':
    V = {'noR2': dict(readiness='fixed2'), 'noR10': dict(readiness='fixed10'), 'noA': dict(audit=False),
         'A_norecal': dict(recal='none'), 'noS': dict(strategic=False), 'noC': dict(conservative_harvest=False),
         'noU': dict(upper_hazard=False), 'noL': dict(link_aware=False)}
    for ep in R.test_episodes():
        for nm, kw in V.items():
            jobs.append((ep, 'RCSR', None, kw, None, 'ablation'))
elif which == 'pareto':
    from exp_val import GRIDS
    for ep in R.subset12():
        for b in (0.5, 0.7, 0.85):
            jobs.append((ep, 'RCSR', None, dict(budget_scale=b), None, 'pareto'))
        for m, g in GRIDS.items():
            for k in g:
                jobs.append((ep, m, k, {}, None, 'pareto'))
elif which == 'sens':
    for ep in R.subset12():
        for a in (0.0, 1 / 480, 1 / 120):
            kw = dict(audit=False) if a == 0 else {}
            jobs.append((ep, 'RCSR', a, kw, dict(audit_rate=a) if a > 0 else None, 'sens_audit'))
        for q in (0.05, 0.25, 0.5):
            jobs.append((ep, 'RCSR', q, dict(q_harvest=q), None, 'sens_qh'))
        for q in (0.5, 0.95):
            jobs.append((ep, 'RCSR', q, dict(q_hazard=q), None, 'sens_qz'))
        for e in (0.1, 0.4):
            jobs.append((ep, 'RCSR', e, dict(eps=e, eps_d=e * 0.4), None, 'sens_eps'))
elif which == 'robust':
    SC = {'link': dict(link_bad=4.0), 'lowsun': dict(irr_scale=0.6), 'cold': dict(temp_offset=-8.0),
          'slowwarm': dict(plant=dict(tau_ref=128.0)), 'fastdecay': dict(plant=dict(tau_off=600.0)),
          'drift': dict(drift_rate=0.06, drift_prob=0.8)}
    for ep in R.subset12():
        for sn, sc in SC.items():
            jobs.append((ep, 'RCSR', None, {}, sc, 'rob_' + sn))
            for m, k in K.items():
                jobs.append((ep, m, k, {}, sc, 'rob_' + sn))
R.run_jobs(jobs, R.RES + f'/{which}.jsonl')
