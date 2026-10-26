"""Extend the original final-model H1 series to seeds 0..4, without changing the protocol."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    python = str(root / 'env_mentors/.venv/bin/python')
    jobs = []
    checks = []
    for label, run in [('real', 'srv13_Dmc1Zn_s0'), ('strong', 'srv13_h1_Dmc1_s0')]:
        jid = f'reproduce_{label}'
        checks.append(jid)
        output = f'ops/{jid}.json'
        jobs.append(dict(id=jid, cmd=[python, '-B', 'ops/check_reproduction.py', '--root', str(root),
                                     '--run', run, '--output', output], outputs=[output], success_json=output))
    training = []
    # Seeds 0,1,2 are in the transferred reference archive. The main Dmc1 baseline already has five seeds.
    for dataset, arms, cache in [
        ('real', ['Dmc1Zn', 'Dmc1B4'], 'fm_mini_644d2b51896d'),
        ('strong', ['Dmc1', 'Dmc1Zn', 'Dmc1B4'], 'h1fm_strong'),
    ]:
        for seed in [3, 4]:
            for arm in arms:
                jid = f'defense_h1_{dataset}_{arm}_s{seed}'
                run_dir = f'notes/analysis/fm_mini/{jid}'
                command = [python, '-B', 'code/fm_mini/train.py', '--cache', f'code/fm_mini/cache/{cache}',
                           '--arm', arm, '--seed', str(seed), '--run-id', jid, '--device', 'cuda',
                           '--batch', '256', '--lr', '1e-3', '--clip', '20', '--epochs', '472',
                           '--minutes', '300', '--patience', '10', '--val-every', '8',
                           '--fast-val', '--recon-weight', '0.2']
                training.append(dict(id=jid, cmd=command, after=checks, training=True,
                                     run_dir=run_dir, dataset=dataset, arm=arm, seed=seed))
    jobs.extend(training)
    for j in training:
        if j['dataset'] == 'strong':
            out = 'notes/analysis/fm_mini/defense_h1_metrics'
            script_args = ['code/fm_mini/h1_eval.py', j['run_dir'],
                   '--cache', 'code/fm_mini/cache/h1fm_strong', '--out', out,
                   '--splits', 'test_id,test_ood', '--device', 'cuda', '--samples', '128']
            outputs = [f"{out}/{j['id']}_{split}.csv" for split in ['test_id', 'test_ood']]
        else:
            script_args = ['code/fm_mini/eval_space.py', j['run_dir'], '--device', 'cuda']
            outputs = [f"{j['run_dir']}/space_eval_{split}.csv" for split in ['test_id', 'test_ood']]
            outputs += [f"{j['run_dir']}/space_eval_pit_{split}.npz" for split in ['test_id', 'test_ood']]
        success = f"ops/{j['id']}_eval.done.json"
        cmd = [python, '-B', 'ops/eval_driver.py', success, *script_args]
        jobs.append(dict(id=j['id'] + '_eval', cmd=cmd, after=[j['id']], outputs=outputs, success_json=success))
    evaluations = [j['id'] for j in jobs if j['id'].endswith('_eval')]
    jobs.append(dict(id='summarize_h1', cmd=[python, '-B', 'ops/summarize_h1.py', '--root', str(root)],
                     after=evaluations,
                     outputs=['final_results/h1_comparisons.csv', 'final_results/H1_FINDINGS.md'],
                     success_json='final_results/h1_summary_complete.json'))
    critical = [*root.glob('code/fm_mini/*.py'), *root.glob('code/decomp_benchmark/*.py'),
                *root.glob('code/period_decomp/*.py'), *root.glob('ops/*.py')]
    plan = dict(protocol='Final Dmc1 H1 extension, seeds 3 and 4', jobs=jobs,
                source_sha256={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(critical) if not f.name.startswith('.')},
                interpretation={'test_id_of_h1fm_strong': 'strong volatility',
                                'test_ood_of_h1fm_strong': 'base volatility',
                                'B4': 'six residual-volatility statistics and horizon position, not a raw encoder'},
                safeguards=dict(max_parallel=2, cpu_affinity='16-23, two disjoint four-CPU lanes',
                                preserve_other_users=True))
    (root / 'ops/plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (root / 'ops/control.json').write_text(json.dumps({'max_parallel': 1}, indent=2) + '\n')
    print(f'Plan: {len(training)} trainings, {len(jobs)-len(training)-len(checks)} evaluations, {len(checks)} reproduction checks')


if __name__ == '__main__':
    main()
