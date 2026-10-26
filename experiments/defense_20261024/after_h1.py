"""Run guarded diagnostics after the H1 queue ends; preparation stays inside the guard."""
import argparse
from datetime import datetime
import fcntl
import json
import hashlib
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deadline', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if root.stat().st_uid != os.getuid():
        raise SystemExit('Owned root required')
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
           'MKL_NUM_THREADS':'1', 'PYTHONDONTWRITEBYTECODE':'1',
           'VIRTUAL_ENV':str(root/'env_mentors/.venv')}
    summary = root/'ops/queue_summary.json'
    end = datetime.fromisoformat(a.deadline).timestamp()
    while time.time() < end:
        if summary.exists():
            old = json.loads(summary.read_text())
            if old.get('finished'):
                with (root/'ops/queue.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(5)
                        continue
                    fcntl.flock(lock,fcntl.LOCK_UN)
                original = root/'ops/h1_queue_summary.json'
                if not original.exists():
                    original.write_text(json.dumps(old,indent=2)+'\n')
                break
        time.sleep(15)
    else:
        raise SystemExit('Analysis deadline reached while waiting for the H1 queue')
    historical=root/'final_results/historical_control'
    historical.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ['H1_FINDINGS.md','h1_by_seed.csv','h1_by_block.csv','h1_comparisons.csv','h1_summary_complete.json']:
        source=root/'final_results'/name
        if source.exists() and not (historical/name).exists():
            shutil.copy2(source,historical/name)
    python=str(root/'env_mentors/.venv/bin/python')
    jobs=[];evaluations=[]
    for seed in [3,4]:
        jid=f'defense_h1_real_Dmc1_s{seed}'
        run=f'notes/analysis/fm_mini/{jid}'
        jobs.append(dict(id=jid,training=True,run_dir=run,
            cmd=[python,'-B','code/fm_mini/train.py','--cache','code/fm_mini/cache/fm_mini_644d2b51896d',
                 '--arm','Dmc1','--seed',str(seed),'--run-id',jid,'--device','cuda','--batch','256',
                 '--lr','1e-3','--clip','20','--epochs','472','--minutes','300','--patience','10',
                 '--val-every','8','--fast-val','--recon-weight','0.2']))
        marker=f'ops/{jid}_eval.done.json'
        eid=jid+'_eval';evaluations.append(eid)
        jobs.append(dict(id=eid,after=[jid],cmd=[python,'-B','ops/eval_driver.py',marker,
                      'code/fm_mini/eval_space.py',run,'--device','cuda'],
                      outputs=[f'{run}/space_eval_{sp}.csv' for sp in ['test_id','test_ood']],success_json=marker))
    jobs.append(dict(id='matched_h1_summary',after=evaluations,
        cmd=[python,'-B','ops/summarize_h1.py','--root',str(root),'--real-control','matched'],
        outputs=['final_results/h1_comparisons.csv','final_results/H1_FINDINGS.md'],
        success_json='final_results/matched_h1_complete.json'))
    # Separate marker prevents historical-control results from being reused as the matched summary.
    jobs[-1]['cmd']=[python,'-B','ops/eval_driver.py','final_results/matched_h1_complete.json',
                    'ops/summarize_h1.py','--root',str(root),'--real-control','matched']
    jobs.append(dict(id='component_diagnostics',after_any=['matched_h1_summary'],
               cmd=[sys.executable,'-B','ops/evaluate_components.py','--root',str(root),'--draws','128'],
               outputs=['final_results/component_intervals.csv','final_results/COMPONENT_PROTOCOL.md'],
               success_json='final_results/components_complete.json'))
    external=[]
    if (root/'ops/external_reference.py').exists():
        jobs.append(dict(id='external_chronos',after=['component_diagnostics'],
                         cmd=[str(root/'code/chronos_ref/.venv/bin/python'),'-B','ops/external_reference.py','--root',str(root),'--mode','chronos'],
                         outputs=['final_results/external_quantiles_test_id.npz','final_results/external_quantiles_test_ood.npz'],
                         success_json='final_results/external_chronos_complete.json'))
        jobs.append(dict(id='matched_quantile_scores',after=['external_chronos'],
                         cmd=[sys.executable,'-B','ops/external_reference.py','--root',str(root),'--mode','score'],
                         outputs=['final_results/matched_quantile_results.csv','final_results/EXTERNAL_REFERENCE.md'],
                         success_json='final_results/external_qcrps_complete.json'))
        external=['external_chronos','matched_quantile_scores']
    if (root/'ops/assemble_evidence.py').exists():
        jobs.append(dict(id='assemble_evidence',after_any=['matched_h1_summary','component_diagnostics',*external],
                         cmd=[sys.executable,'-B','ops/assemble_evidence.py','--root',str(root)],
                         outputs=['final_results/DEFENSE_STATUS.md','final_results/REVIEW_RESPONSE_DRAFT.md'],
                         success_json='final_results/evidence_complete.json'))
    source_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted((root/'ops').glob('*.py')) if not f.name.startswith('.')}
    (root/'ops/analysis_plan.json').write_text(json.dumps({'jobs':jobs,'source_sha256':source_hashes},indent=2)+'\n')
    os.chdir(root)
    os.execve(sys.executable,[sys.executable,'-B','-u','ops/safe_queue.py','--root',str(root),
                            '--plan',str(root/'ops/analysis_plan.json'),'--deadline',a.deadline],env)


if __name__=='__main__':
    main()
