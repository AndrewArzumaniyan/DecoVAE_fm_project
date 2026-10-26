"""Run held-out test evaluation for a saved checkpoint without resuming training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adapters import Arm
from data import L, T, normalize
from runner import OUT, algorithm, cache_components, evaluate
from data import PERIOD_DECOMP_DIR


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run_id')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this execution context')
    torch.set_num_threads(1)
    run=OUT/a.run_id
    reference=json.loads((run/'data_reference.json').read_text())
    cache=Path(reference['cache'])
    for name,expected in reference['sha256'].items():
        if sha256(cache/name)!=expected:
            raise ValueError(f'Data cache hash changed: {name}')
    cfg=json.loads((run/'provenance.json').read_text())['config']
    checkpoint_path=run/'best.pt'
    checkpoint=torch.load(checkpoint_path,map_location=a.device,weights_only=False)
    model=Arm(cfg['arm']).to(a.device)
    model.load_state_dict(checkpoint['model'])
    raw=dict(np.load(cache/'test.npz'))
    z,mean,scale=normalize(raw['x'])
    raw.update(z=z.astype('float32'),mean=mean,scale=scale)
    component_cache=None
    if cfg['arm'] in 'CD':
        adapter_hash=hashlib.sha256(
            (PERIOD_DECOMP_DIR/'period_decomp.py').read_bytes()
            +(Path(__file__).parent/'adapters.py').read_bytes()
        ).hexdigest()[:12]
        component_cache=cache/f'test_period_{adapter_hash}.npz'
        raw['components']=cache_components(raw['z'],component_cache)
    frame,outputs=evaluate(model,raw,a.device,a.samples,a.limit)
    frame['arm']=cfg['arm']; frame['seed']=int(cfg['seed']); frame['checkpoint_epoch']=int(checkpoint['epoch'])
    if 'block' in raw:
        frame['block']=raw['block'][:len(frame)]
        frame['scenario']=frame['block']
        frame['noise_level']=[float(np.mean(raw['variance'][int(i),:T]))**.5 for i in frame.series_id]
    output=run/'checkpoint_test_eval'
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'fm_rows.csv',index=False)
    np.savez_compressed(output/'forecast_outputs.npz',outputs=np.array(outputs,dtype=object))
    summary=frame.groupby('block')[['crps','rmse','coverage90','width90','variance_mae']].mean().reset_index() if 'block' in frame else frame[['crps','rmse','coverage90','width90','variance_mae']].mean().to_frame().T
    summary.to_csv(output/'summary.csv',index=False)
    status=dict(status='complete',run_id=a.run_id,checkpoint=str(checkpoint_path),checkpoint_epoch=int(checkpoint['epoch']),
                validation_crps=float(checkpoint['best']),device=a.device,samples=a.samples,limit=a.limit,
                rows=len(frame),expected_test_rows=len(raw['x']),test_evaluated=True,
                data_reference=reference,checkpoint_sha256=sha256(checkpoint_path),
                component_cache=str(component_cache) if component_cache else None,
                gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None)
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    lines=['# Test-only checkpoint evaluation','',
           f"Run: `{a.run_id}`; selected checkpoint epoch {checkpoint['epoch']}; validation CRPS {checkpoint['best']:.6f}.",
           f"Evaluated {len(frame)} of {len(raw['x'])} test series with {a.samples} forecast samples each on `{a.device}`.",
           'This evaluates the saved validation-selected checkpoint; no optimizer step or training update was run.', '',
           '## Metrics by synthetic block','',summary.to_string(index=False),'']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),rows=len(frame),device=a.device,epoch=int(checkpoint['epoch'])),indent=2),flush=True)


if __name__=='__main__':
    main()
