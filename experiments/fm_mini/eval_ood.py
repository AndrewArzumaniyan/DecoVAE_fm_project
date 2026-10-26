"""Zero-shot (test_ood) evaluation of a finished mini-FM run's best checkpoint; same code path as train.py."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train as T
import runner


def main():
    p = argparse.ArgumentParser(); p.add_argument('run_id'); p.add_argument('--device', default='cpu'); a = p.parse_args()
    path = T.OUT/a.run_id; cfg = json.loads((path/'provenance.json').read_text())['config']
    model = T.make_model(cfg['arm']).to(a.device)
    model.load_state_dict(torch.load(path/'best.pt', weights_only=False, map_location=a.device)['model'])
    ood = T.load_split(Path(cfg['cache']), 'test_ood', cfg['arm'], cfg['clip'])
    df, outputs = runner.evaluate(model, ood, a.device, 128)
    df['dataset'] = ood['dataset'][:len(df)]; df['arm'] = cfg['arm']; df['seed'] = cfg['seed']
    df['ncrps'] = df.crps/ood['scale'][:len(df), 0]; df['nmae'] = df.mae/ood['scale'][:len(df), 0]
    df.to_csv(path/'ood_rows.csv', index=False)
    np.savez_compressed(path/'ood_outputs.npz', mean=np.concatenate([o['mean'] for o in outputs]),
                        q05=np.concatenate([np.quantile(o['samples'], .05, axis=0) for o in outputs]),
                        q95=np.concatenate([np.quantile(o['samples'], .95, axis=0) for o in outputs]),
                        decoded=np.concatenate([o['decoded_normalized'] for o in outputs]))
    status = json.loads((path/'status.json').read_text())
    status.update(ood_evaluated=True, ood_device=a.device, parameters=sum(v.numel() for v in model.parameters()))
    (path/'status.json').write_text(json.dumps(status, indent=2))
    print(df.groupby('dataset').ncrps.mean().to_dict())


if __name__ == '__main__':
    main()
