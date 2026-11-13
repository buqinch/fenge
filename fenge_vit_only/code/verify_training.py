#!/usr/bin/env python3
"""Exercise the actual 1024 trainer, frozen/unfrozen resume and evaluator on a tiny subset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT / 'result/runtime_check_training')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists():
        raise FileExistsError('Choose a fresh verification directory')
    common = ['--config', str(ROOT / 'config/vit_only_m2f_8000_1024.json'),
              '--output', str(output), '--device', args.device, '--epochs', '2',
              '--freeze-epochs', '1', '--batch-size', '2', '--accumulation', '1',
              '--max-train-samples', '3', '--max-val-samples', '2', '--workers', '0',
              '--log-every', '1', '--val-progress-every', '1']
    # Stop only after the complete first epoch is durably committed. This exercises
    # the normal CLI resume without changing the two-epoch learning-rate budget.
    stop_at_boundary = '''import train
original = train.write_json
def write_then_stop(path, value):
    original(path, value)
    if path.name == 'summary.json' and value.get('completed_epochs') == 1:
        raise SystemExit(0)
train.write_json = write_then_stop
train.main()
'''
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'code'))

    def run(name, command):
        with (output / name).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    run('first_epoch.log', [sys.executable, '-u', '-c', stop_at_boundary, *common])
    first = json.loads((output / 'summary.json').read_text())
    assert first['completed_epochs'] == 1 and first['partial']
    run('resume.log', [sys.executable, '-u', str(ROOT / 'code/train.py'), *common,
                       '--resume', str(output / 'last.pt')])
    summary = json.loads((output / 'summary.json').read_text())
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
    assert summary['completed_epochs'] == 2 and summary['status'] == 'complete'
    assert [row['epoch'] for row in history] == [1, 2]
    assert [row['backbones_frozen'] for row in history] == [True, False]
    assert summary['global_update'] == 4 and summary['partial']
    run('evaluation.log', [sys.executable, '-u', str(ROOT / 'code/evaluate.py'),
        '--checkpoint', str(output / 'best_smoke.pt'), '--split', 'val',
        '--device', args.device, '--max-samples', '2', '--workers', '0',
        '--tta', 'multiscale_hflip', '--tta-sizes', '1024', '768',
        '--tta-weights', '0.5', '0.5', '--output', str(output / 'evaluation.json')])
    evaluation = json.loads((output / 'evaluation.json').read_text())
    assert evaluation['partial'] and evaluation['image_count'] == 2
    report = {'passed': True, 'scope': 'Execution check only, not a model performance result',
              'input_size': 1024, 'train_samples': 3, 'validation_samples': 2,
              'frozen_epoch': 1, 'unfrozen_epoch_after_resume': 2,
              'completed_optimizer_updates': 4, 'actual_cli_resume_passed': True,
              'standalone_four_view_evaluation_passed': True, 'directory': str(output)}
    (ROOT / 'result/verification_training.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
