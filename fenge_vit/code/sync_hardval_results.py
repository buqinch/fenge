#!/usr/bin/env python3
"""Read-only remote text mirroring for every-two-epoch hard-validation runs.

Reuses the existing SSH/rsync allowlist and atomic publishing, while checking
the sparse validation schedule and weighted checkpoint-selection score. This
entry is isolated from the mirrors for older every-epoch validation runs.
"""
from __future__ import annotations

import math
from pathlib import Path

import sync_remote_results as mirror

PROTOCOL = 'pool8002_train6000_hard1500_validation_v1'


def validate_boundary(target, files, documents):
    local = target['local']

    def document(name):
        key = Path(name)
        if key in documents: return documents[key]
        path = local / key
        return mirror.strict_json(path.read_text()) if path.is_file() else None

    configuration = document('run_config.json')
    incoming = documents.get(Path('run_config.json'))
    previous_configuration = local / 'run_config.json'
    if incoming is not None and previous_configuration.exists():
        if incoming != mirror.strict_json(previous_configuration.read_text()):
            raise ValueError('Remote run_config differs from the established mirror identity')
    if configuration is None: return None, 'waiting_for_run_config'
    if configuration.get('protocol') != PROTOCOL or configuration.get('validation_every') != 2:
        raise ValueError('This mirror only accepts the explicit hard1500 every-two-epoch protocol')
    summary = document('summary.json')
    if summary is None: return None, 'waiting_for_first_epoch_summary'
    completed = summary.get('completed_epochs')
    if type(completed) is not int or completed < 1:
        raise ValueError('Invalid completed training epoch')
    if (summary.get('best_metric') != 'validation_weighted_score'
            or summary.get('checkpoint_selection_uses_test') is not True
            or summary.get('official_test_used_for_selection') is not True):
        raise ValueError('Summary does not declare hard-validation checkpoint selection')
    previous = local / 'summary.json'
    if previous.exists() and mirror.strict_json(previous.read_text()).get('completed_epochs', -1) > completed:
        raise ValueError('Remote summary moved behind the previously published epoch')
    history_file = files.get(Path('history.jsonl'), local / 'history.jsonl')
    if not history_file.is_file(): return None, 'waiting_for_history'
    history = [mirror.strict_json(line) for line in history_file.read_text().splitlines() if line.strip()]
    by_epoch = {}
    for row in history:
        epoch = row.get('epoch')
        if type(epoch) is not int or epoch < 1 or epoch in by_epoch:
            raise ValueError('Invalid or duplicate history epoch')
        by_epoch[epoch] = row
    best, best_epoch = None, None
    for epoch in range(1, completed + 1):
        if epoch not in by_epoch: return None, f'waiting_for_history_epoch_{epoch:03d}'
        row = by_epoch[epoch]
        should_validate = epoch % 2 == 0 or epoch == configuration['epochs']
        if row.get('validation_performed') is not should_validate:
            raise ValueError('History validation cadence differs from every two epochs plus the final epoch')
        if should_validate:
            record = document(f'validation_epoch_{epoch:03d}.json')
            if record is None: return None, f'waiting_for_validation_epoch_{epoch:03d}'
            score = record.get('validation_weighted_score')
            if (record.get('epoch') != epoch or isinstance(score, bool)
                    or not isinstance(score, (float, int)) or not math.isfinite(score)
                    or row.get('validation_weighted_score') != score
                    or row.get('validation_miou') != record.get('miou')
                    or record.get('data_identity') != summary['data_identity']['val']
                    or record.get('checkpoint_selection_uses_test') is not True):
                raise ValueError('Validation file, identity and training history disagree')
            if best is None or score > best: best, best_epoch = score, epoch
        elif row.get('validation_weighted_score') is not None or row.get('validation_miou') is not None:
            raise ValueError('A skipped validation epoch must not contain a newly measured score')
        if row.get('best_epoch') != best_epoch or row.get('best_miou') != best:
            raise ValueError('History best differs from the first strict weighted validation maximum')
    if summary.get('best_epoch') != best_epoch or summary.get('best_miou') != best:
        raise ValueError('Summary best differs from the completed weighted validation history')
    return summary, None


def main():
    mirror.validate_boundary = validate_boundary
    return mirror.main()


if __name__ == '__main__':
    raise SystemExit(main())
