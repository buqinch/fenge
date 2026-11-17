#!/usr/bin/env python3
"""Read-only remote mirroring for hard-validation and migrated full-test runs.

Reuses the existing SSH/rsync allowlist and atomic publishing, while checking
the sparse validation schedule and weighted checkpoint-selection score. This
entry is isolated from the mirrors for older every-epoch validation runs.
"""
from __future__ import annotations

import math
from pathlib import Path

import sync_remote_results as mirror

PROTOCOL = 'pool8002_train6000_hard1500_validation_v1'
ORIGINAL_ONLY_TRAINING_PROTOCOL = 'pool8002_original_train8002_fulltest_v1'
LEGACY_POLICY = 'hard_validation_best'
FULL_TEST_POLICY = 'full_test_every5'


def _transition_epoch(transition, configuration, previous=None):
    """Bind the only supported policy change without opening model/data changes."""
    if (not isinstance(transition, dict)
            or transition.get('schema') != 'lusnar_training_policy_transition_v1'
            or transition.get('from_policy') != LEGACY_POLICY
            or transition.get('to_policy') != FULL_TEST_POLICY
            or type(transition.get('resume_epoch')) is not int
            or transition['resume_epoch'] < 1
            or not transition.get('old_source_sha256')
            or transition.get('new_source_sha256') != configuration.get('source_sha256')
            or not configuration.get('source_sha256')):
        raise ValueError('Policy transition does not bind the epoch and source inventories')
    old_sources, new_sources = transition['old_source_sha256'], transition['new_source_sha256']
    allowed_sources = {'code/train_hardval.py', 'code/train_hardval_distributed.py', 'code/fulltest_policy.py'}
    if (not isinstance(old_sources, dict) or not isinstance(new_sources, dict)
            or any(old_sources.get(key) != new_sources.get(key)
                   for key in set(old_sources) | set(new_sources) if key not in allowed_sources)
            or previous is not None and old_sources != previous.get('source_sha256')):
        raise ValueError('Policy transition changed non-training sources or its previous identity')
    return transition['resume_epoch']


def _validate_full_metadata(record, *, summary=False):
    if (record.get('selection_policy') != FULL_TEST_POLICY
            or record.get('best_metric') != 'weighted_score'
            or record.get('validation_performed') is not False
            or record.get('early_stopping_enabled') is not False
            or record.get('hard_validation_enabled') is not False
            or record.get('full_test_every_epochs') != 5
            or record.get('status') == 'early_stopped'
            or record.get('checkpoint_selection_uses_test') is not True):
        raise ValueError('Full-test policy must disable legacy validation and early stopping')
    for key in ('best_epoch', 'best_miou', 'best_validation_score'):
        if key not in record or record[key] is not None:
            raise ValueError('Full-test trainer must not publish a validation-selected best')
    for key in ('validation_miou', 'validation_weighted_score', 'miou',
                'validation_interval', 'latest_validation_epoch'):
        if record.get(key) is not None:
            raise ValueError('Full-test trainer must not publish new hard-validation scores')
    early = record.get('early_stopping', {})
    if early and (early.get('patience') != 0 or early.get('stopped') is not False):
        raise ValueError('Full-test early-stopping state is inconsistent')
    if summary and record.get('official_test_used_for_selection') is not True:
        raise ValueError('Full-test summary must disclose test-set selection')


def _legacy_prefix(configuration, by_epoch, completed, document, identity):
    best, best_epoch = None, None
    for epoch in range(1, completed + 1):
        if epoch not in by_epoch: return None, None, f'waiting_for_history_epoch_{epoch:03d}'
        row = by_epoch[epoch]
        if row.get('selection_policy', LEGACY_POLICY) != LEGACY_POLICY:
            raise ValueError('Legacy history prefix unexpectedly changed selection policy')
        should_validate = epoch % 2 == 0 or epoch == configuration['epochs']
        if row.get('validation_performed') is not should_validate:
            raise ValueError('History validation cadence differs from every two epochs plus the final epoch')
        if should_validate:
            record = document(f'validation_epoch_{epoch:03d}.json')
            if record is None: return None, None, f'waiting_for_validation_epoch_{epoch:03d}'
            score = record.get('validation_weighted_score')
            if (record.get('epoch') != epoch or isinstance(score, bool)
                    or not isinstance(score, (float, int)) or not math.isfinite(score)
                    or row.get('validation_weighted_score') != score
                    or row.get('validation_miou') != record.get('miou')
                    or record.get('data_identity') != identity
                    or record.get('checkpoint_selection_uses_test') is not True):
                raise ValueError('Validation file, identity and training history disagree')
            if best is None or score > best: best, best_epoch = score, epoch
        elif row.get('validation_weighted_score') is not None or row.get('validation_miou') is not None:
            raise ValueError('A skipped validation epoch must not contain a newly measured score')
        if row.get('best_epoch') != best_epoch or row.get('best_miou') != best:
            raise ValueError('History best differs from the first strict weighted validation maximum')
    return best, best_epoch, None


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
        previous_config = mirror.strict_json(previous_configuration.read_text())
        if incoming != previous_config:
            allowed = {'selection_policy', 'early_stopping_patience', 'source_sha256', 'resume', 'config'}
            changed = {k for k in set(incoming) | set(previous_config)
                       if incoming.get(k) != previous_config.get(k)}
            if (incoming.get('selection_policy') != 'full_test_every5'
                    or incoming.get('early_stopping_patience') != 0
                    or previous_config.get('selection_policy', LEGACY_POLICY) != LEGACY_POLICY
                    or not changed <= allowed):
                raise ValueError('Remote run_config differs from the established mirror identity')
            _transition_epoch(document('policy_transition.json'), incoming, previous_config)
    if configuration is None: return None, 'waiting_for_run_config'
    original_only = configuration.get('protocol') == ORIGINAL_ONLY_TRAINING_PROTOCOL
    if (configuration.get('protocol') not in (PROTOCOL, ORIGINAL_ONLY_TRAINING_PROTOCOL)
            or configuration.get('validation_every') != 2):
        raise ValueError('This mirror only accepts the explicit hard1500 every-two-epoch protocol')
    if original_only and (configuration.get('selection_policy') != FULL_TEST_POLICY
                          or configuration.get('early_stopping_patience') != 0):
        raise ValueError('Original-only training requires full-test selection with early stopping disabled')
    summary = document('summary.json')
    if summary is None: return None, 'waiting_for_first_epoch_summary'
    completed = summary.get('completed_epochs')
    if type(completed) is not int or completed < 1:
        raise ValueError('Invalid completed training epoch')
    if (summary.get('checkpoint_selection_uses_test') is not True
            or summary.get('official_test_used_for_selection') is not True):
        raise ValueError('Summary does not declare hard-validation checkpoint selection')
    previous = local / 'summary.json'
    if previous.exists():
        previous_summary = mirror.strict_json(previous.read_text())
        if previous_summary.get('completed_epochs', -1) > completed:
            raise ValueError('Remote summary moved behind the previously published epoch')
        if previous_summary.get('data_identity') != summary.get('data_identity'):
            raise ValueError('Remote summary changed the established training/validation identity')
    history_file = files.get(Path('history.jsonl'), local / 'history.jsonl')
    if not history_file.is_file(): return None, 'waiting_for_history'
    history = [mirror.strict_json(line) for line in history_file.read_text().splitlines() if line.strip()]
    by_epoch = {}
    for row in history:
        epoch = row.get('epoch')
        if type(epoch) is not int or epoch < 1 or epoch in by_epoch:
            raise ValueError('Invalid or duplicate history epoch')
        by_epoch[epoch] = row
    full_mode = configuration.get('selection_policy') == FULL_TEST_POLICY
    summary_full = summary.get('selection_policy') == FULL_TEST_POLICY
    if summary_full and not full_mode:
        return None, 'waiting_for_full_test_run_config'
    if full_mode:
        if configuration.get('early_stopping_patience') != 0:
            raise ValueError('Full-test training cannot enable early stopping')
        if set(range(1, completed + 1)) - set(by_epoch):
            return None, 'waiting_for_complete_history'
        identity = summary.get('data_identity', {})
        if (identity.get('train', {}).get('image_count') != (8002 if original_only else 6000)
                or identity.get('val', {}).get('image_count') != 1500
                or not identity['val'].get('hard_manifest_sha256')):
            raise ValueError('Full-test policy must inherit the fixed training and hard-validation identities')
        if original_only:
            from hardval_data import validate_original_only_identity
            validate_original_only_identity(identity['train'])
            if summary.get('train_images') != 8002 or summary.get('validation_images') != 1500:
                raise ValueError('Original-only summary counts differ from the fixed data identities')
        transition = document('policy_transition.json')
        if original_only and transition is not None:
            raise ValueError('Original-only training starts under full-test policy without a legacy transition')
        boundary = _transition_epoch(transition, configuration) if transition is not None else 0
        legacy_count = min(boundary, completed)
        if not summary_full:
            if not transition or completed > boundary:
                raise ValueError('Legacy summary is outside the declared policy-transition boundary')
            # run_config is published on resume, before the first new epoch.
            # Its old summary remains a valid legacy boundary until that epoch
            # commits; never relabel its validation best as a full-test best.
        best, best_epoch, pending = _legacy_prefix(
            configuration, by_epoch, legacy_count, document, identity['val'])
        if pending: return None, pending
        for epoch in range(legacy_count + 1, completed + 1):
            _validate_full_metadata(by_epoch[epoch])
        if summary_full:
            if completed <= boundary:
                raise ValueError('A full-test summary cannot precede its first new-policy epoch')
            _validate_full_metadata(summary, summary=True)
            return summary, None
    else:
        best, best_epoch, pending = _legacy_prefix(
            configuration, by_epoch, completed, document, summary['data_identity']['val'])
        if pending: return None, pending
    if summary.get('best_metric') != 'validation_weighted_score':
        raise ValueError('Legacy summary must use validation_weighted_score')
    if summary.get('best_epoch') != best_epoch or summary.get('best_miou') != best:
        raise ValueError('Summary best differs from the completed weighted validation history')
    return summary, None


def main():
    mirror.validate_boundary = validate_boundary
    return mirror.main()


if __name__ == '__main__':
    raise SystemExit(main())
