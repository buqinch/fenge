"""Explicit continuation policy and immutable current-epoch test candidates."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch

LEGACY_POLICY = 'hard_validation_best'
FULL_TEST_POLICY = 'full_test_every5'
TRANSITION_SCHEMA = 'lusnar_training_policy_transition_v1'
REMOVED_INFERENCE_KEYS = frozenset({
    'optimizer', 'scheduler', 'scaler', 'rng', 'rank_states',
    'loader_generator', 'val_generator', 'early_stopping',
})


def policy_of(configuration):
    return configuration.get('selection_policy', LEGACY_POLICY)


def full_test_enabled(args):
    return getattr(args, 'selection_policy', LEGACY_POLICY) == FULL_TEST_POLICY


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_transition(saved, configuration):
    """Return an audit only for the one authorized legacy-to-full transition."""
    old = saved['train_config']
    before, after = policy_of(old), policy_of(configuration)
    if after == FULL_TEST_POLICY and configuration.get('early_stopping_patience') != 0:
        raise ValueError('Full-test policy requires early_stopping_patience=0')
    if saved.get('inference_only') is True:
        raise ValueError('An inference-only test candidate cannot resume training')
    if saved.get('selection_policy', before) != before:
        raise ValueError('Checkpoint selection policy disagrees with its training configuration')
    if before == after:
        return None
    if before != LEGACY_POLICY or after != FULL_TEST_POLICY:
        raise ValueError('Only hard-validation to full-test policy migration is supported')
    if configuration.get('early_stopping_patience') != 0:
        raise ValueError('Full-test policy requires early_stopping_patience=0')
    resume = Path(configuration['resume'])
    if resume.name != 'last.pt' or resume.resolve().parent != Path(configuration['output']).resolve():
        raise ValueError('Policy migration requires the original run rolling last.pt')
    path = Path(configuration['output']) / 'policy_transition.json'
    document = json.loads(path.read_text())
    if (document.get('schema') != TRANSITION_SCHEMA
            or document.get('from_policy') != before or document.get('to_policy') != after
            or document.get('resume_checkpoint_sha256') != file_sha256(resume)
            or document.get('old_source_sha256') != old.get('source_sha256')
            or document.get('new_source_sha256') != configuration.get('source_sha256')
            or not old.get('source_sha256') or not configuration.get('source_sha256')):
        raise ValueError('Policy transition is not bound to this checkpoint and both source inventories')
    if 'resume_epoch' in document and document['resume_epoch'] != saved['epoch']:
        raise ValueError('Policy transition checkpoint epoch differs')
    allowed_sources = {'code/train_hardval.py', 'code/train_hardval_distributed.py', 'code/fulltest_policy.py'}
    old_sources, new_sources = old['source_sha256'], configuration['source_sha256']
    if any(old_sources.get(key) != new_sources.get(key)
           for key in set(old_sources) | set(new_sources) if key not in allowed_sources):
        raise ValueError('Policy migration cannot change model, data, loss or optimizer implementation sources')
    return {**copy.deepcopy(document), 'transition_file_sha256': file_sha256(path),
            'checkpoint_epoch': saved['epoch'], 'global_update': saved['global_update'],
            'legacy_validation_best': {'epoch': saved.get('best_epoch'),
                                       'score': saved.get('best_miou'),
                                       'metric': 'validation_weighted_score'},
            'legacy_early_stopping': copy.deepcopy(saved.get('early_stopping')),
            'state_continuity': 'model, optimizer, scheduler, scaler and per-rank RNG restored unchanged',
            'protocol_change': 'hard-validation and early stopping disabled; current epoch tested every five epochs'}


def persist_transition_audit(output, migration, previous_config, write_json):
    if migration is None:
        return
    for name, value in [('run_config_before_policy_transition.json', previous_config),
                        ('policy_transition_applied.json', migration)]:
        path = Path(output) / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError('Existing policy transition audit differs: ' + str(path))
        else:
            write_json(path, value)


def policy_metadata(args, epoch, transition=None):
    if not full_test_enabled(args):
        return {'selection_policy': LEGACY_POLICY,
                'early_stopping_enabled': args.early_stopping_patience > 0}
    candidate = epoch % 5 == 0 or epoch == args.epochs
    return {'selection_policy': FULL_TEST_POLICY, 'early_stopping_enabled': False,
            'hard_validation_enabled': False, 'validation_interval': None,
            'latest_validation_epoch': None, 'early_stopping_patience_unit': None,
            'best_epoch': None, 'best_miou': None, 'best_validation_score': None,
            'best_metric': 'weighted_score',
            'best_miou_definition': 'not selected by trainer; full-test best is managed externally',
            'selection_scene': 'full_Moon_2_4_9', 'full_test_every_epochs': 5,
            'selected_by_test': False, 'full_test_candidate': candidate,
            'checkpoint_role': 'full_test_candidate' if candidate else 'rolling_training_state',
            'legacy_validation_best': (transition or {}).get('legacy_validation_best'),
            'policy_transition': transition}


def inference_checkpoint(checkpoint):
    """Retain model tensor references; do not clone the model onto another GPU."""
    result = {key: value for key, value in checkpoint.items()
              if key not in REMOVED_INFERENCE_KEYS and not key.startswith('_')}
    result.update(inference_only=True, checkpoint_role='full_test_candidate', selected_by_test=False)
    return result


def _receipt_identity(checkpoint):
    hashes = checkpoint.get('model_tensor_sha256_per_rank')
    if hashes and len(set(hashes)) == 1:
        model_hash = hashes[0]
    else:
        digest = hashlib.sha256()
        for name, tensor in sorted(checkpoint['model'].items()):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(name.encode() + b'\0' + str(tensor.dtype).encode() + b'\0')
            digest.update(json.dumps(list(tensor.shape)).encode() + b'\0')
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        model_hash = digest.hexdigest()
    return {'epoch': checkpoint['epoch'], 'global_update': checkpoint['global_update'],
            'model_config': checkpoint['model_config'], 'data_identity': checkpoint['data_identity'],
            'source_sha256': checkpoint['train_config']['source_sha256'],
            'selection_policy': FULL_TEST_POLICY, 'model_tensor_sha256': model_hash}


def _completed_evaluation(receipt):
    evaluation = receipt.get('evaluation', {})
    path = Path(evaluation.get('metrics_path') or '')
    return (evaluation.get('status') == 'complete'
            and evaluation.get('checkpoint_sha256') == receipt.get('checkpoint_sha256')
            and bool(evaluation.get('metrics_sha256')) and path.is_file()
            and file_sha256(path) == evaluation['metrics_sha256'])


def _write_receipt(path, receipt):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def publish_checkpoints(output, checkpoint, total_epochs, atomic_save, *, recovering=False):
    """Called after rolling last.pt and before publishing the epoch summary."""
    if policy_of(checkpoint['train_config']) != FULL_TEST_POLICY:
        return
    output, epoch = Path(output), checkpoint['epoch']
    if epoch % 5 == 0 or epoch == total_epochs:
        destination = output / 'full_test_checkpoints' / f'epoch_{epoch:03d}.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = destination.with_suffix('.json')
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        identity = _receipt_identity(checkpoint)
        if receipt and any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError('Full-test receipt differs from the resumed model or configuration')
        completed_removed = recovering and not destination.exists() and receipt and _completed_evaluation(receipt)
        if destination.exists():
            if not recovering:
                raise FileExistsError('Refusing to replace an immutable full-test candidate: ' + str(destination))
            if receipt:
                if receipt.get('checkpoint_sha256') != file_sha256(destination):
                    raise ValueError('Existing immutable candidate bytes differ from its receipt')
            else:
                existing = torch.load(destination, map_location='cpu', weights_only=True)
                keys = ('epoch', 'global_update', 'model_config', 'data_identity', 'selection_policy')
                if (existing.get('inference_only') is not True
                        or any(existing.get(key) != checkpoint.get(key) for key in keys)
                        or existing.get('train_config', {}).get('source_sha256') != checkpoint['train_config'].get('source_sha256')
                        or set(existing['model']) != set(checkpoint['model'])
                        or any(not torch.equal(existing['model'][key], value.detach().cpu())
                               for key, value in checkpoint['model'].items())):
                    raise ValueError('Existing immutable candidate differs from the resumed model')
        elif not completed_removed:
            if receipt and not recovering:
                raise FileExistsError('Refusing to replace a previously published full-test candidate')
            atomic_save(destination, inference_checkpoint(checkpoint))
            if receipt:
                receipt = {'reconstruction_history': [*receipt.get('reconstruction_history', []),
                    {'checkpoint_sha256': receipt.get('checkpoint_sha256'),
                     'evaluation': receipt.get('evaluation'),
                     'reason': 'candidate missing without valid completed evaluation proof; restored from rolling last'}]}
        if not completed_removed and (receipt is None or 'checkpoint_sha256' not in receipt):
            receipt = {**(receipt or {}), **identity, 'schema': 'lusnar_full_test_candidate_receipt_v1',
                       'checkpoint_path': str(destination), 'checkpoint_sha256': file_sha256(destination),
                       'checkpoint_bytes': destination.stat().st_size, 'inference_only': True,
                       'selected_by_test': False}
            _write_receipt(receipt_path, receipt)
    if epoch == total_epochs:
        destination = output / 'final.pt'
        temporary = output / '.final.pt.tmp'
        temporary.unlink(missing_ok=True)
        try:
            os.link(output / 'last.pt', temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def recover_policy_outputs(output, saved, args, configuration, transition, tracker_state, atomic_save):
    """Repair a crash after last.pt, preserving both its bytes and history prefix."""
    if not full_test_enabled(args):
        return
    checkpoint = {key: value for key, value in saved.items() if not key.startswith('_')}
    checkpoint.update(train_config=configuration, run_config=configuration,
                      early_stopping=tracker_state, metrics=None, validation_performed=False)
    checkpoint.update(policy_metadata(args, saved['epoch'], transition))
    # A completed old-policy run requires a distinct evaluation task, not a
    # continuation which would relabel its already-complete final checkpoint.
    if policy_of(saved['train_config']) != FULL_TEST_POLICY and saved['epoch'] >= args.epochs:
        raise ValueError('Policy continuation requires at least one remaining training epoch')
    publish_checkpoints(output, checkpoint, args.epochs, atomic_save, recovering=True)
