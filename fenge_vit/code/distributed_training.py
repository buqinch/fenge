"""Synchronous data-parallel primitives for the independent ViT trainer.

No model wrapping is needed: gradients are bucket-reduced after both intact and
optional masked backward passes. This supports frozen backbone transitions and
parameters unused by the masked semantic branch without reducer hook ambiguity.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import Sampler

from vitseg.metrics import (ConfusionMatrix, autocast_context, predict_semantic_scores,
                            tta_metadata, _complete_official_test)


def world_info():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def reduce_sum(value):
    result = value.clone()
    if world_info()[1] > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def gather_objects(value):
    world = world_info()[1]
    if world == 1:
        return [value]
    # c10d may mutate its internally allocated tensors outside the caller's
    # inference-mode context; allocate ordinary tensors for object collectives.
    with torch.inference_mode(False):
        result = [None] * world
        dist.all_gather_object(result, value)
    return result


def broadcast_object(value, src=0):
    if world_info()[1] == 1:
        return value
    with torch.inference_mode(False):
        values = [value]
        dist.broadcast_object_list(values, src=src)
    return values[0]


def assert_all_finite(value, label):
    flag = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if world_info()[1] > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag):
        raise FloatingPointError(f'Non-finite {label} on at least one distributed rank')


class NoPaddingShardSampler(Sampler):
    """Disjoint stride shards of one deterministic permutation, without repeats.

    Training requires equal shard lengths so all ranks execute collectives in the
    same order. Validation permits unequal lengths (including empty shards).
    """
    def __init__(self, dataset, rank=None, world_size=None, shuffle=False, seed=42,
                 require_equal=False):
        actual_rank, actual_world = world_info()
        self.rank = actual_rank if rank is None else rank
        self.world_size = actual_world if world_size is None else world_size
        if type(self.world_size) is not int or self.world_size < 1 or type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise ValueError('Invalid distributed shard rank/world size')
        if require_equal and len(dataset) % self.world_size:
            raise ValueError('Training image count must be divisible by world size; padding/repeated samples are forbidden')
        self.dataset, self.shuffle, self.seed, self.epoch = dataset, bool(shuffle), int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices[self.rank::self.world_size])

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))


def distributed_set_loss(outputs, targets, criterion):
    """Global Mask2Former objective under averaged gradient synchronization.

    Cross-entropy uses its target-weight denominator (including no-object 0.1),
    while BCE/Dice use matched-mask count. Averaging unweighted rank means would
    be incorrect when ranks contain different numbers of GT classes. Every
    auxiliary stage has the same number of matched classes but its own matching.
    The returned rank losses are multiplied by world size so their rank average
    and the averaged parameter gradients equal the global microbatch objective.
    """
    losses = criterion(outputs, targets)
    world = world_info()[1]
    queries = outputs['pred_logits'].shape[1]
    valid_images = 0
    masks = 0
    for target in targets:
        classes = target[target != criterion.ignore_index].unique()
        if len(classes):
            valid_images += 1
            masks += len(classes)
    no_object = float(criterion.no_object_weights[-1].detach())
    class_denominator = masks + (valid_images * queries - masks) * no_object
    local = torch.tensor([class_denominator, masks], dtype=torch.float64, device=targets.device)
    global_counts = reduce_sum(local)
    class_scale = world * local[0] / global_counts[0].clamp_min(1e-30)
    mask_scale = world * local[1] / global_counts[1].clamp_min(1)
    result = {}
    total = outputs['pred_logits'].sum() * 0.0 + outputs['pred_masks'].sum() * 0.0
    auxiliary = total
    suffixes = [''] + [f'_{i}' for i in range(len(outputs.get('aux_outputs', [])))]
    for suffix in suffixes:
        for name, scale in [('loss_class', class_scale), ('loss_mask', mask_scale), ('loss_dice', mask_scale)]:
            result[name + suffix] = losses[name + suffix] * scale.to(losses[name + suffix].dtype)
        weighted = (criterion.class_weight * result['loss_class' + suffix]
                    + criterion.mask_weight * result['loss_mask' + suffix]
                    + criterion.dice_weight * result['loss_dice' + suffix])
        if suffix:
            auxiliary = auxiliary + weighted
        else:
            total = total + weighted
    result['loss_auxiliary'] = auxiliary
    result['loss_total'] = total + auxiliary
    return result



def distributed_masked_loss(local_loss, local_denominator, global_denominator):
    """Scale one local NLL for world-averaged gradients over a whole window."""
    if local_denominator < 0 or global_denominator < local_denominator:
        raise ValueError('Invalid local/global masked support')
    return local_loss * (world_info()[1] * local_denominator / max(global_denominator, 1))


def synchronize_gradients(model, bucket_bytes=25 * 1024 * 1024):
    """Average globally used parameter gradients; preserve globally unused None.

    Parameters missing a gradient on one rank contribute zeros, while a parameter
    unused everywhere retains grad=None so AdamW does not spuriously decay it.
    """
    world = world_info()[1]
    if world == 1:
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return
    device = parameters[0].device
    flags = torch.tensor([int(p.grad is not None) for p in parameters], device=device, dtype=torch.int32)
    dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    active = [p for p, count in zip(parameters, flags.tolist()) if count]
    buckets, bucket, current_bytes, current_dtype = [], [], 0, None
    for parameter in active:
        if parameter.grad is not None and parameter.grad.is_sparse:
            raise ValueError('Sparse gradients are not supported by this trainer')
        size = parameter.numel() * parameter.element_size()
        if bucket and (current_bytes + size > bucket_bytes or parameter.dtype != current_dtype):
            buckets.append(bucket)
            bucket, current_bytes = [], 0
        bucket.append(parameter)
        current_bytes += size
        current_dtype = parameter.dtype
    if bucket:
        buckets.append(bucket)
    for bucket in buckets:
        values = [p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
                  for p in bucket]
        flat = torch.cat(values)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for parameter in bucket:
            gradient = flat[offset:offset + parameter.numel()].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.copy_(gradient)
            offset += parameter.numel()


def broadcast_model(model, src=0):
    if world_info()[1] > 1:
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=src)


@torch.inference_mode()
def distributed_evaluate(model, loader, device, amp='bf16', tta='none', tta_sizes=None,
                         tta_weights=None, progress_every=50):
    """Evaluate unequal no-padding shards and combine integer confusion counts."""
    device = torch.device(device)
    rank, world = world_info()
    metadata = tta_metadata(tta, tta_sizes, tta_weights)
    total, scenes, seen, count, batches = ConfusionMatrix(), {}, [], 0, 0
    native_grid, was_training = True, model.training
    model.eval()
    try:
        for batch in loader:
            images, targets = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
            native_grid = native_grid and tuple(targets.shape[-2:]) == (1024, 1024)
            with autocast_context(device, amp):
                scores = predict_semantic_scores(model, images, tta, tta_sizes, tta_weights)
            if scores.shape[-2:] != targets.shape[-2:]:
                scores = F.interpolate(scores.float(), targets.shape[-2:], mode='bilinear', align_corners=False)
            prediction = scores.argmax(1)
            total.update(targets, prediction)
            for index, scene in enumerate(batch['scene']):
                entry = scenes.setdefault(str(scene), {'matrix': ConfusionMatrix(), 'image_count': 0})
                entry['matrix'].update(targets[index], prediction[index])
                entry['image_count'] += 1
            seen.extend(str(x) for x in batch['source_id'])
            count += len(images)
            batches += 1
            if rank == 0 and progress_every and batches % progress_every == 0:
                print(f'Distributed validation rank 0: {count}/{len(loader.sampler)} shard images', flush=True)
    finally:
        model.train(was_training)
    payload = {'matrix': total.matrix.tolist(), 'scenes': {key: {'matrix': value['matrix'].matrix.tolist(),
               'image_count': value['image_count']} for key, value in scenes.items()},
               'source_ids': seen, 'image_count': count, 'batches': batches, 'native_grid': native_grid}
    parts = gather_objects(payload)
    all_sources = [source for part in parts for source in part['source_ids']]
    count = sum(part['image_count'] for part in parts)
    if len(all_sources) != count or len(set(all_sources)) != count:
        raise ValueError('Distributed validation repeated or omitted source identities')
    if count != len(loader.dataset):
        raise ValueError('Distributed validation does not cover the configured dataset exactly')
    total = ConfusionMatrix()
    scenes = {}
    for part in parts:
        total.matrix += np.asarray(part['matrix'], dtype=np.int64)
        for scene, entry in part['scenes'].items():
            merged = scenes.setdefault(scene, {'matrix': ConfusionMatrix(), 'image_count': 0})
            merged['matrix'].matrix += np.asarray(entry['matrix'], dtype=np.int64)
            merged['image_count'] += entry['image_count']
    if not np.array_equal(sum((entry['matrix'].matrix for entry in scenes.values()), np.zeros((5, 5), dtype=np.int64)), total.matrix):
        raise ValueError('Distributed scene matrices do not sum to the pooled matrix')
    identity = dict(getattr(loader.dataset, 'identity', {}))
    partial = bool(getattr(loader.dataset, 'partial', False))
    result = {**total.compute(), **metadata, 'image_count': count, 'partial': partial,
              'data_identity': identity, 'validation_unique_source_count': len(all_sources),
              'scenes': {scene: {**entry['matrix'].compute(), 'image_count': entry['image_count']}
                         for scene, entry in sorted(scenes.items())},
              'model_forward_calls': sum(part['batches'] for part in parts) * metadata['tta_forward_calls_per_batch'],
              'loader_passes': 1, 'weighted_score': None,
              'distributed_validation': {'world_size': world, 'shard_image_counts': [p['image_count'] for p in parts],
                                         'padding': False, 'duplicate_source_count': 0}}
    if identity.get('split') == 'test' or set(scenes) & {'Moon_2', 'Moon_4', 'Moon_9'}:
        complete = not partial and _complete_official_test(identity, count, len(all_sources),
            {scene: entry['image_count'] for scene, entry in scenes.items()}, all(p['native_grid'] for p in parts))
        result.update(test_scope='complete_official_test' if complete else 'partial_test',
                      official_test_scope=complete, partial=partial or not complete,
                      weighted_score_definition='Weighted mean of independently accumulated scene mIoUs; distinct from pooled mIoU',
                      weighted_score_weights={'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2},
                      weighted_score_formula='(Moon_2 mIoU + Moon_4 mIoU + 2*Moon_9 mIoU)/4')
        if complete:
            result['weighted_score'] = (scenes['Moon_2']['matrix'].compute()['miou']
                                       + scenes['Moon_4']['matrix'].compute()['miou']
                                       + 2 * scenes['Moon_9']['matrix'].compute()['miou']) / 4
    return result
