#!/usr/bin/env python3
"""Exercise the real hard-validation trainer, including an odd-epoch resume.

Run under torchrun with the normal training CLI. The optional injected stop
happens after a durable epoch summary; it is for small, explicitly partial
execution checks only and does not change production source files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import train_hardval_distributed as trainer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--interrupt-after-epoch', type=int)
    parser.add_argument('--record-memory', action='store_true')
    options, remaining = parser.parse_known_args()
    args = trainer.base.parse_args(remaining)
    if args.max_train_samples is None or args.max_val_samples is None:
        raise ValueError('Runtime verification requires explicit partial train/val sizes')
    if options.interrupt_after_epoch is not None:
        original_write = trainer.base.write_json

        def write_and_interrupt(path, value):
            original_write(path, value)
            if (Path(path).name == 'summary.json'
                    and value.get('completed_epochs') == options.interrupt_after_epoch):
                raise RuntimeError('verification_injected_epoch_boundary_stop')

        trainer.base.write_json = write_and_interrupt
    status = 'failed'
    try:
        trainer.main(remaining)
        status = 'complete'
    except RuntimeError as error:
        if 'verification_injected_epoch_boundary_stop' not in str(error):
            raise
        status = 'expected_interruption'
        print('Expected verification stop after durable epoch checkpoint.', flush=True)
    finally:
        if options.record_memory and torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            rank = int(os.environ.get('RANK', '0'))
            record = {'status': status, 'rank': rank, 'device': device,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'per_rank_batch_size': args.batch_size, 'input_size': args.input_size,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'scope': 'entire partial training process including validation and optimizer state',
                'other_process_memory_included': False}
            path = Path(args.output) / f'runtime_memory_rank_{rank:02d}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
