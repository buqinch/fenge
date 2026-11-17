#!/usr/bin/env python3
"""Run an unchanged evaluator with a per-process CUDA allocator limit."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fraction', required=True, type=float)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not 0 < args.fraction <= 1 or not command:
        parser.error('A fraction in (0,1] and an evaluator script are required')
    import torch
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.fraction, 0)
    script = Path(command[0]).resolve()
    sys.argv = [str(script), *command[1:]]
    sys.path.insert(0, str(script.parent))
    print(f'CUDA allocator fraction={args.fraction}; evaluator={script}', flush=True)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
