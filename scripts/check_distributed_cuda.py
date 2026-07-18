"""Fail unless Accelerate exposes the planned two-rank A40 topology."""

from __future__ import annotations

import json


def main() -> None:
    import torch
    from accelerate import Accelerator

    accelerator = Accelerator()
    if accelerator.num_processes != 2:
        raise RuntimeError(
            f"distributed CUDA probe requires 2 ranks, got {accelerator.num_processes}"
        )
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"distributed CUDA probe requires 2 visible GPUs, got {torch.cuda.device_count()}"
        )
    if torch.cuda.current_device() != accelerator.local_process_index:
        raise RuntimeError(
            "distributed CUDA rank does not own the matching local CUDA device"
        )
    device_name = str(torch.cuda.get_device_name(accelerator.local_process_index))
    if "A40" not in device_name.upper():
        raise RuntimeError(f"distributed CUDA probe requires A40, got {device_name!r}")
    accelerator.wait_for_everyone()
    print(
        json.dumps(
            {
                "world_size": accelerator.num_processes,
                "rank": accelerator.process_index,
                "local_rank": accelerator.local_process_index,
                "device": device_name,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
