"""
Distributed Data Parallel Training.
"""

import numpy as np
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDP_overlap

def setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.barrier()
    dist.destroy_process_group()

def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def _infer_device(rank: int, backend: str) -> str:
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend='nccl' requires CUDA, but torch.cuda.is_available() is False.")
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("backend='nccl' requires at least one CUDA device.")
        local_rank = rank % device_count
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cpu"

def _allreduce_flat_grads(model: torch.nn.Module, world_size: int) -> None:
    """
    Naive DDP gradient synchronization by flattening all grads into a single tensor,
    all-reducing once, then unflattening back into per-parameter grads.
    """
    params_with_grad: list[torch.nn.Parameter] = []
    grads: list[torch.Tensor] = []
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        # This helper only supports dense grads (which is what we have for transformer training).
        if g.is_sparse:
            raise RuntimeError("Sparse gradients are not supported by _allreduce_flat_grads.")
        params_with_grad.append(p)
        grads.append(g)

    if not grads:
        return

    # Pack with torch internal utilities (dense only)
    flat = torch._utils._flatten_dense_tensors(grads)

    # Communicate (SUM then average)
    dist.all_reduce(flat)
    flat.div_(world_size)

    # Unpack back into existing grad tensors (preserves optimizer references)
    synced_grads = torch._utils._unflatten_dense_tensors(flat, grads)
    for p, synced_g in zip(params_with_grad, synced_grads):
        # copy_ into the existing grad tensor object
        p.grad.copy_(synced_g)

def _ddp_worker(
    rank,
    world_size,
    backend,
    vocab_size,
    context_length,
    rope_theta,
    global_batch_size,
    warmup_steps,
    benchmark_steps,
    print_every_step,
):
    """
    Per process.
    """
    setup(rank, world_size, backend=backend)

    # Device mapping
    device = _infer_device(rank=rank, backend=backend)

    xl = dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
    assert global_batch_size % world_size == 0
    local_bs = global_batch_size // world_size

    torch.manual_seed(rank)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=xl["d_model"],
        num_layers=xl["num_layers"],
        num_heads=xl["num_heads"],
        d_ff=xl["d_ff"],
        rope_theta=rope_theta,
    ).to(device)

    ddp_model = DDP_overlap(model)
    ddp_model.train()
    optimizer = AdamW(ddp_model.parameters())

    # # broadcast initial params from rank 0
    # with torch.inference_mode():
    #     for p in model.parameters():
    #         dist.broadcast(p.data, src=0)

    # model.train()
    # optimizer = AdamW(model.parameters())

    dataset = np.arange(vocab_size)
    torch.manual_seed(0)
    full_inputs, full_targets = get_batch(
        dataset=dataset,
        batch_size=global_batch_size,
        context_length=context_length,
        device=device
    )

    # Shard batch
    offset = rank * local_bs
    inputs = full_inputs[offset : offset + local_bs]
    targets = full_targets[offset : offset + local_bs]

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        # outs = model(inputs)
        outs = ddp_model(inputs)
        loss = cross_entropy(outs, targets)
        loss.backward()
        _sync(device)
        ddp_model.finish_gradient_synchronization()
        _sync(device)

        # # all_reduce and average grads (flat buffer)
        # with torch.no_grad():
        #     _allreduce_flat_grads(model=model, world_size=world_size)

        optimizer.step()
        _sync(device)

    # Timed steps
    dist.barrier()
    _sync(device)
    step_times = []
    comm_times = []
    for _ in range(benchmark_steps):
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        step_start = time.perf_counter()

        # outs = model(inputs)
        outs = ddp_model(inputs)
        loss = cross_entropy(outs, targets)
        loss.backward()

        _sync(device)
        comm_start = time.perf_counter()
        ddp_model.finish_gradient_synchronization()
        # with torch.no_grad():
        #     _allreduce_flat_grads(model=model, world_size=world_size)
        _sync(device)
        comm_end = time.perf_counter()

        optimizer.step()
        _sync(device)
        step_end = time.perf_counter()

        step_times.append(step_end - step_start)
        comm_times.append(comm_end - comm_start)

        if print_every_step and rank == 0:
            print(
                {
                    "step_time_s": float(step_times[-1]),
                    "comm_time_s": float(comm_times[-1]),
                    "comm_frac": float(comm_times[-1] / step_times[-1]) if step_times[-1] > 0 else float("nan"),
                }
            )

    if rank == 0:
        step_arr = np.array(step_times, dtype=np.float64)
        comm_arr = np.array(comm_times, dtype=np.float64)
        comm_frac = comm_arr / step_arr
        print(
            {
                "model": "xl",
                "backend": backend,
                "world_size": world_size,
                "global_batch_size": global_batch_size,
                "warmup_steps": warmup_steps,
                "benchmark_steps": benchmark_steps,
                "step_time_avg_s": float(step_arr.mean()),
                "step_time_std_s": float(step_arr.std()),
                "comm_time_avg_s": float(comm_arr.mean()),
                "comm_time_std_s": float(comm_arr.std()),
                "comm_frac_avg": float(comm_frac.mean()),
                "comm_frac_std": float(comm_frac.std()),
            }
        )

    cleanup()

def benchmark_xl_naive_ddp(
    world_size=2,
    backend="nccl",
    vocab_size=10000,
    context_length=256,
    rope_theta=10000.0,
    global_batch_size=4,
    warmup_steps=5,
    benchmark_steps=10,
    print_every_step=False,
):
    mp.spawn(
        _ddp_worker,
        args=(
            world_size,
            backend,
            vocab_size,
            context_length,
            rope_theta,
            global_batch_size,
            warmup_steps,
            benchmark_steps,
            print_every_step,
        ),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    benchmark_xl_naive_ddp(world_size=2, backend="nccl")
