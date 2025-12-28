"""
Benchmarking the runtime of the all-reduce operation in the single-node multi-process setup.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def benchmark_all_reduce(rank, world_size, backend, data_size_mb):
    """
    Benchmark all-reduce in a single-node multi-process setup. Vary between
    backends: Gloo, NCCL
    data size: 1MB, 10MB, 100MB, 1GB
    processes: 2, 4, 6

    5 warmup steps
    10 benchmark steps
    """
    # Determine device
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(device)

    setup(rank, world_size, backend)

    # number of float32 elements (4 bytes each)
    num_elements = (data_size_mb * 1024 * 1024) // 4
    data = torch.randn(num_elements, device=device)

    # Warmup
    for _ in range(5):
        dist.all_reduce(data)

    # Timing
    iters = 10
    start_time = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(data)
    if backend == "nccl":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / iters

    if rank == 0:
        print(f"{backend} | {world_size} workers | {data_size_mb}MB | Time: {elapsed_time:.6f}s")

    cleanup()

if __name__ == "__main__":
    backends = ["gloo", "nccl"]
    world_sizes = [2, 4, 6]
    data_sizes_mb = [1, 10, 100, 1000]  # 1 MB to 1 GB

    for backend in backends:
        if backend == "nccl" and not torch.cuda.is_available():
            print("Skipping NCCL: CUDA not available.")
            continue
        for ws in world_sizes:
            for size in data_sizes_mb:
                mp.spawn(
                    fn=benchmark_all_reduce,
                    args=(ws, backend, size),
                    nprocs=ws,
                    join=True,
                )
