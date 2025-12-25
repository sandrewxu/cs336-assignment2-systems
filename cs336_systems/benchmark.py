"""
Model benchmarking
Section 1.1: Profiling and Benchmarking
"""

from unittest.runner import _ResultClassType
import numpy as np
import time
import torch
import typer

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

model_parameters = {
    "small": {
    "d_model": 768,
    "d_ff": 3072,
    "num_layers": 12,
    "num_heads": 12,
    },
    "medium": {
    "d_model": 1024,
    "d_ff": 4096,
    "num_layers": 24,
    "num_heads": 16,
    }, 
    "large": {
    "d_model": 1280,
    "d_ff": 5120,
    "num_layers": 36,
    "num_heads": 20,
    }, 
    "xl": {
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
    },
    "2.7B": {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
    },
}

def benchmark_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    rope_theta: float,
    batch_size: int,
    warmup_steps: int,
    benchmark_steps: int,
    forward_only: bool,
    device: str,
) -> dict[str, float]:
    """
    Given hyperparameters, initialize a model. Generate a random batch of data.
    Run `warmup_steps` warmup steps, and time the execution of `n` steps.
    """
    # Initialize model given hyperparameters
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    # Generate random batch of data
    dataset = np.arange(100000)
    inputs, targets = get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device
    )
    # Run w warmup steps
    for _ in range(warmup_steps):
        outs = model(inputs)
        torch.cuda.synchronize()
        if not forward_only:
            loss = cross_entropy(inputs, targets)
            loss.backward()
            torch.cuda.synchronize()

    benchmark_times = np.zeros(benchmark_steps, dtype=np.float64)
    # Benchmark n steps
    for step in range(benchmark_steps):
        start_time = time.perf_counter()
        outs = model(inputs)
        torch.cuda.synchronize()
        if not forward_only:
            loss = cross_entropy(inputs, targets)
            loss.backward()
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        benchmark_times[step] = end_time - start_time

    results = {
        "steps": benchmark_steps,
        "avg": benchmark_times.mean(),
        "std": benchmark_times.std(), 
        "forward_only": forward_only,
    }
    return results

if __name__ == "__main__":
    results = {}
    for model in model_parameters:
        results[model] = benchmark_model(
        vocab_size=10000,
        context_length=1024,
        d_model=model_parameters[model]["d_model"],
        d_ff=model_parameters[model]["d_ff"],
        num_layers=model_parameters[model]["num_layers"],
        num_heads=model_parameters[model]["num_heads"],
        rope_theta=10000.0,
        batch_size=4,
        warmup_steps=5,
        benchmark_steps=10,
        forward_only=False,
        device="cuda",
    )
    print(results)
