"""
Model benchmarking
Section 1.1: Profiling and Benchmarking

Times forward and backward passes of a model, with ability to vary hyperparameters.
"""

import json
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
    ).to(device)
    model.train()
    # Generate random batch of data
    dataset = np.arange(vocab_size)
    inputs, targets = get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device
    )

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # Run w warmup steps
    for _ in range(warmup_steps):
        outs = model(inputs)
        sync()
        if not forward_only:
            loss = cross_entropy(outs, targets)
            loss.backward()
            model.zero_grad()
            sync()

    benchmark_times = np.zeros(benchmark_steps, dtype=np.float64)
    # Benchmark n steps
    for step in range(benchmark_steps):
        start_time = time.perf_counter()
        outs = model(inputs)
        sync()
        if not forward_only:
            loss = cross_entropy(outs, targets)
            loss.backward()
            model.zero_grad()
            sync()
        end_time = time.perf_counter()
        benchmark_times[step] = end_time - start_time

    results = {
        "forward_only": forward_only,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "avg": benchmark_times.mean(),
        "std": benchmark_times.std(), 
    }
    return results

app = typer.Typer(help="Benchmark transfomer models")

@app.command()
def main(
    models: list[str] = typer.Option(
        ["all"], "--model", "-m", help="Models to benchmark (can specify multiple)"
    ),
    vocab_size: int = typer.Option(10000, "--vocab-size"),
    context_length: int = typer.Option(256, "--context-length"),
    rope_theta: float = typer.Option(10000.0, "--rope-theta"),
    batch_size: int = typer.Option(4, "--batch_size"),
    warmup_steps: int = typer.Option(5, "--warmup-steps"),
    benchmark_steps: int = typer.Option(10, "--benchmark-steps"),
    forward_only: bool = typer.Option(False, "--forward-only", help="Only benchmark forward pass"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)"),
    output_json: str = typer.Option(None, "--output-json", help="Path to save JSON results"),
):
    """
    Benchmark transformer models with configurable hyperparameters.
    """
    if "all" in models:
        models_to_benchmark = list(model_parameters.keys())
    else:
        models_to_benchmark = models

    results = {}
    for model in models_to_benchmark:
        results[model] = benchmark_model(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=model_parameters[model]["d_model"],
            d_ff=model_parameters[model]["d_ff"],
            num_layers=model_parameters[model]["num_layers"],
            num_heads=model_parameters[model]["num_heads"],
            rope_theta=rope_theta,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
            forward_only=forward_only,
            device=device,
        )

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        print(results)

if __name__ == "__main__":
    app()
