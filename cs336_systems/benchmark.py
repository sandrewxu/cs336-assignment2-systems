"""
Model benchmarking
Section 1.1: Profiling and Benchmarking

Times forward and backward passes of a model, with ability to vary hyperparameters.
"""

from contextlib import nullcontext
from einops import einsum
from jaxtyping import Bool, Float
import json
import math
import numpy as np
import time
import torch
import torch.cuda.nvtx as nvtx
import typer
from typing import Optional

from cs336_basics.data import get_batch
import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, softmax
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys    d_k"],
    V: Float[torch.Tensor, " ... keys    d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    """
    Annotated scaled dot-product attention.
    """
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

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
    mixed_precision: bool,
    profile_memory: bool,
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
    optimizer = AdamW(model.parameters())
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

    model_context = (
        torch.autocast(device_type=device, dtype=torch.bfloat16) 
        if mixed_precision else nullcontext()
    )

    # Run w warmup steps
    with nvtx.range("warmup steps"):
        for _ in range(warmup_steps):
            with model_context:
                outs = model(inputs)
            sync()
            if not forward_only:
                loss = cross_entropy(outs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                sync()

    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    benchmark_times = np.zeros(benchmark_steps, dtype=np.float64)
    # Benchmark n steps
    with nvtx.range("benchmark steps"):
        for step in range(benchmark_steps):
            start_time = time.perf_counter()
            with nvtx.range("forward pass"):
                with model_context:
                    outs = model(inputs)
                sync()
            if not forward_only:
                with nvtx.range("backward pass"):
                    loss = cross_entropy(outs, targets)
                    loss.backward()
                    sync()
                with nvtx.range("optimizer step"):
                    optimizer.step()
                    optimizer.zero_grad()
                    sync()
            end_time = time.perf_counter()
            benchmark_times[step] = end_time - start_time
    if profile_memory:
        pass_str = "forward" if forward_only else "fullstep"
        mixed_str = "mixed" if mixed_precision else "fullprecision"
        torch.cuda.memory._dump_snapshot(f"memory_snapshot_{pass_str}_{mixed_str}_{context_length}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

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
    mixed_precision: bool = typer.Option(False, "--mixed-precision", help="mixed precision using bfloat16"),
    profile_memory: bool = typer.Option(False, "--profile-memory", help="memory profiling with PyTorch"),
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
            mixed_precision=mixed_precision,
            profile_memory=profile_memory,
        )

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        print(results)

if __name__ == "__main__":
    app()
