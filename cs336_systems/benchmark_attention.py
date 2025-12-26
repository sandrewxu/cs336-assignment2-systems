"""
Benchmarking attention.
"""

import math
import time
import torch
import torch.nn as nn

from cs336_basics.model import scaled_dot_product_attention as naive_attention
compiled_attention = torch.compile(naive_attention)

batch_size = 8
warmup_steps = 5
benchmark_steps = 100
device = "cuda"
attention_type = compiled_attention

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")

results = []

for d_model in [16, 32, 64, 128]:
    for seq_len in [256, 1024, 4096, 8192, 16384]:
        print(f"Benchmarking d_model={d_model}, seq_len={seq_len} with {attention_type.__name__}...")

        try:
            torch.cuda.empty_cache()

            # Create random Q, K, V
            Q = torch.empty((batch_size, seq_len, d_model), device=device, requires_grad=True)
            K = torch.empty((batch_size, seq_len, d_model), device=device, requires_grad=True)
            V = torch.empty((batch_size, seq_len, d_model), device=device, requires_grad=True)
            std = math.sqrt(2 / (seq_len + d_model))
            nn.init.trunc_normal_(Q, std=std, a=-3*std, b=3*std)
            nn.init.trunc_normal_(K, std=std, a=-3*std, b=3*std)
            nn.init.trunc_normal_(V, std=std, a=-3*std, b=3*std)

            # Warmup
            for _ in range(warmup_steps):
                output = attention_type(Q, K, V)
                torch.cuda.synchronize()
                # Dummy backward
                loss = output.sum()
                loss.backward()
                torch.cuda.synchronize()
                # Zero gradients
                Q.grad = None
                K.grad = None
                V.grad = None

            # Time 100 forward passes
            forward_times = []
            for i in range(benchmark_steps):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                output = attention_type(Q, K, V)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                forward_times.append(end_time-start_time)

            avg_forward_time = sum(forward_times) / len(forward_times)

            # Measure memory before backward pass
            # Create a fresh output for backward pass
            output = attention_type(Q, K, V)
            torch.cuda.synchronize()
            memory_before_backward = torch.cuda.memory_allocated(device) / (1024**3)    # Convert to GB

            # Time 100 backward passes
            backward_times = []
            for i in range(benchmark_steps):
                # Recompute output for each backward pass (need fresh computational graph)
                # Detach and reattach to create fresh graph
                Q_detached = Q.detach().requires_grad_(True)
                K_detached = K.detach().requires_grad_(True)
                V_detached = V.detach().requires_grad_(True)

                output = attention_type(Q_detached, K_detached, V_detached)
                loss = output.sum()

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                backward_times.append(end_time - start_time)

                # Update Q, K, V for next iteration (copy detached versions)
                Q = Q_detached
                K = K_detached
                V = V_detached

                # Zero gradients
                Q.grad = None
                K.grad = None
                V.grad = None

            avg_backward_time = sum(backward_times) / len(backward_times)

            results.append({
                "d_model": d_model,
                "seq_len": seq_len,
                "forward_time_ms": avg_forward_time * 1000,
                "backward_time_ms": avg_backward_time * 1000,
                "memory_before_backward_gb": memory_before_backward,
                "status": "success"
            })

        except RuntimeError as e:
            if "out of memory" in str(e):
                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_time_ms": None,
                    "backward_time_ms": None,
                    "memory_before_backward_gb": None,
                    "status": "OOM",
                })
                print(f"  Out of memory!")
                torch.cuda.empty_cache()
            else:
                print(f"  Error: {e}", flush=True)
        except Exception as e:
            # Catch any other exceptions
            print(f"  Unexpected error: {e}", flush=True)
            results.append({
                "d_model": d_model,
                "seq_len": seq_len,
                "forward_time_ms": None,
                "backward_time_ms": None,
                "memory_before_backward_gb": None,
                "status": "ERROR"
            })
            torch.cuda.empty_cache()

# Print results table
print("\n" + "="*80)
print("RESULTS TABLE")
print("="*80)
print(f"{'d_model':<10} {'seq_len':<10} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Memory (GB)':<15} {'Status':<10}")
print("-"*80)
for r in results:
    forward_str = f"{r['forward_time_ms']:.3f}" if r['forward_time_ms'] is not None else "OOM"
    backward_str = f"{r['backward_time_ms']:.3f}" if r['backward_time_ms'] is not None else "OOM"
    memory_str = f"{r['memory_before_backward_gb']:.3f}" if r['memory_before_backward_gb'] is not None else "N/A"
    print(f"{r['d_model']:<10} {r['seq_len']:<10} {forward_str:<15} {backward_str:<15} {memory_str:<15} {r['status']:<10}")
