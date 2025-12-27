"""
Benchmarking attention.
"""

import math
import pandas as pd
import time
import torch
import triton
import triton.testing
from typing import Callable
import torch.nn as nn

from cs336_basics.model import scaled_dot_product_attention as naive_attention
from cs336_systems.flash_attention import FlashAttention2
torch._dynamo.config.cache_size_limit = 128
torch.set_float32_matmul_precision('high')

def benchmark_attention():
    configs = []
    seq_lens = [2**i for i in range(7, 17)] # 128 to 65536
    d_models = [16, 32, 64, 128]
    precisions = [torch.float32, torch.bfloat16]
    batch_size = 1
    device = "cuda"

    results = []

    for dtype in precisions:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "f32"
        for d_model in d_models:
            for seq_len in seq_lens:
                # Randomly generate Q, K, V
                q = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, requires_grad=True)
                k = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, requires_grad=True)
                v = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, requires_grad=True)
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()

                implementations = [
                    ("Triton-FA2", FlashAttention2.apply, True),
                    ("PyTorch-Native", naive_attention, causal_mask),
                ]

                row = {"seq_len": seq_len, "d_model": d_model, "dtype": dtype_str}

                for name, fn, arg in implementations:
                    try:
                        torch.cuda.empty_cache()

                        def fwd():
                            return fn(q, k, v, arg)

                        ms_fwd = triton.testing.do_bench(fwd)
                        output = fwd()
                        do = torch.randn_like(output)

                        def bwd():
                            output.backward(do, retain_graph=True)

                        ms_bwd = triton.testing.do_bench(bwd)

                        def e2e():
                            out = fn(q, k, v, arg)
                            out.backward(do)

                        ms_e2e = triton.testing.do_bench(e2e)

                        row[f"{name}_fwd"] = ms_fwd
                        row[f"{name}_bwd"] = ms_bwd
                        row[f"{name}_e2e"] = ms_e2e
                    except Exception as e:
                        print(f"Error {name} @ {seq_len}, {d_model}, {dtype_str}: {e}")
                        row[f"{name}_fwd"] = float('nan')
                        row[f"{name}_bwd"] = float('nan')
                        row[f"{name}_e2e"] = float('nan')
                results.append(row)
                print(f"Finished: seq={seq_len}, d={d_model}, dtype={row['dtype']}")

    df = pd.DataFrame(results)
    # Clean output formatting
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print("\n" + "="*80)
    print("FINAL BENCHMARK RESULTS (LATENCY IN MS)")
    print("="*80)
    print(df)

    # Save to CSV for the deliverable report
    df.to_csv("flash_attention_benchmark_results.csv", index=False)

if __name__ == "__main__":
    benchmark_attention()
