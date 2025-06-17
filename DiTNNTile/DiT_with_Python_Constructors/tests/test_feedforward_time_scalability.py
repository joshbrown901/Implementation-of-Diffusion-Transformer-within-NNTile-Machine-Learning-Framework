import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits
from typing import List, Union
import torch
import torch.nn as nn
import pytest
from PointwiseFeedForwardNumpy import FeedForwardNumpy
from PointwiseFeedForwardTorch import FeedForwardTorch
from PointwiseFeedForwardNNTile import FeedForwardNNTile

import numpy as np
import torch
import time
import psutil
import os
import matplotlib.pyplot as plt
import gc
from torch._inductor import config as inductor_config

# Configuration
inductor_config.compile_threads = 1
torch.set_num_threads(1)
D = 2048       # Llama-1.3b hidden size
F = 5504       # Feed-forward hidden dimension
B = 1          # Batch size
SEQ_LENGTHS = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10000, 20000, 30000]
WARMUP_STEPS = 2
MEASURE_STEPS = 5
np.random.seed(0)
torch.manual_seed(0)
np_ff  = FeedForwardNumpy(D, F)


# Benchmark Functions
def benchmark_nntile_gpu(seq_len):
    nntile_config = nntile.starpu.Config(1,1,0,0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = FeedForwardNNTile(B, seq_len, F, D, np_ff.W1, np_ff.b1, np_ff.W2, np_ff.b2)
    
    # Create data directly on GPU
    x = np.random.randn(B, seq_len, D).astype(np.float32)
    grad_out = np.random.randn(B, seq_len, D).astype(np.float32)
    
    # Warmup
    for _ in range(WARMUP_STEPS):
        out = model.forward(x)
        dx = model.backward(grad_out)
    
    # Memory measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    
    for _ in range(MEASURE_STEPS):
        out = model.forward(x)
        dx = model.backward(grad_out)
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return elapsed, gpu_mem


def benchmark_pytorch_gpu(seq_len):
    """
    Benchmark a FeedForwardTorch model on GPU for a given sequence length,
    without torch.compile.

    Returns:
        elapsed (float): average time per iteration (seconds), or nan on error.
        gpu_mem (float): peak GPU memory used in GB, or nan on error.
    Raises:
        RuntimeError if NaNs are detected during warmup or measurement.
    """
    # Fix seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda")
    # Instantiate the model and move to GPU
    model = FeedForwardTorch(D, F).to(device)

    # Prepare input and grad_out on GPU
    x = torch.randn(B, seq_len, D, device=device, requires_grad=True)
    grad_out = torch.randn(B, seq_len, D, device=device)

    # Warmup iterations: stabilize caches, JIT kernels, etc.
    for _ in range(WARMUP_STEPS):
        out = model(x)
        # Check for NaNs
        if torch.isnan(out).any():
            raise RuntimeError(f"NaN detected in model output during warmup at seq_len={seq_len}")
        # Use a simple loss to ensure backward is exercised
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    # Prepare for measurement: synchronize and reset peak stats
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    # Timed iterations
    for _ in range(MEASURE_STEPS):
        out = model(x)
        if torch.isnan(out).any():
            raise RuntimeError(f"NaN detected in model output during measurement at seq_len={seq_len}")
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    # Synchronize to ensure all ops complete
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    # Peak memory in bytes → GB
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed, gpu_mem




def run_benchmark():
    """
    Run benchmarks for each sequence length in SEQ_LENGTHS.
    Returns a list of rows:
      [seq_len, nntile_time, torch_time, nntile_mem, torch_mem]
    """
    results = []
    for seq_len in SEQ_LENGTHS:
        print(f"Benchmarking seq_len={seq_len}")

        # NNTile GPU benchmark
        try:
            nntile_time, nntile_mem = benchmark_nntile_gpu(seq_len)
        except Exception as e:
            print(f"NNTile GPU error at seq_len={seq_len}: {e}")
            nntile_time, nntile_mem = float('nan'), float('nan')

        # PyTorch GPU (no compile)
        try:
            torch_time, torch_mem = benchmark_pytorch_gpu(seq_len)
        except Exception as e:
            print(f"PyTorch GPU error at seq_len={seq_len}: {e}")
            torch_time, torch_mem = float('nan'), float('nan')

        results.append([
            seq_len,
            nntile_time,
            torch_time,
            nntile_mem,
            torch_mem
        ])
    return results



def plot_results(results, save_path="gpu_benchmark.png"):
    """
    Plot time and memory vs. sequence length.
    Saves plot to save_path.
    """
    seq_lens = [r[0] for r in results]
    nntile_times = [r[1] for r in results]
    torch_times = [r[2] for r in results]
    nntile_mems = [r[3] for r in results]
    torch_mems = [r[4] for r in results]

    plt.figure(figsize=(10, 8))

    # Time subplot
    plt.subplot(2, 1, 1)
    plt.plot(seq_lens, nntile_times, 'o-', label='NNTile GPU')
    plt.plot(seq_lens, torch_times, 's-', label='PyTorch GPU')
    plt.title("FeedForward Execution Time (GPU)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Time per step (s)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    plt.legend()

    # Memory subplot
    plt.subplot(2, 1, 2)
    plt.plot(seq_lens, nntile_mems, 'o-', label='NNTile GPU')
    plt.plot(seq_lens, torch_mems, 's-', label='PyTorch GPU')
    plt.title("GPU Memory Usage")
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory (GB)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")



def print_results_table(results):
    """
    Print a Markdown-like table of the benchmark results.
    """
    print("\nBenchmark Results:")
    header = (
        "| Seqlen | NNTile (s) | PyTorch (s) | "
        "NNTile Mem (GB) | PyTorch Mem (GB) |"
    )
    sep = (
        "|--------|------------|-------------|"
        "-----------------|------------------|"
    )
    print(header)
    print(sep)
    for r in results:
        seq_len, n_time, t_time, n_mem, t_mem = r
        # Format floats or nan
        def fmt(x):
            return f"{x:.6f}" if not (isinstance(x, float) and np.isnan(x)) else "nan"
        print(f"| {seq_len:6d} | {fmt(n_time):>10} | {fmt(t_time):>11} | "
              f"{fmt(n_mem):>15} | {fmt(t_mem):>16} |")


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available! Exiting...")
        exit(1)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total GPU memory: {total_mem:.2f} GB")

    # Run benchmark
    results = run_benchmark()

    # Print and plot
    print_results_table(results)
    plot_results(results, save_path="gpu_benchmark.png")




# Configuration
inductor_config.compile_threads = 1
torch.set_num_threads(1)
D_MIN = 512      # Minimum hidden size
D_MAX = 6000     # Maximum hidden size
D_STEP = 512     # Step size for hidden size
F_RATIO = 2.6875 # F = F_RATIO * D (based on Llama-1.3b ratio 5504/2048)
SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 3072, 4096]  # Fixed sequence lengths
B = 1            # Batch size
WARMUP_STEPS = 2
MEASURE_STEPS = 5

np.random.seed(0)
torch.manual_seed(0)
np_ff  = FeedForwardNumpy(D, F)# Configuration
inductor_config.compile_threads = 1
torch.set_num_threads(1)
D_MIN = 512      # Minimum hidden size
D_MAX = 6000     # Maximum hidden size
D_STEP = 512     # Step size for hidden size
F_RATIO = 2.6875 # F = F_RATIO * D (based on Llama-1.3b ratio 5504/2048)
SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 3072, 4096]  # Fixed sequence lengths
B = 1            # Batch size
WARMUP_STEPS = 2
MEASURE_STEPS = 5

np.random.seed(0)
torch.manual_seed(0)
np_ff  = FeedForwardNumpy(D, F)

# Benchmark Functions
def benchmark_nntile(seq_len):
    nntile_config = nntile.starpu.Config(1,1,0,0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = FeedForwardNNTile(B, seq_len, F, D, np_ff.W1, np_ff.b1, np_ff.W2, np_ff.b2)
    
    # Create data directly on GPU
    x = np.random.randn(B, seq_len, D).astype(np.float32)
    grad_out = np.random.randn(B, seq_len, D).astype(np.float32)
    
    # Warmup
    for _ in range(WARMUP_STEPS):
        out = model.forward(x)
        dx = model.backward(grad_out)
    
    # # Memory measurement
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    
    for _ in range(MEASURE_STEPS):
        out = model.forward(x)
        dx = model.backward(grad_out)
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    # gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return elapsed



def benchmark_pytorch(seq_len):
    """
    Benchmark a FeedForwardTorch model on GPU for a given sequence length,
    without torch.compile.

    Returns:
        elapsed (float): average time per iteration (seconds), or nan on error.
        gpu_mem (float): peak GPU memory used in GB, or nan on error.
    Raises:
        RuntimeError if NaNs are detected during warmup or measurement.
    """
    # Fix seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda")
    # Instantiate the model and move to GPU
    model = FeedForwardTorch(D, F).to(device)

    # Prepare input and grad_out on GPU
    x = torch.randn(B, seq_len, D, device=device, requires_grad=True)
    grad_out = torch.randn(B, seq_len, D, device=device)

    # Warmup iterations: stabilize caches, JIT kernels, etc.
    for _ in range(WARMUP_STEPS):
        out = model(x)
        # Check for NaNs
        if torch.isnan(out).any():
            raise RuntimeError(f"NaN detected in model output during warmup at seq_len={seq_len}")
        # Use a simple loss to ensure backward is exercised
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    # Prepare for measurement: synchronize and reset peak stats
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    # Timed iterations
    for _ in range(MEASURE_STEPS):
        out = model(x)
        if torch.isnan(out).any():
            raise RuntimeError(f"NaN detected in model output during measurement at seq_len={seq_len}")
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    # Synchronize to ensure all ops complete
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    # # Peak memory in bytes → GB
    # gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed




# Main Benchmark
def run_benchmark():
    results = {}
    
    # Generate hidden sizes to test
    D_values = list(range(D_MIN, D_MAX + 1, D_STEP))
    
    for seq_len in SEQ_LENGTHS:
        print(f"\nBenchmarking seq_len={seq_len}")
        ratios = []
        
        for D in D_values:
            F = int(D * F_RATIO)  # Calculate F based on fixed ratio
            
            try:
                # Benchmark NNTile
                nntile_time = benchmark_nntile(seq_len)
                
                # Benchmark PyTorch
                torch_time = benchmark_pytorch(seq_len)
                
                # Calculate ratio
                ratio = nntile_time / torch_time
                ratios.append(ratio)
                print(f"D={D}, F={F}: ratio = {ratio:.4f}")
                
            except RuntimeError as e:  # Handle OOM errors
                print(f"OOM at seq_len={seq_len}, D={D}, F={F}: {str(e)}")
                ratios.append(float('nan'))
        
        results[seq_len] = (D_values, ratios)
    
    return results




# Visualization
def plot_results(results):
    plt.figure(figsize=(10, 6))
    
    # Create plot for each sequence length
    for seq_len, (D_values, ratios) in results.items():
        # Filter out OOM results
        valid_D = []
        valid_ratios = []
        for d, r in zip(D_values, ratios):
            if not np.isnan(r):
                valid_D.append(d)
                valid_ratios.append(r)
        
        plt.plot(valid_D, valid_ratios, 'o-', label=f'Seq len = {seq_len}')
    
    plt.title("NNTile/PyTorch Execution Time Ratio")
    plt.xlabel("Hidden Size (D)")
    plt.ylabel("Time Ratio (NNTile / PyTorch)")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_ratio.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available! Exiting...")
        exit()
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Run benchmark
    benchmark_results = run_benchmark()
    
    # Output results and plot
    plot_results(benchmark_results)
    print("\nChart saved as 'time_ratio.png'")