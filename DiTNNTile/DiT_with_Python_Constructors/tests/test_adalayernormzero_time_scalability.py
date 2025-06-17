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
from nntile.layer.layer_norm import LayerNorm
import torch.nn.functional as F
from mlp_nntile import MLPNNTile
from scale_shift_nntile import ScaleShiftNNTile
from AdaptiveLayerNormZeroTorch import AdaptiveLayerNormZeroTorch
from AdaptiveLayerNormZeroNNTile import AdaptiveLayerNormZeroNNTile
from nntile.tensor import TensorMoments, to_numpy, from_array, fill_async, clear_async, sum_slice_async, norm_slice_async, hypot_scalar_inverse_async, prod_slice_async, add_slice_async, add_inplace_async, add_slice_inplace_async, sumprod_slice_async
from nntile.layer.layer_norm import LayerNorm
from layer_norm_noaffine_nntile import LayerNormNoAffine

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc

# ==============================================
# === Adjust these constants as needed
D_MIN = 512       # Minimum hidden size
D_MAX = 4096      # Maximum hidden size
D_STEP = 512      # Step size for hidden size
# Sequence lengths to test
SEQ_LENGTHS = [64, 128, 256, 512, 1024]  
B = 1             # Batch size; reduce if CPU is too slow or memory constrained
WARMUP_STEPS = 2
MEASURE_STEPS = 5
# ==============================================

# Fix global seeds once
np.random.seed(0)
torch.manual_seed(0)

# Utility: benchmark NNTile CPU for AdaptiveLayerNormZero, parameterized by seq_len and D
def benchmark_nntile_cpu(seq_len, D):
    """
    Benchmark the NNTile CPU implementation of AdaptiveLayerNormZero for given seq_len and hidden size D.
    Returns:
        elapsed_time (float): average time per iteration (seconds), or nan on error.
    """
    # Configure StarPU for CPU-only: 1 CPU worker, 0 GPUs
    try:
        nntile.starpu.Config(1, 0, 0, 0)
    except Exception:
        pass

    # Fix seeds per call for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare inputs on CPU, convert to NumPy
    # B x seq_len x D  and  B x D
    x_torch = torch.randn(B, seq_len, D, requires_grad=True)
    emb_torch = torch.randn(B, D, requires_grad=True)
    x_np = x_torch.detach().numpy()
    emb_np = emb_torch.detach().numpy()

    # Initialize weights & bias, convert to NumPy
    W_full = torch.randn(6 * D, D)
    b_full = torch.randn(6 * D)
    W_np = W_full.detach().numpy().astype(np.float32)
    b_np = b_full.detach().numpy().astype(np.float32)

    Ws = np.split(W_np, 6, axis=0)
    bs = np.split(b_np, 6)

    # Instantiate NNTile module (CPU path)
    try:
        tile_mod = AdaptiveLayerNormZeroNNTile(
            B, seq_len, D,
            Ws[0], Ws[1], Ws[2], Ws[3], Ws[4], Ws[5],
            bs[0], bs[1], bs[2], bs[3], bs[4], bs[5],
            bias=True
        )
    except Exception as e:
        print(f"[NNTile CPU] Instantiation error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Pre-generate upstream gradients as NumPy arrays (fixed shapes)
    rng = np.random.RandomState(0)
    try:
        out0, gate_msa0, shift_mlp0, scale_mlp0, gate_mlp0 = tile_mod.forward(x_np, emb_np)
    except Exception as e:
        print(f"[NNTile CPU] Forward error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')
    shape_out = out0.shape
    shape_gate_msa = gate_msa0.shape
    shape_shift_mlp = shift_mlp0.shape
    shape_scale_mlp = scale_mlp0.shape
    shape_gate_mlp = gate_mlp0.shape

    d_out = rng.randn(*shape_out).astype(np.float32)
    d_gate_msa = rng.randn(*shape_gate_msa).astype(np.float32)
    d_shift_mlp = rng.randn(*shape_shift_mlp).astype(np.float32)
    d_scale_mlp = rng.randn(*shape_scale_mlp).astype(np.float32)
    d_gate_mlp = rng.randn(*shape_gate_mlp).astype(np.float32)

    # Warmup (not timed)
    for _ in range(WARMUP_STEPS):
        try:
            _ = tile_mod.forward(x_np, emb_np)
            _ = tile_mod.backward(x_np, d_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp)
        except Exception as e:
            print(f"[NNTile CPU] Warmup error at seq_len={seq_len}, D={D}: {e}")
            return float('nan')

    # Clean up
    gc.collect()

    # Timed measurement
    start_time = time.perf_counter()
    try:
        for _ in range(MEASURE_STEPS):
            out_nnt, gate_msa_n, shift_mlp_n, scale_mlp_n, gate_mlp_n = tile_mod.forward(x_np, emb_np)
            _ = tile_mod.backward(x_np, d_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp)
    except Exception as e:
        print(f"[NNTile CPU] Measurement error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    return elapsed


# Utility: benchmark PyTorch CPU for AdaptiveLayerNormZero, parameterized by seq_len and D
def benchmark_pytorch_cpu(seq_len, D):
    """
    Benchmark the PyTorch CPU implementation of AdaptiveLayerNormZero for given seq_len and hidden size D.
    Returns:
        elapsed_time (float): average time per iteration (seconds), or nan on error.
    """
    # Fix seeds per call
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu")
    try:
        model = AdaptiveLayerNormZeroTorch(D, bias=True).to(device)
    except Exception as e:
        print(f"[PyTorch CPU] Instantiation error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Initialize parameters on CPU
    W_full = torch.randn(6 * D, D)
    b_full = torch.randn(6 * D)
    with torch.no_grad():
        model.W.copy_(W_full)
        model.b.copy_(b_full)

    # Prepare inputs on CPU
    x_torch = torch.randn(B, seq_len, D, device=device, requires_grad=True)
    emb_torch = torch.randn(B, D, device=device, requires_grad=True)

    # Warmup (not timed)
    for _ in range(WARMUP_STEPS):
        try:
            out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
        except Exception as e:
            print(f"[PyTorch CPU] Warmup forward error at seq_len={seq_len}, D={D}: {e}")
            return float('nan')

        # Generate upstream grads
        d_out = torch.randn_like(out_torch)
        d_gate_msa = torch.randn_like(gate_msa_t)
        d_shift_mlp = torch.randn_like(shift_mlp_t)
        d_scale_mlp = torch.randn_like(scale_mlp_t)
        d_gate_mlp = torch.randn_like(gate_mlp_t)

        try:
            out_torch.backward(d_out, retain_graph=True)
            gate_msa_t.backward(d_gate_msa, retain_graph=True)
            shift_mlp_t.backward(d_shift_mlp, retain_graph=True)
            scale_mlp_t.backward(d_scale_mlp, retain_graph=True)
            gate_mlp_t.backward(d_gate_mlp)
        except Exception as e:
            print(f"[PyTorch CPU] Warmup backward error at seq_len={seq_len}, D={D}: {e}")
            return float('nan')

        model.zero_grad()
        x_torch.grad = None
        emb_torch.grad = None

    gc.collect()

    # Timed measurement
    start_time = time.perf_counter()
    try:
        for _ in range(MEASURE_STEPS):
            out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)

            d_out = torch.randn_like(out_torch)
            d_gate_msa = torch.randn_like(gate_msa_t)
            d_shift_mlp = torch.randn_like(shift_mlp_t)
            d_scale_mlp = torch.randn_like(scale_mlp_t)
            d_gate_mlp = torch.randn_like(gate_mlp_t)

            out_torch.backward(d_out, retain_graph=True)
            gate_msa_t.backward(d_gate_msa, retain_graph=True)
            shift_mlp_t.backward(d_shift_mlp, retain_graph=True)
            scale_mlp_t.backward(d_scale_mlp, retain_graph=True)
            gate_mlp_t.backward(d_gate_mlp)

            model.zero_grad()
            x_torch.grad = None
            emb_torch.grad = None
    except Exception as e:
        print(f"[PyTorch CPU] Measurement error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    return elapsed


# Main benchmark loop: collect ratios
def run_benchmark_cpu_ratios():
    """
    Runs benchmarks for each seq_len and D, returns a dict:
      { seq_len: (D_values_list, ratio_list) }
    where ratio = nntile_time / pytorch_time.
    """
    results = {}
    D_values = list(range(D_MIN, D_MAX + 1, D_STEP))
    for seq_len in SEQ_LENGTHS:
        print(f"\nBenchmarking seq_len={seq_len} (CPU)...")
        ratios = []
        for D in D_values:
            # Measure NNTile CPU
            n_time = benchmark_nntile_cpu(seq_len, D)
            # Measure PyTorch CPU
            t_time = benchmark_pytorch_cpu(seq_len, D)
            if np.isnan(n_time) or np.isnan(t_time) or t_time == 0.0:
                ratio = float('nan')
            else:
                ratio = n_time / t_time
            ratios.append(ratio)
            print(f"  D={D:<5}  NNTile time={n_time:.4f}s  PyTorch time={t_time:.4f}s  ratio={ratio:.4f}")
        results[seq_len] = (D_values, ratios)
    return results

# Plotting function for the ratio vs D for each seq_len
def plot_cpu_ratios(results, save_path="cpu_time_ratio.png"):
    plt.figure(figsize=(10, 6))
    for seq_len, (D_values, ratios) in results.items():
        # Filter valid points
        valid_D = []
        valid_ratios = []
        for d, r in zip(D_values, ratios):
            if not np.isnan(r):
                valid_D.append(d)
                valid_ratios.append(r)
        if len(valid_D) > 0:
            plt.plot(valid_D, valid_ratios, 'o-', label=f"Seq len={seq_len}")
    plt.title("CPU: NNTile / PyTorch Execution Time Ratio for AdaptiveLayerNormZero")
    plt.xlabel("Hidden Size D")
    plt.ylabel("Time Ratio (NNTile_CPU / PyTorch_CPU)")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # No GPU check, this is CPU-only.
    benchmark_results = run_benchmark_cpu_ratios()
    plot_cpu_ratios(benchmark_results)


# Configuration
inductor_config.compile_threads = 1
torch.set_num_threads(1)
D = 1800      # Llama-1.3b hidden size      # Feed-forward hidden dimension
B = 1         # Batch size
SEQ_LENGTHS = [16, 32, 64, 256, 512, 1024, 2048, 3064, 4096, 6000, 8000,10000, 20000, 25000]
WARMUP_STEPS = 2
MEASURE_STEPS = 5
np.random.seed(0)
torch.manual_seed(0)



def benchmark_nntile_gpu(seq_len):
    nntile_config = nntile.starpu.Config(1,1,0,0)
    device = torch.device("cuda")
    np.random.seed(0)
    torch.manual_seed(0)

    # Create random inputs (CPU first)
    x_torch_cpu = torch.randn(B, seq_len, D, requires_grad=True)
    emb_torch_cpu = torch.randn(B, D, requires_grad=True)

    # Initialize weights and bias once
    W_full = torch.randn(6 * D, D)
    b_full = torch.randn(6 * D)

    # Convert to NumPy (for NNTile)
    x_np = x_torch_cpu.detach().numpy().astype(np.float32)
    emb_np = emb_torch_cpu.detach().numpy().astype(np.float32)
    W_np = W_full.detach().numpy().astype(np.float32)
    b_np = b_full.detach().numpy().astype(np.float32)

    # Split weights for NNTile
    Ws = np.split(W_np, 6, axis=0)
    bs = np.split(b_np, 6)

    # Create NNTile module
    tile_mod = AdaptiveLayerNormZeroNNTile(
        B, seq_len, D,
        Ws[0], Ws[1], Ws[2], Ws[3], Ws[4], Ws[5],
        bs[0], bs[1], bs[2], bs[3], bs[4], bs[5],
        bias=True
    )

    # Create PyTorch model on GPU
    model = AdaptiveLayerNormZeroTorch(D, bias=True).to(device)

    # Copy weights and bias to PyTorch model on GPU
    with torch.no_grad():
        model.W.copy_(W_full)
        model.b.copy_(b_full)

    # Move input tensors to GPU
    x_torch = x_torch_cpu.detach().to(device).requires_grad_()
    emb_torch = emb_torch_cpu.detach().to(device).requires_grad_()

    # Warmup PyTorch
    for _ in range(WARMUP_STEPS):
        _ = model(x_torch, emb_torch)

    # Prepare random gradients for backward
    out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
    d_out = np.random.randn(*out_torch.shape).astype(np.float32)
    d_gate_msa = np.random.randn(*gate_msa_t.shape).astype(np.float32)
    d_shift_mlp = np.random.randn(*shift_mlp_t.shape).astype(np.float32)
    d_scale_mlp = np.random.randn(*scale_mlp_t.shape).astype(np.float32)
    d_gate_mlp = np.random.randn(*gate_mlp_t.shape).astype(np.float32)

    # Warmup NNTile
    for _ in range(WARMUP_STEPS):
        _ = tile_mod.forward(x_np, emb_np)
        _ = tile_mod.backward(
            x_np,
            d_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp
        )

    # Measure NNTile runtime
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        _ = tile_mod.forward(x_np, emb_np)
        _ = tile_mod.backward(
            x_np,
            d_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

    return elapsed, gpu_mem


def benchmark_pytorch_gpu(seq_len):
    """
    Benchmark an AdaptiveLayerNormZeroTorch model on GPU for a given sequence length.
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
    model = AdaptiveLayerNormZeroTorch(D, bias=True).to(device)

    # Initialize parameters randomly on CPU, then copy into model on GPU.
    # If you already have W_full, b_full globally, you can override here similarly.
    W_full = torch.randn(6 * D, D)
    b_full = torch.randn(6 * D)
    with torch.no_grad():
        # model.W and model.b should be on GPU; copy_ from CPU to GPU works.
        model.W.copy_(W_full)
        model.b.copy_(b_full)

    # Prepare inputs directly on GPU
    x_torch = torch.randn(B, seq_len, D, device=device, requires_grad=True)
    emb_torch = torch.randn(B, D, device=device, requires_grad=True)

    # Warmup iterations
    for _ in range(WARMUP_STEPS):
        out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
        # Generate upstream gradients on GPU, matching shapes
        # Using torch.randn on GPU ensures same device
        d_out = torch.randn_like(out_torch)
        d_gate_msa = torch.randn_like(gate_msa_t)
        d_shift_mlp = torch.randn_like(shift_mlp_t)
        d_scale_mlp = torch.randn_like(scale_mlp_t)
        d_gate_mlp = torch.randn_like(gate_mlp_t)

        # Check for NaNs in output
        if torch.isnan(out_torch).any():
            raise RuntimeError(f"NaN detected in model output during warmup at seq_len={seq_len}")

        # Backward through each output; keep retain_graph=True except for the last backward
        # Here we do them in sequence; to simplify, you can combine into a single scalar loss if possible.
        # Using retain_graph so that repeated backward works; torch will accumulate gradients.
        out_torch.backward(d_out, retain_graph=True)
        gate_msa_t.backward(d_gate_msa, retain_graph=True)
        shift_mlp_t.backward(d_shift_mlp, retain_graph=True)
        scale_mlp_t.backward(d_scale_mlp, retain_graph=True)
        # For the last backward, retain_graph can be False:
        gate_mlp_t.backward(d_gate_mlp)  # no retain_graph

        # Zero gradients before next iteration
        model.zero_grad()
        x_torch.grad = None
        emb_torch.grad = None

    # Prepare for measurement: synchronize and reset peak stats
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    # Timed iterations
    for _ in range(MEASURE_STEPS):
        out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
        # Generate upstream gradients on GPU
        d_out = torch.randn_like(out_torch)
        d_gate_msa = torch.randn_like(gate_msa_t)
        d_shift_mlp = torch.randn_like(shift_mlp_t)
        d_scale_mlp = torch.randn_like(scale_mlp_t)
        d_gate_mlp = torch.randn_like(gate_mlp_t)

        if torch.isnan(out_torch).any():
            raise RuntimeError(f"NaN detected in model output during measurement at seq_len={seq_len}")

        out_torch.backward(d_out, retain_graph=True)
        gate_msa_t.backward(d_gate_msa, retain_graph=True)
        shift_mlp_t.backward(d_shift_mlp, retain_graph=True)
        scale_mlp_t.backward(d_scale_mlp, retain_graph=True)
        gate_mlp_t.backward(d_gate_mlp)

        model.zero_grad()
        x_torch.grad = None
        emb_torch.grad = None

    # Synchronize to ensure all ops complete
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
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


def plot_results(results, save_path="gpu_benchmark_adaptivelayernormzero.png"):
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
    plt.title("AdaptiveLayerNormZero Execution Time (GPU)")
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
        def fmt(x):
            return f"{x:.6f}" if not (isinstance(x, float) and np.isnan(x)) else "nan"
        print(f"| {seq_len:6d} | {fmt(n_time):>10} | {fmt(t_time):>11} | "
              f"{fmt(n_mem):>15} | {fmt(t_mem):>16} |")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available! Exiting...")
        exit(1)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total GPU memory: {total_mem:.2f} GB")

    results = run_benchmark()
    print_results_table(results)
    plot_results(results)


import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc

# ==============================================
# === Adjust these constants as needed
D_MIN = 512       # Minimum hidden size
D_MAX = 4096      # Maximum hidden size
D_STEP = 512      # Step size for hidden size
# Sequence lengths to test
SEQ_LENGTHS = [64, 128, 256, 512, 1024]
B = 1             # Batch size; keep small for GPU memory constraints
WARMUP_STEPS = 2
MEASURE_STEPS = 5
# ==============================================

# Fix global seeds once
np.random.seed(0)
torch.manual_seed(0)

# Utility: benchmark NNTile GPU for AdaptiveLayerNormZero, parameterized by seq_len and D
def benchmark_nntile_gpu(seq_len, D):
    """
    Benchmark the NNTile GPU implementation of AdaptiveLayerNormZero for given seq_len and hidden size D.
    Returns:
        elapsed_time (float): average time per iteration (seconds), or nan on error.
    """
    # Configure StarPU: 1 CPU worker, 1 GPU
    try:
        nntile.starpu.Config(1, 1, 0, 0)
    except Exception:
        pass

    # Fix seeds per call for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare inputs on CPU first, then convert for GPU and NNTile:
    # We create CPU tensors for PyTorch copy later, and NumPy arrays for NNTile.
    try:
        x_torch_cpu = torch.randn(B, seq_len, D, dtype=torch.float32)
        emb_torch_cpu = torch.randn(B, D, dtype=torch.float32)
    except Exception as e:
        print(f"[NNTile GPU] Input creation error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Initialize weights & bias once
    try:
        W_full = torch.randn(6 * D, D, dtype=torch.float32)
        b_full = torch.randn(6 * D, dtype=torch.float32)
    except Exception as e:
        print(f"[NNTile GPU] Weight init error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Convert to NumPy for NNTile
    x_np = x_torch_cpu.detach().cpu().numpy().astype(np.float32)
    emb_np = emb_torch_cpu.detach().cpu().numpy().astype(np.float32)
    W_np = W_full.detach().cpu().numpy().astype(np.float32)
    b_np = b_full.detach().cpu().numpy().astype(np.float32)

    # Split weights for NNTile
    try:
        Ws = np.split(W_np, 6, axis=0)
        bs = np.split(b_np, 6)
    except Exception as e:
        print(f"[NNTile GPU] Weight split error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Instantiate NNTile module (GPU path via StarPU)
    try:
        tile_mod = AdaptiveLayerNormZeroNNTile(
            B, seq_len, D,
            Ws[0], Ws[1], Ws[2], Ws[3], Ws[4], Ws[5],
            bs[0], bs[1], bs[2], bs[3], bs[4], bs[5],
            bias=True
        )
    except Exception as e:
        print(f"[NNTile GPU] Instantiation error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Prepare PyTorch model on GPU
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("[NNTile GPU] CUDA not available")
        return float('nan')
    try:
        model = AdaptiveLayerNormZeroTorch(D, bias=True).to(device)
    except Exception as e:
        print(f"[NNTile GPU] PyTorch model instantiation error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Copy weights and bias to PyTorch model on GPU
    try:
        with torch.no_grad():
            model.W.copy_(W_full)
            model.b.copy_(b_full)
    except Exception as e:
        print(f"[NNTile GPU] PyTorch weight copy error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Move inputs to GPU for PyTorch forward/backward
    try:
        x_torch = x_torch_cpu.to(device).requires_grad_(True)
        emb_torch = emb_torch_cpu.to(device).requires_grad_(True)
    except Exception as e:
        print(f"[NNTile GPU] Input to GPU error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Prepare upstream gradients for NNTile (NumPy)
    # First run one forward to get output shapes from PyTorch or NNTile?
    # We'll use PyTorch forward to get shapes for gradient arrays:
    try:
        out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
    except Exception as e:
        print(f"[NNTile GPU] PyTorch forward error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Convert shapes to NumPy gradients
    try:
        shape_out = tuple(out_torch.shape)
        shape_gate_msa = tuple(gate_msa_t.shape)
        shape_shift_mlp = tuple(shift_mlp_t.shape)
        shape_scale_mlp = tuple(scale_mlp_t.shape)
        shape_gate_mlp = tuple(gate_mlp_t.shape)
    except Exception as e:
        print(f"[NNTile GPU] Output shape extraction error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    rng = np.random.RandomState(0)
    d_out = rng.randn(*shape_out).astype(np.float32)
    d_gate_msa = rng.randn(*shape_gate_msa).astype(np.float32)
    d_shift_mlp = rng.randn(*shape_shift_mlp).astype(np.float32)
    d_scale_mlp = rng.randn(*shape_scale_mlp).astype(np.float32)
    d_gate_mlp = rng.randn(*shape_gate_mlp).astype(np.float32)

    # Warmup: PyTorch
    try:
        for _ in range(WARMUP_STEPS):
            _ = model(x_torch, emb_torch)
    except Exception as e:
        print(f"[NNTile GPU] PyTorch warmup error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Warmup: NNTile
    try:
        for _ in range(WARMUP_STEPS):
            _ = tile_mod.forward(x_np, emb_np)
            _ = tile_mod.backward(x_np, d_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp)
    except Exception as e:
        print(f"[NNTile GPU] NNTile warmup error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Ensure GPU is ready
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Timed measurement for NNTile GPU:
    # Note: We assume tile_mod.forward/backward schedule GPU work via StarPU;
    # we include torch.cuda.synchronize() after to sync default stream.
    start_time = time.perf_counter()
    try:
        for _ in range(MEASURE_STEPS):
            _ = tile_mod.forward(x_np, emb_np)
            _ = tile_mod.backward(x_np, d_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp)
        # Synchronize GPU to ensure all kernels finish
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[NNTile GPU] Measurement error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS

    return elapsed


# Utility: benchmark PyTorch GPU for AdaptiveLayerNormZero, parameterized by seq_len and D
def benchmark_pytorch_gpu(seq_len, D):
    """
    Benchmark the PyTorch GPU implementation of AdaptiveLayerNormZero for given seq_len and hidden size D.
    Returns:
        elapsed_time (float): average time per iteration (seconds), or nan on error.
    """
    if not torch.cuda.is_available():
        print("[PyTorch GPU] CUDA not available")
        return float('nan')

    # Fix seeds per call
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda")
    # Instantiate model on GPU
    try:
        model = AdaptiveLayerNormZeroTorch(D, bias=True).to(device)
    except Exception as e:
        print(f"[PyTorch GPU] Instantiation error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Initialize parameters on CPU then copy to GPU
    try:
        W_full = torch.randn(6 * D, D, dtype=torch.float32)
        b_full = torch.randn(6 * D, dtype=torch.float32)
        with torch.no_grad():
            model.W.copy_(W_full)
            model.b.copy_(b_full)
    except Exception as e:
        print(f"[PyTorch GPU] Weight init/copy error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Prepare inputs on GPU
    try:
        x_torch = torch.randn(B, seq_len, D, device=device, requires_grad=True)
        emb_torch = torch.randn(B, D, device=device, requires_grad=True)
    except Exception as e:
        print(f"[PyTorch GPU] Input creation error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Warmup
    try:
        for _ in range(WARMUP_STEPS):
            out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
    except Exception as e:
        print(f"[PyTorch GPU] Warmup forward error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Prepare upstream gradients once (reuse shapes)
    try:
        out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
    except Exception as e:
        print(f"[PyTorch GPU] Forward error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')
    try:
        d_out = torch.randn_like(out_torch)
        d_gate_msa = torch.randn_like(gate_msa_t)
        d_shift_mlp = torch.randn_like(shift_mlp_t)
        d_scale_mlp = torch.randn_like(scale_mlp_t)
        d_gate_mlp = torch.randn_like(gate_mlp_t)
    except Exception as e:
        print(f"[PyTorch GPU] Gradient init error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Warmup backward
    try:
        out_torch.backward(d_out, retain_graph=True)
        gate_msa_t.backward(d_gate_msa, retain_graph=True)
        shift_mlp_t.backward(d_shift_mlp, retain_graph=True)
        scale_mlp_t.backward(d_scale_mlp, retain_graph=True)
        gate_mlp_t.backward(d_gate_mlp)
        model.zero_grad()
        x_torch.grad = None
        emb_torch.grad = None
    except Exception as e:
        print(f"[PyTorch GPU] Warmup backward error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')

    # Ensure GPU is ready
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Timed measurement
    start_time = time.perf_counter()
    try:
        for _ in range(MEASURE_STEPS):
            out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = model(x_torch, emb_torch)
            out_torch.backward(d_out, retain_graph=True)
            gate_msa_t.backward(d_gate_msa, retain_graph=True)
            shift_mlp_t.backward(d_shift_mlp, retain_graph=True)
            scale_mlp_t.backward(d_scale_mlp, retain_graph=True)
            gate_mlp_t.backward(d_gate_mlp)
            model.zero_grad()
            x_torch.grad = None
            emb_torch.grad = None
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[PyTorch GPU] Measurement error at seq_len={seq_len}, D={D}: {e}")
        return float('nan')
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    return elapsed


# Main benchmark loop: collect ratios
def run_benchmark_gpu_ratios():
    """
    Runs benchmarks for each seq_len and D, returns a dict:
      { seq_len: (D_values_list, ratio_list) }
    where ratio = nntile_gpu_time / pytorch_gpu_time.
    """
    results = {}
    D_values = list(range(D_MIN, D_MAX + 1, D_STEP))
    for seq_len in SEQ_LENGTHS:
        print(f"\nBenchmarking seq_len={seq_len} (GPU)...")
        ratios = []
        for D in D_values:
            # Measure NNTile GPU
            n_time = benchmark_nntile_gpu(seq_len, D)
            # Measure PyTorch GPU
            t_time = benchmark_pytorch_gpu(seq_len, D)
            if np.isnan(n_time) or np.isnan(t_time) or t_time == 0.0:
                ratio = float('nan')
            else:
                ratio = n_time / t_time
            ratios.append(ratio)
            print(f"  D={D:<5}  NNTile GPU time={n_time:.4f}s  PyTorch GPU time={t_time:.4f}s  ratio={ratio:.4f}")
        results[seq_len] = (D_values, ratios)
    return results

# Plotting function for the ratio vs D for each seq_len
def plot_gpu_ratios(results, save_path="gpu_time_ratio.png"):
    plt.figure(figsize=(10, 6))
    for seq_len, (D_values, ratios) in results.items():
        valid_D = []
        valid_ratios = []
        for d, r in zip(D_values, ratios):
            if not np.isnan(r):
                valid_D.append(d)
                valid_ratios.append(r)
        if len(valid_D) > 0:
            plt.plot(valid_D, valid_ratios, 'o-', label=f"Seq len={seq_len}")
    plt.title("GPU: NNTile / PyTorch Execution Time Ratio for AdaptiveLayerNormZero")
    plt.xlabel("Hidden Size D")
    plt.ylabel("Time Ratio (NNTile_GPU / PyTorch_GPU)")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

# Main execution
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available! Exiting...")
        exit(1)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total GPU memory: {total_mem:.2f} GB")

    benchmark_results = run_benchmark_gpu_ratios()
    plot_gpu_ratios(benchmark_results)