import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch._inductor import config as inductor_config



# Configs
inductor_config.compile_threads = 1
torch.set_num_threads(1)
D = 512
B = 1
SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 3064, 4096, 6000, 8000, 10000, 15000]
WARMUP_STEPS = 2
MEASURE_STEPS = 5
F = 4 * D
np.random.seed(0)
torch.manual_seed(0)


config = {'hidden_size': D}

# Torch setup
torch_model = TransformerLayer(config)
torch_model.eval()

# x_torch = torch.randn(B, N, D, requires_grad=True)
# cond_torch = torch.randn(B, D, requires_grad=True)

# Save initial weights and biases
with torch.no_grad():
    W0 = torch_model.adaptive_norm_layer[1].weight.detach().numpy().astype(np.float32)
    b0 = torch_model.adaptive_norm_layer[1].bias.detach().numpy().astype(np.float32)
    W1 = torch_model.mlp_block[0].weight.detach().numpy().astype(np.float32)
    b1 = torch_model.mlp_block[0].bias.detach().numpy().astype(np.float32)
    W2 = torch_model.mlp_block[2].weight.detach().numpy().astype(np.float32)
    b2 = torch_model.mlp_block[2].bias.detach().numpy().astype(np.float32)



def benchmark_nntile_gpu(seq_len):
    nntile_config = nntile.starpu.Config(1, 1, 0, 0)
    np.random.seed(0)
    torch.manual_seed(0)

    model = TransformerLayerNNTile(
        B, seq_len, F, D,
        W0[:D], W0[D:2*D], W0[2*D:3*D], W0[3*D:4*D], W0[4*D:5*D], W0[5*D:],  # W0...W5
        b0[:D], b0[D:2*D], b0[2*D:3*D], b0[3*D:4*D], b0[4*D:5*D], b0[5*D:],  # b0...b5
        W1, b1, W2, b2,
        bias=True
    )
    x = np.random.randn(B, seq_len, D).astype(np.float32)
    cond = np.random.randn(B, D).astype(np.float32)
    grad_out = np.random.randn(B, seq_len, D).astype(np.float32)

    for _ in range(WARMUP_STEPS):
        out = model.forward(x, cond)
        _ = model.backward(grad_out)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    for _ in range(MEASURE_STEPS):
        out = model.forward(x, cond)
        _ = model.backward(grad_out)

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed, gpu_mem


def benchmark_pytorch_gpu(seq_len):
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda")

    model = TransformerLayer({'hidden_size': D}).to(device)
    x = torch.randn(B, seq_len, D, device=device, requires_grad=True)
    cond = torch.randn(B, D, device=device, requires_grad=True)
    grad_out = torch.randn(B, seq_len, D, device=device)

    for _ in range(WARMUP_STEPS):
        out = model(x, cond)
        if torch.isnan(out).any():
            raise RuntimeError(f"NaN in output at seq_len={seq_len}")
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    for _ in range(MEASURE_STEPS):
        out = model(x, cond)
        if torch.isnan(out).any():
            raise RuntimeError(f"NaN in output at seq_len={seq_len}")
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / MEASURE_STEPS
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed, gpu_mem


def run_benchmark():
    results = []
    for seq_len in SEQ_LENGTHS:
        print(f"Benchmarking seq_len={seq_len}")
        try:
            nntile_time, nntile_mem = benchmark_nntile_gpu(seq_len)
        except Exception as e:
            print(f"NNTile error @ seq_len={seq_len}: {e}")
            nntile_time, nntile_mem = float('nan'), float('nan')

        try:
            torch_time, torch_mem = benchmark_pytorch_gpu(seq_len)
        except Exception as e:
            print(f"PyTorch error @ seq_len={seq_len}: {e}")
            torch_time, torch_mem = float('nan'), float('nan')

        results.append([seq_len, nntile_time, torch_time, nntile_mem, torch_mem])
    return results


def plot_results(results, save_path="identity_dit_gpu_benchmark.png"):
    seq_lens = [r[0] for r in results]
    nntile_times = [r[1] for r in results]
    torch_times = [r[2] for r in results]
    nntile_mems = [r[3] for r in results]
    torch_mems = [r[4] for r in results]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(seq_lens, nntile_times, 'o-', label='NNTile GPU')
    plt.plot(seq_lens, torch_times, 's-', label='PyTorch GPU')
    plt.title("Identity DiT Execution Time (GPU)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(seq_lens, nntile_mems, 'o-', label='NNTile GPU')
    plt.plot(seq_lens, torch_mems, 's-', label='PyTorch GPU')
    plt.title("Identity DiT GPU Memory Usage")
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
    print("| Seqlen | NNTile (s) | PyTorch (s) | NNTile Mem (GB) | PyTorch Mem (GB) |")
    print("|--------|------------|-------------|------------------|-------------------|")
    for r in results:
        seq_len, n_time, t_time, n_mem, t_mem = r
        fmt = lambda x: f"{x:.6f}" if not np.isnan(x) else "nan"
        print(f"| {seq_len:6d} | {fmt(n_time):>10} | {fmt(t_time):>11} | {fmt(n_mem):>16} | {fmt(t_mem):>17} |")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    results = run_benchmark()
    print_results_table(results)
    plot_results(results)