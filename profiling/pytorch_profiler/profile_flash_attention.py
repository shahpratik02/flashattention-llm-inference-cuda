import torch
import torch.profiler as profiler
import sys
import os

# Add parent directory to path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

# Import custom modules with error handling
try:
    from cuda_flash_attention.attention import CustomFlashAttention
except ImportError as e:
    print(f"ERROR: Could not import modules. Did you compile CUDA kernels?")
    print(f"Run: python -m cuda_flash_attention.compile")
    print(f"Error: {e}")
    sys.exit(1)

def _get_cuda_time_us(evt):
    """Handle PyTorch profiler API differences across versions."""
    return getattr(
        evt,
        "self_cuda_time_total",
        getattr(evt, "cuda_time_total", getattr(evt, "device_time_total", 0.0)),
    )

def profile_attention_implementations():
    """Profile different attention implementations"""
    
    # Configuration
    batch_size = 4
    seq_len = 1024
    hidden_dim = 512
    num_heads = 8
    device = "cuda"
    
    print(f"Profiling Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num heads: {num_heads}")
    print("-" * 50)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Initialize weight matrices
    w_q = torch.randn(hidden_dim, hidden_dim, device=device)
    w_k = torch.randn(hidden_dim, hidden_dim, device=device)
    w_v = torch.randn(hidden_dim, hidden_dim, device=device)
    w_o = torch.randn(hidden_dim, hidden_dim, device=device)
    
    # Create attention modules
    flash_attn = CustomFlashAttention(w_q, w_k, w_v, w_o, hidden_dim, num_heads).to(device)
    
    # Warm-up runs
    print("Warming up...")
    for _ in range(10):
        _ = flash_attn(x, causal=True)
    torch.cuda.synchronize()
    
    print("Starting profiling...")
    
    # Profile with PyTorch Profiler
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with profiler.record_function("flash_attention_forward"):
            output = flash_attn(x, causal=True)
            torch.cuda.synchronize()
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("PROFILING RESULTS - TABLE VIEW")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    # Export Chrome trace (can view locally at chrome://tracing)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    trace_path = os.path.join(results_dir, "flash_attention_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("View it locally by opening chrome://tracing in Chrome browser")
    
    # Export stacks (for flamegraph)
    stacks_path = os.path.join(results_dir, "flash_attention_stacks.txt")
    prof.export_stacks(stacks_path, "self_cuda_time_total")
    print(f"Stack traces exported to: {stacks_path}")
    
    return prof

def profile_one_model(name, run_fn, results_dir):
    """Profile one implementation and export its own chrome trace."""
    # Warm-up to reduce first-run effects
    for _ in range(10):
        _ = run_fn()
    torch.cuda.synchronize()

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
    ) as prof:
        with profiler.record_function(name):
            _ = run_fn()
            torch.cuda.synchronize()

    trace_name = f"trace_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json"
    trace_path = os.path.join(results_dir, trace_name)
    prof.export_chrome_trace(trace_path)

    total_cuda_us = sum(_get_cuda_time_us(evt) for evt in prof.key_averages())

    print(f"\n{name}:")
    print(f"  Total CUDA time: {total_cuda_us / 1000.0:.3f} ms")
    print(f"  Trace: {trace_path}")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=10,
        )
    )
    return total_cuda_us

def profile_comparison():
    """Compare multiple implementations side-by-side"""
    
    batch_size = 4
    seq_len = 512
    hidden_dim = 512
    num_heads = 8
    device = "cuda"
    
    print("\n" + "=" * 80)
    print("COMPARING IMPLEMENTATIONS")
    print("=" * 80)
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Prepare weights
    w_q = torch.randn(hidden_dim, hidden_dim, device=device)
    w_k = torch.randn(hidden_dim, hidden_dim, device=device)
    w_v = torch.randn(hidden_dim, hidden_dim, device=device)
    w_o = torch.randn(hidden_dim, hidden_dim, device=device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    flash_model = CustomFlashAttention(w_q, w_k, w_v, w_o, hidden_dim, num_heads).to(device)
    native_model = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True).to(device)

    flash_us = profile_one_model(
        "FlashAttention (CUDA)",
        lambda: flash_model(x, causal=False),
        results_dir,
    )
    native_us = profile_one_model(
        "PyTorch Native",
        lambda: native_model(x, x, x, need_weights=False),
        results_dir,
    )
    
    # Print comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    flash_ms = flash_us / 1000.0
    native_ms = native_us / 1000.0
    speedup = (native_ms / flash_ms) if flash_ms > 0 else 0.0
    print(f"{'FlashAttention (CUDA)':30s}: {flash_ms:8.3f} ms")
    print(f"{'PyTorch Native':30s}: {native_ms:8.3f} ms")
    print(f"{'Speedup (vs Native)':30s}: {speedup:8.3f}x")

    return {
        "flash_ms": flash_ms,
        "native_ms": native_ms,
        "speedup_x": speedup,
    }

if __name__ == "__main__":
    print("PyTorch Profiler Demo")
    print("=" * 80)
    
    # Profile flash attention
    prof = profile_attention_implementations()

    # Compare traces (Flash vs PyTorch native)
    results = profile_comparison()
    
    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("Check the results/ directory for exported traces")
    print("=" * 80)