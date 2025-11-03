### Task 4

Non-causal self-attention speedups:
- batch_size = 4, seq_len = 128: X.XX
- batch_size = 64, seq_len = 4096: X.XX

Causal self-attention speedups:
- batch_size = 4, seq_len = 128: X.XX
- batch_size = 64, seq_len = 4096: X.XX

### Task 5

Prefill & Decode speedups:
- Time to first token (TTFT): X.XX
- Time between tokens (TBT): X.XX

### Task 6

Prefill & Decode times (ms):
- Prefill time (TTFT): X.XX
- Average decode time (TBT): X.XX

### Task 7

Generated output:

What are the advantages of using flashattention kernels compared to the naive pytorch implementation? Think about the shared memory usage, the computation complexity, and the performance. <...>