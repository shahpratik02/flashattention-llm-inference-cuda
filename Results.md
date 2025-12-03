### Task 4

Non-causal self-attention speedups:
- batch_size = 4, seq_len = 128: 2.59
- batch_size = 64, seq_len = 4096: 7.78

Causal self-attention speedups:
- batch_size = 4, seq_len = 128: 1.75
- batch_size = 64, seq_len = 4096: 2.15

### Task 5

Prefill & Decode speedups:
- Time to first token (TTFT): 3.60
- Time between tokens (TBT): 3.12

### Task 6

Prefill & Decode times (ms):
- Prefill time (TTFT): 2851.68 ms
- Average decode time (TBT): 51.88 ms

### Task 7

Generated output:

What are the advantages of using flashattention kernels compared to the naive pytorch implementation? Think about the shared memory usage, the computation complexity, and the performance. <...>

### Questions for the beta testers 

> Your feedback is essential for improving this project. You don't need to write a detailed report, but please share any significant experiences you had—what worked, what didn't, and any challenges you faced. Your insights are incredibly helpful as we prepare to launch this for all students next semester. Thanks!

- Were you able to finish the project?

Yes

- In general, how difficult was the project? Compared to the current Project 2 (Bitonic Sort), Project 3/4 (Simulator), and Project 5 (Compiler), how would you rank the difficulty?

I would say it was the most difficult but also the most interesting

- How much time did you spend on each task (task 1-7)?

1-2 hours

- How familiar were you with Python, NumPy, and PyTorch programming? (Including working with multi-dimensional tensors such as slicing and transposing.)

Very familiar

- How familiar were you with LLM concepts such as transformers, self-attention, causal masks, etc.?

Very Familiar

- How familiar were you with the FlashAttention paper?

Familiar

- How difficult was it to implement attention in Python?

3/5

- How difficult was it to implement softmax, GEMM, and attention in CUDA?

4/5

- Did you use any AI tools (ChatGPT, Cursor, etc.) while doing the project? If so, how helpful were they? Do you think students could get full marks on this project with “vibe coding”?

Yes, to help me where I am stuck. I think becasue of the TODO sections the students will be able to get full marks just by vibe coding

- If we were to add additional tasks, what concepts or assignments should we include?

I would say add something theoretical which is not easy to vibe code

- How could the project description be improved to help students better understand the task and the underlying concepts?

Its good as it is
