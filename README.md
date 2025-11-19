# Reproduce HF/SGLang same logprobs

### Version
1. transformers: `4.57.1`
2. sglang: `0.5.5`

### Steps
1. download model with `hf download Qwen/Qwen3-0.6B --local-dir ./qwen-0.6b` 
2. generate rollout: `python -m minimal_on_policy run_sglang qwen-0.6b qwen-0.6b_text.json 50 1 False`
3. compare: `python -m minimal_on_policy run_hf_lenient_compare qwen-0.6b qwen-0.6b_text.json 0 qwen-0.6b_report.json`

### Result
```bash
✓ Prompt 1: | logprob_diff: max=0.0000000000, min=0.0000000000, max_abs=0.0000000000, min_abs=0.0000000000, mean_abs=0.0000000000
✓ Prompt 2: | logprob_diff: max=0.0000000000, min=0.0000000000, max_abs=0.0000000000, min_abs=0.0000000000, mean_abs=0.0000000000
✓ Prompt 3: | logprob_diff: max=0.0000000000, min=0.0000000000, max_abs=0.0000000000, min_abs=0.0000000000, mean_abs=0.0000000000
✓ Prompt 4: | logprob_diff: max=0.0000000000, min=0.0000000000, max_abs=0.0000000000, min_abs=0.0000000000, mean_abs=0.0000000000
```

### Credit/Info
minimal script based on slime + sglang [true on policy](https://github.com/THUDM/slime/tree/main/examples/true_on_policy)

same as https://gist.github.com/nanjiangwill/dda73f2ac66c8447b759111452b9fe0e
