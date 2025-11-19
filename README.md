# Matching Logprobs Between HF and SGLang

### Version
1. transformers: `4.57.1`
2. sglang: `0.5.5`

### Steps
1. download model with `hf download Qwen/Qwen3-0.6B --local-dir ./qwen-0.6b` 
2. generate rollout: `python -m minimal_on_policy run_sglang qwen-0.6b qwen-0.6b_text.json 50 1 True`
3. compare: `python -m minimal_on_policy run_hf_lenient_compare qwen-0.6b qwen-0.6b_text.json qwen-0.6b_report.json`

### Result
```bash
✓ Prompt 1: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 2: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 3: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 4: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 5: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 6: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 7: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 8: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 9: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 10: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 11: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 12: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 13: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 14: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 15: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 16: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 17: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 18: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 19: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 20: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 21: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 22: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 23: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 24: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 25: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 26: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000
✓ Prompt 27: max_abs_diff=0.0000000000, mean_abs_diff=0.0000000000

Overall: 0 failures, max_abs_diff=0.0000000000
```

### Credit/Info
minimal script based on slime + sglang [true on policy](https://github.com/THUDM/slime/tree/main/examples/true_on_policy)
