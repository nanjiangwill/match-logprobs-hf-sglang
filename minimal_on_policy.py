import json
import os
import fire
import sglang as sgl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup environment
os.environ["NCCL_ALGO"] = "allreduce:tree"
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

# Setup batch invariant mode
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
from transformers.models.qwen3 import modeling_qwen3
enable_batch_invariant_mode(enable_bmm=False)
modeling_qwen3.apply_rotary_pos_emb = torch.compile(dynamic=True)(modeling_qwen3.apply_rotary_pos_emb)

def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Get log probabilities for specific token indices."""
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a galaxy far away, a",
    "Q: Explain why the sky is blue in simple terms. A:",
    "User: Write a haiku about autumn leaves. Assistant:",
]

PROMPTS_ADVANCED = [
    # Q&A with suffix cues
    "Q: What is the difference between precision and recall in machine learning? A:",
    "Q: Explain why the sky appears red during some sunsets, in simple terms. A:",
    "Question: Outline three major causes of the 2008 financial crisis. Answer:",
    "User: Give a concise summary of the plot of 'The Odyssey' in two sentences. Assistant:",
    # Creative with continuation prefix/suffix
    "Story prompt: The last lighthouse on the island blinked for the final time as the storm rolled in. Continue the story:",
    "Poem prompt: Write a haiku about autumn rivers and drifting leaves. Continue:",
    "Creative writing: Begin a sci-fi microstory with “When the stars started whispering,” and continue for 4-6 sentences:",
    "Fantasy worldbuilding seed: In a city where memories are traded like coins, describe a typical marketplace scene. Continue:",
    # Instruction-following with explicit “Assistant:” cue
    "User: List five practical tips for avoiding overfitting when training neural networks. Assistant:",
    "User: Provide a Python function that returns the prime factors of an integer n, with a short explanation after the code. Assistant:",
    "User: Translate into Spanish (informal tone): 'Please text me when you arrive; I'll be waiting by the cafe.' Assistant:",
    "User: Draft a polite email declining a meeting because of a schedule conflict, proposing two alternative times. Assistant:",
    # Reasoning/math with worked-solution cue
    "Problem: A car travels 45 minutes at 80 km/h and then 30 minutes at 60 km/h. What total distance did it cover? Show your steps:",
    "Riddle: I have keys but no locks, space but no rooms; you can enter but can't go outside. What am I? Explanation:",
    "Logic: If all Bloops are Glumps and some Glumps are Wazzles, what can we conclude about Bloops and Wazzles? Reasoning:",
    # Dialogue completions with role tags (prevents immediate EOS)
    "System: You are a helpful assistant.\nUser: Tell me a joke about mathematicians.\nAssistant:",
    "Interviewer: What motivates you to work on open-source projects?\nCandidate:",
    "Doctor: What symptoms have you been experiencing?\nPatient:",
    "Teacher: Explain photosynthesis in one short paragraph.\nStudent:",
    # Structured / list / outline with trailing colon
    "Outline: Provide a three-part outline for a tutorial on Git branching strategies:",
    "Checklist: What should I verify before deploying a web app to production? Provide 7 bullet points:",
    "Compare: Summarize key similarities and differences between Keynesian and Monetarist economics in 4-6 sentences:",
    "Explain like I'm 10: What is an API, and why do apps use them? Answer:",
]

def run_sglang(checkpoint_path: str, output_path: str, max_tokens: int = 10, tp_size: int = 1, use_advanced_prompts: bool = False):
    """Generate text with SGLang and save logprobs."""
    tok = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tp_size != 1:
        print("WARNING: tp_size != 1 detected. Logprobs will be incorrect.")
    model = sgl.Engine(
        model_path=checkpoint_path,
        tp_size=tp_size,
        disable_cuda_graph=False,
        mem_fraction_static=0.7,
        enable_memory_saver=True,
        attention_backend="fa3",
        # Core on-policy parameters
        rl_on_policy_target="fsdp",
        enable_deterministic_inference=True,
    )

    results = []
    prompts = PROMPTS + (PROMPTS_ADVANCED if use_advanced_prompts else [])
    for prompt in prompts:
        input_ids = tok(prompt, return_tensors="pt")["input_ids"].tolist()
        out = model.generate(
            input_ids=input_ids,
            sampling_params={
                "temperature": 1.0,
                "max_new_tokens": max_tokens,
                "top_p": 1.0,
                "top_k": -1,
                "sampling_seed": 42,
            },
            return_logprob=True,
        )[0]

        output_logprobs = out["meta_info"]["output_token_logprobs"]
        results.append({
            "prompt": prompt,
            "output_text": prompt + out["text"],
            "tokens": [tok.decode([t[1]]) for t in output_logprobs],
            "token_ids": [t[1] for t in output_logprobs],
            "logprobs": [float(t[0]) for t in output_logprobs],
        })

    with open(output_path, "w") as f:
        json.dump({"checkpoint_path": checkpoint_path, "results": results}, f, indent=2)
    print(f"Saved {len(results)} generations to {output_path}")

def run_hf_lenient_compare(checkpoint_path: str, other_path: str, output_path: str | None = None):
    """Compare HuggingFace logprobs against SGLang results."""
    with open(other_path) as f:
        other_results = json.load(f)["results"]

    tok = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_3"
    ).cuda().eval()

    device = next(model.parameters()).device
    summaries = []
    failures = 0

    for i, other_res in enumerate(other_results, start=1):
        prompt = other_res["prompt"]
        other_token_ids = other_res["token_ids"]
        other_logprobs = torch.tensor(other_res["logprobs"], device=device)

        with torch.inference_mode():
            prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
            full_ids = torch.tensor([prompt_ids + other_token_ids], device=device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=full_ids).logits[0, len(prompt_ids)-1:-1]
            
            hf_logprobs = selective_log_softmax(logits, torch.tensor(other_token_ids, device=device))
            diffs = (hf_logprobs - other_logprobs).tolist()
            abs_diffs = [abs(d) for d in diffs]

        max_abs = max(abs_diffs)
        status = "✗" if max_abs > 0 else "✓"
        failures += (max_abs > 0)
        
        print(f"{status} Prompt {i}: max_abs_diff={max_abs:.10f}, mean_abs_diff={sum(abs_diffs)/len(abs_diffs):.10f}")
        
        summaries.append({
            "prompt": prompt,
            "max_abs_logprob_diff": max_abs,
            "mean_abs_logprob_diff": sum(abs_diffs) / len(abs_diffs),
        })

    all_diffs = [d for s in summaries for d in [s["max_abs_logprob_diff"]]]
    print(f"\nOverall: {failures} failures, max_abs_diff={max(all_diffs):.10f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"checkpoint_path": checkpoint_path, "prompts": summaries}, f, indent=2)

if __name__ == "__main__":
    fire.Fire({
        "run_sglang": run_sglang,
        "run_hf_lenient_compare": run_hf_lenient_compare,
    })