import json
import os
import fire
import sglang as sgl
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)


def _setup_env():
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

def _setup_batch_invariant_mode():
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
    from transformers.models.qwen3 import modeling_qwen3
    enable_batch_invariant_mode(enable_bmm=False)
    modeling_qwen3.apply_rotary_pos_emb = torch.compile(dynamic=True)(modeling_qwen3.apply_rotary_pos_emb)
    
# def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
#     """Fused version of the common `log_softmax -> gather` operation.
#     The fused version of this operation avoids the (potentially large) memory overhead
#     of allocating a new tensor to store the full logprobs.
#     Parameters:
#         logits: Tensor of shape [..., V] containing model logits.
#         input_ids: Tensor of shape [...] of token indices whose log-probabilities are gathered.
#     Returns:
#         Tensor of shape [...] containing the log-probabilities corresponding to `input_ids`.
#     """
#     # Compute log(sum(exp(logits))) for normalization
#     log_sum_exp = torch.logsumexp(logits, dim=-1, keepdim=True)
    
#     # Get the logits for the selected tokens
#     selected_logits = torch.gather(logits, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    
#     # Compute log probabilities as: selected_logit - log(sum(exp(all_logits)))
#     return selected_logits - log_sum_exp.squeeze(-1)

def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused version of the common `log_softmax -> gather` operation.
    The fused version of this operation avoids the (potentially large) memory overhead
    of allocating a new tensor to store the full logprobs.
    Parameters:
        logits: Tensor of shape [..., V] containing model logits.
        input_ids: Tensor of shape [...] of token indices whose log-probabilities are gathered.
    Returns:
        Tensor of shape [...] containing the log-probabilities corresponding to `input_ids`.
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

_setup_env()
_setup_batch_invariant_mode()
selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)

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

def _load_sglang_model(ckpt_path: str, use_cuda_graphs: bool, tp_size: int):
    return sgl.Engine(
        model_path=ckpt_path,
        tp_size=tp_size,
        disable_cuda_graph=not use_cuda_graphs,
        mem_fraction_static=0.7,
        attention_backend="fa3",
        rl_on_policy_target="fsdp",
        enable_deterministic_inference=True,
        enable_memory_saver=True,
    )

def run_sglang_inference(
    checkpoint_path: str, output_path: str, max_tokens: int = 10, tp_size: int = 1, use_advanced_prompts: bool = False
):
    tok = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = _load_sglang_model(ckpt_path=checkpoint_path, use_cuda_graphs=True, tp_size=tp_size)

    sampling_params = {
        "temperature": 1.0,
        "max_new_tokens": max_tokens,
        "top_p": 1.0,
        "top_k": -1,
        "sampling_seed": 42,
    }

    prompts = PROMPTS + (PROMPTS_ADVANCED if use_advanced_prompts else [])
    results = []

    for prompt in tqdm.tqdm(prompts, desc="Running inference"):
        input_ids = tok(prompt, return_tensors="pt")["input_ids"].tolist()
        outs = model.generate(input_ids=input_ids, sampling_params=sampling_params, return_logprob=True)
        out = outs[0]

        output_logprobs = out["meta_info"]["output_token_logprobs"]
        token_ids = [t[1] for t in output_logprobs]
        
        results.append({
            "prompt": prompt,
            "output_text": prompt + out["text"],
            "tokens": [tok.decode([tid]) for tid in token_ids],
            "token_ids": token_ids,
            "logprobs": [float(t[0]) for t in output_logprobs],
        })

    with open(output_path, "w") as f:
        json.dump({"checkpoint_path": checkpoint_path, "results": results}, f, indent=2)

    print(f"Saved {len(results)} generations to {output_path}")

def run_hf_lenient_compare(checkpoint_path: str, other_path: str, threshold: int, output_path: str | None = None):
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
        other_token_ids = other_res.get("token_ids")
        other_logprobs = other_res.get("logprobs")
        other_logprobs = torch.tensor(other_logprobs, device=device)
        lp_diffs, abs_lp_diffs = [], []

        with torch.inference_mode():
            # Build the full sequence: prompt + generated tokens
            prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
            full_ids = prompt_ids + other_token_ids
            
            # Single forward pass to get all logits
            input_ids = torch.tensor([full_ids], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits[0]  # shape: [seq_len, vocab_size]

            pred_logits = logits[len(prompt_ids)-1:-1]  # shape: [len(other_token_ids), vocab_size]
            target_ids = torch.tensor(other_token_ids, device=device)
            # Use selective_log_softmax to efficiently get log probs for targets
            hf_logprobs = selective_log_softmax_raw(pred_logits, target_ids)

            # Calculate logprob differences
            if other_logprobs is not None:
                for t, hf_lp in enumerate(hf_logprobs):
                    if t < len(other_logprobs):
                        diff = hf_lp.item() - other_logprobs[t].item()
                        lp_diffs.append(diff)
                        abs_lp_diffs.append(abs(diff))

        max_abs_lp = max(abs_lp_diffs) if abs_lp_diffs else 0.0
        min_abs_lp = min(abs_lp_diffs) if abs_lp_diffs else 0.0
        max_lp = max(lp_diffs) if lp_diffs else 0.0
        min_lp = min(lp_diffs) if lp_diffs else 0.0
        mean_abs_lp = sum(abs_lp_diffs) / len(abs_lp_diffs) if abs_lp_diffs else 0.0
        
        if max_abs_lp > threshold:
            failures += 1
            status = "✗"
        else:
            status = "✓"
            
        print(f"{status} Prompt {i}: | "
              f"logprob_diff: max={max_lp:.10f}, min={min_lp:.10f}, "
              f"max_abs={max_abs_lp:.10f}, min_abs={min_abs_lp:.10f}, mean_abs={mean_abs_lp:.10f}")

        summaries.append({
            "prompt": prompt,
            "logprob_diffs": lp_diffs,
            "max_logprob_diff": max_lp,
            "min_logprob_diff": min_lp,
            "max_abs_logprob_diff": max_abs_lp,
            "min_abs_logprob_diff": min_abs_lp,
            "mean_abs_logprob_diff": mean_abs_lp,
        })

    all_lp_diffs = [d for s in summaries for d in s.get("logprob_diffs", [])]
    all_abs_diffs = [abs(d) for d in all_lp_diffs]
    overall_max_lp = max(all_lp_diffs) if all_lp_diffs else 0.0
    overall_min_lp = min(all_lp_diffs) if all_lp_diffs else 0.0
    overall_max_abs_lp = max(all_abs_diffs) if all_abs_diffs else 0.0
    overall_min_abs_lp = min(all_abs_diffs) if all_abs_diffs else 0.0
    overall_mean_abs_lp = sum(all_abs_diffs) / len(all_abs_diffs) if all_abs_diffs else 0.0
    
    print(f"\nOverall: failures={failures}")
    print(f"Logprob diff: max={overall_max_lp:.10f}, min={overall_min_lp:.10f}")
    print(f"Logprob |diff|: max={overall_max_abs_lp:.10f}, min={overall_min_abs_lp:.10f}, mean={overall_mean_abs_lp:.10f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"checkpoint_path": checkpoint_path, "prompts": summaries}, f, indent=2)


def apply_non_moe_tp_transformers(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        }
    )
    parallelize_module(
        model.model,
        tp_mesh,
        {
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                # output_layouts=Shard(1),
            ),
            # "norm": SequenceParallel(),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for decode_layer in model.model.layers:
        layer_plan = {
            # "input_layernorm": SequenceParallel(),
            # "self_attn": prepare_module_input(
            #     input_layouts=(Shard(1), Replicate(), None),
            #     desired_input_layouts=(Replicate(), Replicate(), None),
            # ),
            "self_attn.q_proj": colwise_parallel(),
            "self_attn.k_proj": colwise_parallel(),
            "self_attn.v_proj": colwise_parallel(),
            # "attention.q_norm": SequenceParallel(sequence_dim=2),
            # "attention.k_norm": SequenceParallel(sequence_dim=2),
            # "self_attn.o_proj": rowwise_parallel(output_layouts=Shard(1)),
            "self_attn.o_proj": rowwise_parallel(),
            # "post_attention_layernorm": SequenceParallel(),
        }

        layer_plan.update(
            {
                # "mlp": prepare_module_input(
                #     input_layouts=(Shard(1),),
                #     desired_input_layouts=(Replicate(),),
                # ),
                "mlp.gate_proj": colwise_parallel(),
                # "mlp.down_proj": rowwise_parallel(output_layouts=Shard(1)),
                "mlp.down_proj": rowwise_parallel(),
                "mlp.up_proj": colwise_parallel(),
            }
        )

        parallelize_module(
            module=decode_layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


def run_fsdp2_lenient_compare(checkpoint_path: str, other_path: str, threshold: int, output_path: str | None = None, tp_size: int = 2, dp_size: int = 4):
    from torch.distributed.fsdp import (
        fully_shard,
    )
    from torch.distributed.device_mesh import init_device_mesh
    
    try:
        torch.distributed.init_process_group("nccl")
        mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh["tp"]
        dp_mesh = mesh["dp"]

        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)

        with open(other_path) as f:
            other_results = json.load(f)["results"]

        tok = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, 
            trust_remote_code=True,
            attn_implementation="flash_attention_3",
        )

        apply_non_moe_tp_transformers(model, tp_mesh, loss_parallel=False)

        modules = [
            module
            for name, module in model.named_modules()
            if module.__class__.__name__ in model._no_split_modules
            or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
        ]

        for idx, module in enumerate(modules):
            fully_shard(module, mesh=dp_mesh)
        model = fully_shard(model, mesh=dp_mesh)

        device = torch.cuda.current_device()

        summaries = []
        failures = 0

        for i, other_res in enumerate(other_results, start=1):
            prompt = other_res["prompt"]
            other_token_ids = other_res.get("token_ids")
            other_logprobs = other_res.get("logprobs")
            other_logprobs = torch.tensor(other_logprobs, device=device)
            lp_diffs, abs_lp_diffs = [], []

            with torch.inference_mode():
                # Build the full sequence: prompt + generated tokens
                prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
                full_ids = prompt_ids + other_token_ids
                
                # Single forward pass to get all logits
                input_ids = torch.tensor([full_ids], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids).logits[0]  # shape: [seq_len, vocab_size]

                pred_logits = logits[len(prompt_ids)-1:-1]  # shape: [len(other_token_ids), vocab_size]
                # torch.distributed.breakpoint()
                target_ids = torch.tensor(other_token_ids, device=device)
                # Use selective_log_softmax to efficiently get log probs for targets
                hf_logprobs = selective_log_softmax_raw(pred_logits, target_ids)

                # Calculate logprob differences
                if other_logprobs is not None:
                    for t, hf_lp in enumerate(hf_logprobs):
                        if t < len(other_logprobs):
                            diff = hf_lp.item() - other_logprobs[t].item()
                            lp_diffs.append(diff)
                            abs_lp_diffs.append(abs(diff))

            if torch.distributed.get_rank() == 0:
                max_abs_lp = max(abs_lp_diffs) if abs_lp_diffs else 0.0
                min_abs_lp = min(abs_lp_diffs) if abs_lp_diffs else 0.0
                max_lp = max(lp_diffs) if lp_diffs else 0.0
                min_lp = min(lp_diffs) if lp_diffs else 0.0
                mean_abs_lp = sum(abs_lp_diffs) / len(abs_lp_diffs) if abs_lp_diffs else 0.0
                
                if max_abs_lp > threshold:
                    failures += 1
                    status = "✗"
                else:
                    status = "✓"
                    
                print(f"{status} Prompt {i}: | "
                    f"logprob_diff: max={max_lp:.10f}, min={min_lp:.10f}, "
                    f"max_abs={max_abs_lp:.10f}, min_abs={min_abs_lp:.10f}, mean_abs={mean_abs_lp:.10f}")

                summaries.append({
                    "prompt": prompt,
                    "logprob_diffs": lp_diffs,
                    "max_logprob_diff": max_lp,
                    "min_logprob_diff": min_lp,
                    "max_abs_logprob_diff": max_abs_lp,
                    "min_abs_logprob_diff": min_abs_lp,
                    "mean_abs_logprob_diff": mean_abs_lp,
                })

        if torch.distributed.get_rank() == 0:
            all_lp_diffs = [d for s in summaries for d in s.get("logprob_diffs", [])]
            all_abs_diffs = [abs(d) for d in all_lp_diffs]
            overall_max_lp = max(all_lp_diffs) if all_lp_diffs else 0.0
            overall_min_lp = min(all_lp_diffs) if all_lp_diffs else 0.0
            overall_max_abs_lp = max(all_abs_diffs) if all_abs_diffs else 0.0
            overall_min_abs_lp = min(all_abs_diffs) if all_abs_diffs else 0.0
            overall_mean_abs_lp = sum(all_abs_diffs) / len(all_abs_diffs) if all_abs_diffs else 0.0
            
            print(f"\nOverall: failures={failures}")
            print(f"Logprob diff: max={overall_max_lp:.10f}, min={overall_min_lp:.10f}")
            print(f"Logprob |diff|: max={overall_max_abs_lp:.10f}, min={overall_min_abs_lp:.10f}, mean={overall_mean_abs_lp:.10f}")

            if output_path:
                with open(output_path, "w") as f:
                    json.dump({"checkpoint_path": checkpoint_path, "prompts": summaries}, f, indent=2)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(
        {
            "run_sglang": run_sglang_inference,
            "run_hf_lenient_compare": run_hf_lenient_compare,
            "run_fsdp2_lenient_compare": run_fsdp2_lenient_compare,
        }
    )