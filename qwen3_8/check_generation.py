"""Compare a short greedy text continuation on the checkpoint and the V2 runner."""

import argparse
import json
import os


PROMPTS = ("The first number is", "What is 4 + 3?")


def transformers_tokens(path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.float32
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    outputs = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            start = inputs.shape[1]
            for _ in range(4):
                token = (
                    model(input_ids=inputs, use_cache=False)
                    .logits[:, -1]
                    .argmax(-1, keepdim=True)
                )
                inputs = torch.cat([inputs, token], dim=1)
                if token.item() == tokenizer.eos_token_id:
                    break
            outputs.append(inputs[0, start:].tolist())
    return outputs


def v2_tokens(path):
    # vLLM reads these when it initializes the platform and chooses its runner.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["TORCH_CACHING_PRECOMPILE"] = "0"
    os.environ["VLLM_NEURON_DISABLE_PARALLEL_TRACE"] = "1"
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=path,
        dtype="bfloat16",
        tensor_parallel_size=8,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        num_gpu_blocks_override=64,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        generation_config="vllm",
        hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {},
                "num_batched_tokens_buckets": [512],
                "num_seqs_buckets": [1, 2],
            }
        },
    )
    return [
        item.outputs[0].token_ids
        for item in engine.generate(
            list(PROMPTS), SamplingParams(temperature=0, max_tokens=4)
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("hf", "v2"))
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--reference", help="Compare against the JSON saved by --output on the HF run"
    )
    parser.add_argument("--output", help="Save prompt/token pairs to this JSON file")
    args = parser.parse_args()
    tokens = (transformers_tokens if args.backend == "hf" else v2_tokens)(
        args.checkpoint
    )
    result = dict(zip(PROMPTS, tokens, strict=True))
    print(json.dumps(result))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    if args.reference:
        with open(args.reference) as handle:
            expected = json.load(handle)
        if result != expected:
            raise SystemExit(f"Greedy token mismatch; HF reference: {expected}")
