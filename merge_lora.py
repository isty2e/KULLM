import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def main(args):
    assert args.base_model, (
        "Please specify a value for BASE_MODEL environment variable, e.g."
        " `export BASE_MODEL=huggyllama/llama-7b`"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading LoRA weights")
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.lora_path,
        torch_dtype=torch.float16,
    )

    print("Merging LoRA weights")
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    print("Saving the merged model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="nlpai-lab/kullm-polyglot-12.8b-v2")
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path, default="ckpt/polyglot-13b-kullm_v3-3e-5"
    )

    args = parser.parse_args()

    main(args)
