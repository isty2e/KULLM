import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast


def main(args):
    BASE_MODEL = args.base_model
    assert BASE_MODEL, (
        "Please specify a value for BASE_MODEL environment variable, e.g."
        " `export BASE_MODEL=huggyllama/llama-7b`"
    )

    base_model = GPTNeoXForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    print(f"base_model vocab size: {model_vocab_size}")

    print(f"Loading LoRA tokenizer from {args.lora_path}...")
    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.lora_path)
    tokenizer_vocab_size = len(tokenizer)
    print(f"tokenizer vocab size: {tokenizer_vocab_size}")

    if model_vocab_size != tokenizer_vocab_size:
        base_model.resize_token_embeddings(tokenizer_vocab_size)
        print(f"Extended vocabulary size to {tokenizer_vocab_size}")

    print("Loading LoRA weights")
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.lora_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )
    lora_model.merge_and_unload()
    lora_model.save_pretrained(args.output_dir, safe_serialization=True)

    # lora_model_sd = lora_model.state_dict()
    # deloreanized_sd = {
    #     k.replace("base_model.gpt_neox.", ""): v
    #     for k, v in lora_model_sd.items()
    #     if "lora" not in k
    # }

    # GPTNeoXForCausalLM.save_pretrained(
    #     base_model, save_directory=args.output_dir, state_dict=deloreanized_sd
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="EleutherAI/polyglot-ko-12.8b")
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path, default="ckpt/polyglot-13b-kullm_v3-3e-5"
    )

    args = parser.parse_args()

    main(args)
