import argparse
from pathlib import Path
from typing import Union

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

from utils.prompter import Prompter

prompter = Prompter("kullm")


def get_model(
    base_model: str, lora_path: Union[Path, str]
) -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        base_model,
        # load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.merge_and_unload()
    model.eval()

    return model


def get_tokenizer(base_model: str) -> GPTNeoXTokenizerFast:
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(base_model)
    tokenizer.padding_side = "left"

    return tokenizer


def generate(
    instruction: str,
    input_text: str,
    model: GPTNeoXForCausalLM,
    tokenizer: GPTNeoXTokenizerFast,
):
    prompt = prompter.generate_prompt(instruction, input_text)

    inputs = tokenizer(prompt, padding=False, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    gen_tokens = model.generate(
        inputs=input_ids,
        max_new_tokens=512,
    )
    gen_text = (
        tokenizer.batch_decode(gen_tokens)[0]
        .split("### 응답:")[1]
        .split("<|endoftext|>")[0]
        .strip()
    )

    return gen_text


def main(args):
    model = get_model(args.base_model, args.lora_model)
    tokenizer = get_tokenizer(args.base_model)
    dataset = load_dataset("json", data_files=str(args.data))

    # pipe = get_pipe(model, tokenizer)

    i = 0
    for data in dataset["train"]:
        instruction = data["instruction"]
        input_text = data["input"]
        output_text = data["output"]
        output_pred = generate(instruction, input_text, model, tokenizer)
        # output_pred = generate(instruction, input_text, pipe)

        print("#####################################")
        print("################Input:")
        print(input_text)
        print("################Output:")
        print(output_text)
        print("################Pred:")
        print(output_pred)

        if i > 10:
            break
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--base_model", default="beomi/KoAlpaca-Polyglot-12.8B"
    )
    parser.add_argument("--lora_model", type=Path)

    args = parser.parse_args()
    main(args)
