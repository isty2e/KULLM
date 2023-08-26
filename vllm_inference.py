import argparse
from pathlib import Path

import polars as pl
from datasets import load_dataset
from vllm import LLM, SamplingParams

from utils.prompter import Prompter

prompter = Prompter("kullm")


def get_ids_and_prompts(data_path: Path) -> tuple[list[str], list[str]]:
    dataset = load_dataset("json", data_files=str(data_path))
    data_ids, prompts = [], []

    for data in dataset["train"]:
        data_id = data["id"]
        instruction = data["instruction"]
        input_text = data["input"]

        prompt = prompter.generate_prompt(instruction, input_text)

        data_ids.append(data_id)
        prompts.append(prompt)

    assert len(data_ids) == len(prompts)

    return data_ids, prompts


def main(args):
    data_ids, prompts = get_ids_and_prompts(args.data)

    llm = LLM(model=str(args.model), tensor_parallel_size=args.num_gpus)
    sampling_params = SamplingParams(
        best_of=5,
        frequency_penalty=0.05,
        temperature=0.2,
        max_new_tokens=1024,
        top_k=50,
        top_p=0.95,
        use_beam_search=True,
        stop=["</끝>", "<|endoftext|>"],
    )

    outputs = llm.generate(prompts, sampling_params)

    df = pl.DataFrame({"TextHandle": data_ids, "Korean (Model)": outputs})
    df.write_parquet(args.output_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model_str", default="Model")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--output_df", type=Path)

    args = parser.parse_args()

    if args.output_df is None:
        args.output_df = Path(f"{args.model_str}.parquet")

    main(args)
