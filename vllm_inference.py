import argparse
from pathlib import Path
from typing import Optional

import polars as pl
from datasets import load_dataset
from vllm import LLM, SamplingParams

from utils.prompter import Prompter

prompter = Prompter("kullm")


def get_ids_and_prompts(
    data_path: Path, max_samples: Optional[int] = None, shuffle: bool = False
) -> tuple[list[str], list[str]]:
    dataset = load_dataset("json", data_files=str(data_path))
    if shuffle:
        dataset = dataset.shuffle()
    data_ids, prompts = [], []

    for i, data in enumerate(dataset["train"]):
        if max_samples is not None and i >= max_samples:
            break

        data_id = data["id"]
        instruction = data["instruction"]
        input_text = data["input"]

        prompt = prompter.generate_prompt(instruction, input_text)

        data_ids.append(data_id)
        prompts.append(prompt)

    assert len(data_ids) == len(prompts)

    return data_ids, prompts


def main(args):
    print("Loading data")
    data_ids, prompts = get_ids_and_prompts(args.data, args.max_samples, args.shuffle)

    print("Loading model")
    # Tensor parallelism won't work because of divisibility
    llm = LLM(model=str(args.model), tokenizer="nlpai-lab/kullm-polyglot-12.8b-v2")

    sampling_kwargs = {
        "best_of": 5,
        "frequency_penalty": 0.05,
        "max_tokens": args.max_tokens,
        "stop": ["</끝>", "<|endoftext|>"],
    }

    if args.use_beam_search:
        sampling_kwargs |= {
            "temperature": 0.0,
            "use_beam_search": True,
        }
    else:
        sampling_kwargs |= {
            "temperature": 0.2,
            "top_k": 50,
            "top_p": 0.95,
        }
    sampling_params = SamplingParams(**sampling_kwargs)

    print("Generating texts")
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    df = pl.DataFrame(
        {"TextHandle": data_ids, "Korean (Model)": generated_texts}
    )
    df.write_parquet(args.output_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model_str", default="Model")
    parser.add_argument("--output_df", type=Path)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--use_beam_search", action="store_true")
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()

    if args.output_df is None:
        args.output_df = Path(f"{args.model_str}.parquet")

    main(args)
