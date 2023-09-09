import argparse
from pathlib import Path
from typing import Optional

import polars as pl
from datasets import load_dataset
from vllm import LLM, SamplingParams

from utils.prompter import Prompter

prompter = Prompter("conditional_translation")


def get_ids_and_prompts(
    data_path: Path,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    use_prompter: bool = False,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    if Path(data_path).suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=data_path)
    elif Path(data_path).suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=data_path)
    else:
        dataset = load_dataset(data_path)

    if shuffle:
        dataset = dataset.shuffle()

    data_ids, prompts = [], []

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(dataset["train"])

    for i in range(end_idx - start_idx + 1):
        if max_samples is not None and i >= max_samples:
            break

        data = dataset["train"][start_idx + i]

        data_id = data["id"]

        if use_prompter:
            instruction = data["instruction"]
            input_text = data["input"]
            prompt = prompter.generate_prompt(instruction, input_text)
        else:
            prompt = data["prompt"]

        data_ids.append(data_id)
        prompts.append(prompt)

    assert len(data_ids) == len(prompts)

    return data_ids, prompts


def main(args):
    print("Loading data")
    data_ids, prompts = get_ids_and_prompts(
        args.data,
        args.max_samples,
        args.shuffle,
        args.use_prompter,
        args.start_idx,
        args.end_idx,
    )

    print("Loading model")
    # Tensor parallelism won't work because of divisibility
    llm = LLM(model=str(args.model), tokenizer=args.base_model)

    sampling_kwargs = {
        "best_of": args.best_of,
        "presence_penalty": 0.02,
        "frequency_penalty": 0.2,
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
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
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
    parser.add_argument(
        "--base_model", default="nlpai-lab/kullm-polyglot-12.8b-v2"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model_str", default="Model")
    parser.add_argument("--output_df", type=Path)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--use_beam_search", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--use_prompter", action="store_true")
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.1)

    args = parser.parse_args()

    if args.output_df is None:
        args.output_df = Path(f"{args.model_str}.parquet")

    main(args)
