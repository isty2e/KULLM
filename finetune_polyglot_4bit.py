import os
import sys
from pathlib import Path
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.prompter import Prompter


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(
            checkpoint_folder, "pytorch_model.bin"
        )
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def train(
    # model/data params
    # the only required argument
    base_model: str = "nlpai-lab/kullm-polyglot-12.8b-v2",
    data_path: str = "/data/kullm-v2",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 32,
    micro_batch_size: int = 8,
    num_epochs: int = 4,
    learning_rate: float = 1e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 2000,
    eval_interval: int = 200,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["query_key_value", "xxx"],
    # llm hyperparams
    # if False, masks out inputs in loss
    train_on_inputs: bool = True,
    add_eos_token: bool = False,
    # faster, but produces an odd training loss curve
    group_by_length: bool = False,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    # options: false | gradients | all
    wandb_watch: str = "",
    # options: false | true
    wandb_log_model: str = "",
    # either training checkpoint or final adapter
    resume_from_checkpoint: str = None,
    # The prompt template to use, will default to alpaca.
    prompt_template_name: str = "conditional_translation",
    use_prompter: bool = True,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            "Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = wandb_project != "" or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if wandb_project != "":
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_watch != "":
        os.environ["WANDB_WATCH"] = wandb_watch
    if wandb_log_model != "":
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # model = GPTNeoXForCausalLM.from_pretrained(
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    if "quantumai" in base_model:
        padding_side = "right"
        use_fast = False
    else:
        padding_side = "left"
        use_fast = True

    # tokenizer = GPTNeoXTokenizerFast.from_pretrained(base_model)
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, padding_side=padding_side, use_fast=use_fast
    )

    is_llama = False
    for model_name in ("llama", "quantumaikr", "komt"):
        if model_name in base_model.lower():
            is_llama = True
            break

    # unk. we want this to be different from the eos token
    if is_llama:
        tokenizer.pad_token_id = 0
    # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        if train_on_inputs:
            return tokenized_full_prompt

        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        # could be sped up, probably
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def tokenize_prompt(data_point):
        prompt = data_point["prompt"]
        output = data_point["output"]

        if train_on_inputs:
            if output:
                prompt = f"{prompt}{output}"
            return tokenize(prompt)

        tokenized_prompt = tokenize(prompt, add_eos_token=add_eos_token)
        prompt_len = len(tokenized_prompt["input_ids"])

        if add_eos_token:
            prompt_len -= 1

        tokenized_prompt["labels"] = [-100] * prompt_len + tokenized_prompt[
            "labels"
        ][prompt_len:]

        return tokenized_prompt

    model = prepare_model_for_kbit_training(model)

    if is_llama:
        lora_target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if Path(data_path).suffix in {".json", ".jsonl"}:
        data = load_dataset("json", data_files=str(data_path))
    elif Path(data_path).suffix == ".parquet":
        data = load_dataset("parquet", data_files=str(data_path))
    else:
        data = load_dataset(str(data_path))

    if resume_from_checkpoint:
        # Check the available weights and load them
        # Full checkpoint
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            # only LoRA model - LoRA config above has to fit
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            # So the trainer won't try loading its state
            resume_from_checkpoint = False
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    if use_prompter:
        input_to_tokens = generate_and_tokenize_prompt
    else:
        input_to_tokens = tokenize_prompt

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(input_to_tokens)
        val_data = train_val["test"].shuffle().map(input_to_tokens)
    else:
        train_data = data["train"].shuffle().map(input_to_tokens)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_interval if val_set_size > 0 else None,
            save_steps=eval_interval,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=val_set_size > 0,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
