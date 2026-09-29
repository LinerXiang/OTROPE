import json
import os
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from vllm import LLM

if __package__:
    from .gen_vllm_judge import generate_response_vllm
    from .load_prompt import load_all_prompt
else:
    from gen_vllm_judge import generate_response_vllm
    from load_prompt import load_all_prompt


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="result",
        metadata={"help": "Directory for outputs"},
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "Weak judge model path"},
    )
    input_json: Optional[str] = field(
        default="real_data/processed_realdata_pairs.json",
        metadata={"help": "Input processed json file"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"},
    )
    local_index: Optional[int] = field(
        default=0,
        metadata={"help": "Local shard index"},
    )
    num_shards: int = field(default=1, metadata={"help": "Number of independent judge shards"})
    use_tensor_parallel: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use tensor parallel"},
    )
    num_gen: Optional[int] = field(
        default=1,
        metadata={"help": "Number of generations per prompt"},
    )
    gpu_usage: Optional[float] = field(
        default=0.9,
        metadata={"help": "GPU memory utilization for vLLM"},
    )


def load_json_as_dataset(input_json: str) -> Dataset:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.num_shards < 1 or not 0 <= script_args.local_index < script_args.num_shards:
        parser.error("Require num_shards >= 1 and 0 <= local_index < num_shards")
    if script_args.num_gen != 1:
        parser.error("The aggregation pipeline currently requires --num_gen 1")

    output_dir = script_args.output_dir
    local_index = script_args.local_index
    set_seed(script_args.seed)

    model_name_or_path = script_args.model_path
    model_name = model_name_or_path.split("/")[-1]
    input_json = script_args.input_json
    n = script_args.num_gen
    gpu_usage = script_args.gpu_usage
    use_tensor_parallel = script_args.use_tensor_parallel

    os.makedirs(output_dir, exist_ok=True)

    sys_prompt, user_template = load_all_prompt(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    ref_model = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        gpu_memory_utilization=gpu_usage,
        swap_space=16,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4000,
    )

    world_size = script_args.num_shards

    ds = load_json_as_dataset(input_json)
    ds = ds.shuffle(seed=script_args.seed)

    print(f"shape of dataset is: {ds.shape}")

    data_size = len(ds)
    shard_start = data_size * local_index // world_size
    shard_end = data_size * (local_index + 1) // world_size

    output_path = f"{output_dir}/judge_{model_name}"

    if not use_tensor_parallel:
        ds = ds.select(
            np.arange(shard_start, shard_end)
        )
        print("size of dataset shard:", len(ds))

        if len(ds) == 0:
            with open(f"{output_path}_{local_index}.json", "w", encoding="utf-8") as handle:
                json.dump([], handle)
            raise SystemExit(0)
        df = ds.to_pandas()

        # The judge now sees response_a / response_b, not candidate / rejected.
        gen_judge_df = generate_response_vllm(
            df,
            ref_model,
            model_name,
            tokenizer,
            sys_prompt,
            user_template,
            n=n,
            chosen_name="candidate_response",
            rejected_name="opponent_response",
            local_index=local_index,
            seed=script_args.seed,
        )

        gen_judge_df = datasets.Dataset.from_pandas(gen_judge_df).to_list()

        with open(f"{output_path}_{local_index}.json", "w", encoding="utf-8") as f:
            json.dump(gen_judge_df, f, indent=2, ensure_ascii=False)

    else:
        df = ds.to_pandas()
        print("size of dataset:", len(df))

        gen_judge_df = generate_response_vllm(
            df,
            ref_model,
            model_name,
            tokenizer,
            sys_prompt,
            user_template,
            n=n,
            chosen_name="candidate_response",
            rejected_name="opponent_response",
            local_index=local_index,
            seed=script_args.seed,
        )

        gen_judge_df = datasets.Dataset.from_pandas(gen_judge_df).to_list()

        with open(f"{output_path}.json", "w", encoding="utf-8") as f:
            json.dump(gen_judge_df, f, indent=2, ensure_ascii=False)
