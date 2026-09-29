import argparse
import json
import os
from typing import List, Optional, Any

import numpy as np
import pandas as pd
from datasets import load_dataset


def load_dataset_as_dataframe(dataset_name: str, split: str) -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    return ds.to_pandas()


def unwrap_scalar(x: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, np.ndarray):
        x = x.tolist()

    while isinstance(x, list) and len(x) == 1:
        x = x[0]
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, np.ndarray):
            x = x.tolist()

    return x


def unwrap_list(x: Any) -> List[Any]:
    if x is None:
        return []

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, list):
        return x

    return [x]


def unwrap_to_str(x: Any) -> Optional[str]:
    x = unwrap_scalar(x)

    if x is None:
        return None

    if isinstance(x, str):
        x = x.strip()
        return x if x else None

    return str(x).strip()


def to_python_obj(x: Any) -> Any:
    if isinstance(x, dict):
        return {to_python_obj(k): to_python_obj(v) for k, v in x.items()}

    if isinstance(x, list):
        return [to_python_obj(v) for v in x]

    if isinstance(x, tuple):
        return [to_python_obj(v) for v in x]

    if isinstance(x, np.ndarray):
        return to_python_obj(x.tolist())

    if isinstance(x, np.generic):
        return x.item()

    return x


def expand_row_to_pairwise(
    row: pd.Series,
    prompt_col: str,
    chosen_col: str,
    rejected_col: str,
    models_col: str,
    question_id_col: str,
) -> List[dict]:
    prompt = unwrap_to_str(row[prompt_col])
    chosen_response = unwrap_to_str(row[chosen_col])
    rejected_list = unwrap_list(row[rejected_col])
    models = unwrap_list(row[models_col])

    if prompt is None or chosen_response is None:
        return []

    if len(rejected_list) < 3 or len(models) < 4:
        return []

    chosen_model = unwrap_to_str(models[0])
    rejected_models = [unwrap_to_str(m) for m in models[1:4]]
    question_id = unwrap_scalar(row[question_id_col]) if question_id_col in row else None

    if chosen_model is None or any(m is None for m in rejected_models):
        return []

    pairwise_rows = []
    for i in range(3):
        rejected_response = unwrap_to_str(rejected_list[i])
        rejected_model = rejected_models[i]

        if rejected_response is None:
            continue

        pairwise_rows.append(
            {
                "question_id": question_id,
                "prompt": prompt,
                "chosen_model": chosen_model,
                "chosen_response": chosen_response,
                "rejected_model": rejected_model,
                "rejected_response": rejected_response,
            }
        )

    return pairwise_rows


def process_pairwise_item(
    item: dict,
    target_models: List[str],
    reference_models: List[str],
    mixed_models: List[str],
) -> Optional[dict]:
    chosen_model = unwrap_to_str(item["chosen_model"])
    rejected_model = unwrap_to_str(item["rejected_model"])

    candidate_pool = set(target_models) | set(reference_models)
    opponent_pool = set(mixed_models)

    if chosen_model in candidate_pool and rejected_model in opponent_pool:
        candidate_model = chosen_model
        opponent_model = rejected_model
        candidate_response = unwrap_to_str(item["chosen_response"])
        opponent_response = unwrap_to_str(item["rejected_response"])
        label = 1.0
        if candidate_model in target_models:
            whether_train = 0
        elif candidate_model in reference_models:
            whether_train = 1
        else:
            return None  # safety

    elif rejected_model in candidate_pool and chosen_model in opponent_pool:
        candidate_model = rejected_model
        opponent_model = chosen_model
        candidate_response = unwrap_to_str(item["rejected_response"])
        opponent_response = unwrap_to_str(item["chosen_response"])
        label = 0.0
        if candidate_model in target_models:
            whether_train = 0
        elif candidate_model in reference_models:
            whether_train = 1
        else:
            return None  # safety

    else:
        return None

    if candidate_response is None or opponent_response is None:
        return None

    return {
        "question_id": unwrap_scalar(item["question_id"]),
        "candidate_model": candidate_model,
        "opponent_model": opponent_model,
        "label": label,
        "prompt": unwrap_to_str(item["prompt"]),
        "candidate_response": candidate_response,
        "opponent_response": opponent_response,
        "whether_train": whether_train,
    }


def build_processed_data_reward_bench_2(
    df: pd.DataFrame,
    prompt_col: str,
    chosen_col: str,
    rejected_col: str,
    models_col: str,
    subset_col: str,
    question_id_col: str,
    target_models: List[str],
    reference_models: List[str],
    mixed_models: List[str],
) -> List[dict]:
    processed_rows = []

    df = df[df[subset_col] != "Ties"].copy()

    for _, row in df.iterrows():
        pairwise_items = expand_row_to_pairwise(
            row=row,
            prompt_col=prompt_col,
            chosen_col=chosen_col,
            rejected_col=rejected_col,
            models_col=models_col,
            question_id_col=question_id_col,
        )

        for item in pairwise_items:
            processed = process_pairwise_item(
                item=item,
                target_models=target_models,
                reference_models=reference_models,
                mixed_models=mixed_models,
            )
            if processed is not None:
                processed_rows.append(processed)

    for i, row in enumerate(processed_rows):
        row["id"] = str(i)
    return processed_rows


def save_as_json(data, output_json: str) -> None:
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data = to_python_obj(data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pairwise data from allenai/reward-bench-2")

    parser.add_argument("--dataset-name", type=str, default="allenai/reward-bench-2")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--prompt-col", type=str, default="prompt")
    parser.add_argument("--chosen-col", type=str, default="chosen")
    parser.add_argument("--rejected-col", type=str, default="rejected")
    parser.add_argument("--models-col", type=str, default="models")
    parser.add_argument("--subset-col", type=str, default="subset")
    parser.add_argument("--question-id-col", type=str, default="id")

    parser.add_argument("--target-models", type=str, nargs="+", required=True)
    parser.add_argument("--reference-models", type=str, nargs="+", required=True)
    parser.add_argument("--mixed-models", type=str, nargs="+", required=True)

    parser.add_argument("--output-json", type=str, default="processed_reward_bench_2_pairs.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset_as_dataframe(args.dataset_name, args.split)

    processed_data = build_processed_data_reward_bench_2(
        df=df,
        prompt_col=args.prompt_col,
        chosen_col=args.chosen_col,
        rejected_col=args.rejected_col,
        models_col=args.models_col,
        subset_col=args.subset_col,
        question_id_col=args.question_id_col,
        target_models=args.target_models,
        reference_models=args.reference_models,
        mixed_models=args.mixed_models,
    )
    labels_train0 = [x["label"] for x in processed_data if x["whether_train"] == 0]
    labels_train1 = [x["label"] for x in processed_data if x["whether_train"] == 1]
    print("count train=0:", len(labels_train0))
    print("count train=1:", len(labels_train1))
    print("mean label (whether_train=0):", np.mean(labels_train0))
    print("mean label (whether_train=1):", np.mean(labels_train1))

    save_as_json(processed_data, args.output_json)

    print(f"Saved processed data to: {args.output_json}")
    print(f"Number of pairwise rows kept: {len(processed_data)}")


if __name__ == "__main__":
    main()
