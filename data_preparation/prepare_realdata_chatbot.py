import argparse
import ast
import json
import math
from typing import Any, Optional, List

import pandas as pd
import os
import numpy as np
from datasets import load_dataset


def load_dataset_as_dataframe(dataset_name: str, split: str) -> pd.DataFrame:
    """Load a Hugging Face dataset split and convert it to a pandas DataFrame."""
    ds = load_dataset(dataset_name, split=split)
    return ds.to_pandas()


def maybe_parse_conversation(conversation: Any) -> Any:
    """
    Parse a stringified conversation object into a Python object when possible.
    """
    if conversation is None:
        return None

    if isinstance(conversation, (list, dict)):
        return conversation

    if isinstance(conversation, str):
        text = conversation.strip()
        if not text:
            return ""

        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    return conversation


def extract_prompt_response(conversation: Any):
    conversation = maybe_parse_conversation(conversation)

    if hasattr(conversation, "tolist"):
        conversation = conversation.tolist()

    if not isinstance(conversation, list):
        return None, None

    # collect turns
    turns = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue

        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()

        if role in {"user", "assistant"} and content:
            turns.append((role, content))

    if not turns:
        return None, None

    # response = last assistant message
    response = None
    for role, content in reversed(turns):
        if role == "assistant":
            response = content
            break

    if response is None:
        return None, None

    # find last assistant index
    last_assistant_index = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "assistant":
            last_assistant_index = i
            break

    context_turns = turns[:last_assistant_index]

    if not context_turns:
        return None, None

    # build prompt with history
    prompt_parts = []
    for role, content in context_turns:
        if role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")

    prompt = "\n\n".join(prompt_parts).strip()

    if not prompt:
        prompt = None
    if not response:
        response = None

    return prompt, response


def map_winner_to_label(
    winner_value: Any,
    candidate_is_model_a: bool,
    label_for_model_a: str,
    label_for_model_b: str,
    tie_mode: str,
) -> Optional[float]:
    """
    Convert the raw winner field into the desired numeric label.

    Label definition:
    - 1.0: candidate is better
    - 0.0: opponent is better
    - 0.5: tie
    """
    if winner_value is None:
        return None
    if isinstance(winner_value, float) and math.isnan(winner_value):
        return None

    winner_value = str(winner_value).strip()

    if winner_value[:3] == "tie":
        if tie_mode == "drop":
            return None
        elif tie_mode == "zero":
            return 0.0
        elif tie_mode == "one":
            return 1.0
        elif tie_mode == "half":
            return 0.5

    if candidate_is_model_a:
        if winner_value == label_for_model_a:
            return 1.0
        if winner_value == label_for_model_b:
            return 0.0
    else:
        if winner_value == label_for_model_b:
            return 1.0
        if winner_value == label_for_model_a:
            return 0.0

    return None


def process_single_row(
    row: pd.Series,
    model_a_col: str,
    model_b_col: str,
    conversation_a_col: str,
    conversation_b_col: str,
    winner_col: str,
    target_model: str,
    reference_model: str,
    mixed_models: List[str],
    label_for_model_a: str,
    label_for_model_b: str,
    tie_mode: str,
):
    model_a = row[model_a_col]
    model_b = row[model_b_col]

    # set target_model and reference_model as candidates, mixed models as opponents
    candidate_pool = {target_model, reference_model}
    opponent_pool = set(mixed_models)

    if row['turn'] == 1 and model_a in candidate_pool and model_b in opponent_pool:
        candidate_model = model_a
        opponent_model = model_b
        candidate_conversation = row[conversation_a_col]
        opponent_conversation = row[conversation_b_col]
        candidate_is_model_a = True
        if model_a == target_model:
            train_flag = 0
        if model_a == reference_model:
            train_flag = 1

    elif row['turn'] == 1 and model_b in candidate_pool and model_a in opponent_pool:
        candidate_model = model_b
        opponent_model = model_a
        candidate_conversation = row[conversation_b_col]
        opponent_conversation = row[conversation_a_col]
        candidate_is_model_a = False
        if model_b == target_model:
            train_flag = 0
        if model_b == reference_model:
            train_flag = 1

    else:
        return None

    label = map_winner_to_label(
        winner_value=row[winner_col],
        candidate_is_model_a=candidate_is_model_a,
        label_for_model_a=label_for_model_a,
        label_for_model_b=label_for_model_b,
        tie_mode=tie_mode,
    )
    if label is None:
        return None

    prompt1, candidate_response = extract_prompt_response(candidate_conversation)
    if not prompt1 or not candidate_response:
        return None

    prompt2, opponent_response = extract_prompt_response(opponent_conversation)
    if not opponent_response:
        return None
    if prompt1 != prompt2:
        raise ValueError(f"Different prompts for question_id={row['question_id']}")

    question_id = row['question_id']
    return {
        "question_id": question_id,
        "candidate_model": candidate_model,
        "opponent_model": opponent_model,
        "label": label,
        "prompt": prompt1,
        "candidate_response": candidate_response,
        "opponent_response": opponent_response,
        "whether_train": train_flag,
    }


def build_processed_data(
    df: pd.DataFrame,
    model_a_col: str,
    model_b_col: str,
    conversation_a_col: str,
    conversation_b_col: str,
    winner_col: str,
    target_model: str,
    reference_model: str,
    mixed_models: List[str],
    label_for_model_a: str,
    label_for_model_b: str,
    tie_mode: str,
):
    processed_rows = []


    for _, row in df.iterrows():
        item = process_single_row(
            row=row,
            model_a_col=model_a_col,
            model_b_col=model_b_col,
            conversation_a_col=conversation_a_col,
            conversation_b_col=conversation_b_col,
            winner_col=winner_col,
            target_model=target_model,
            reference_model=reference_model,
            mixed_models=mixed_models,
            label_for_model_a=label_for_model_a,
            label_for_model_b=label_for_model_b,
            tie_mode=tie_mode,
        )
        if item is not None:
            processed_rows.append(item)

    return processed_rows


def save_as_json(data, output_json: str) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real-data preference pairs.")

    parser.add_argument("--dataset-name", type=str, default="lmsys/mt_bench_human_judgments")
    parser.add_argument("--split", type=str, default="human")

    parser.add_argument("--model-a-col", type=str, default="model_a")
    parser.add_argument("--model-b-col", type=str, default="model_b")
    parser.add_argument("--conversation-a-col", type=str, default="conversation_a")
    parser.add_argument("--conversation-b-col", type=str, default="conversation_b")
    parser.add_argument("--winner-col", type=str, default="winner")

    parser.add_argument("--label-for-model-a", type=str, default="model_a")
    parser.add_argument("--label-for-model-b", type=str, default="model_b")
    parser.add_argument("--tie-mode", type=str, choices=["drop", "zero", "one", "half"], default="half")

    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--reference-model", type=str, required=True)
    parser.add_argument("--mixed-models", type=str, nargs="+", required=True)

    parser.add_argument("--output-json", type=str, default="processed_realdata_pairs.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset_as_dataframe(args.dataset_name, args.split)
    # print(df[[args.model_a_col, args.model_b_col, args.winner_col]].head(20))

    processed_data = build_processed_data(
        df=df,
        model_a_col=args.model_a_col,
        model_b_col=args.model_b_col,
        conversation_a_col=args.conversation_a_col,
        conversation_b_col=args.conversation_b_col,
        winner_col=args.winner_col,
        target_model=args.target_model,
        reference_model=args.reference_model,
        mixed_models=args.mixed_models,
        label_for_model_a=args.label_for_model_a,
        label_for_model_b=args.label_for_model_b,
        tie_mode=args.tie_mode,
    )

    labels_train0 = [x["label"] for x in processed_data if x["whether_train"] == 0]
    labels_train1 = [x["label"] for x in processed_data if x["whether_train"] == 1]
    print("count train=0:", len(labels_train0))
    print("count train=1:", len(labels_train1))
    print("mean label (whether_train=0):", np.mean(labels_train0))
    print("mean label (whether_train=1):", np.mean(labels_train1))

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    save_as_json(processed_data, args.output_json)

    print(f"Saved processed data to: {args.output_json}")
    print(f"Number of rows kept: {len(processed_data)}")


if __name__ == "__main__":
    main()
