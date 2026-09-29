import os
import csv
import json
import numpy as np

from argparse import ArgumentParser


# =========================================================
# Argument Parsing
# =========================================================
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str, required=True
)
parser.add_argument("--target_filename", type=str, default="judge")
parser.add_argument("--target_variable", type=str, required=True)
parser.add_argument("--ground_truth", type=str, default="ground_truth_judged")
parser.add_argument("--id_name", type=str, default="question_id")
parser.add_argument("--whether_train_col", type=str, default="whether_train")


args = parser.parse_args()

base_path = os.path.expanduser(args.base_path)
target_filename = args.target_filename
target_variable = args.target_variable
ground_truth = args.ground_truth
id_name = args.id_name
whether_train_col = args.whether_train_col


# =========================================================
# Utilities
# =========================================================
def normalize_id(x):
    """Convert id to string for stable sorting/alignment."""
    return str(x)

def normalize_model_name(name: str) -> str:
    """
    Normalize model filename into a clean model name for CSV headers.

    Examples:
      judge_CodeLlama-7b-Instruct-hf.json -> CodeLlama-7B-Instruct-hf
      judge_gemma-2-2b-it.json           -> Gemma-2-2B-IT
    """
    name = str(name).strip()

    if name.startswith("judge_"):
        name = name[len("judge_"):]

    if name.endswith(".json"):
        name = name[:-5]

    return name

def value_to_float_label(val, dataset_path, _id, column):
    """
    Convert value into float label.
    Only supports:
      - 0
      - 1
      - 0.5
      - [0], [1], [0.5]
    """
    if isinstance(val, list):
        if len(val) != 1:
            raise ValueError(
                f"id={_id}: '{column}' has length {len(val)} in {dataset_path}"
            )
        val = val[0]

    try:
        val = float(val)
    except Exception:
        raise ValueError(
            f"id={_id}: '{column}' must be one of 0/1/0.5 in {dataset_path}, got {val}"
        )

    if val not in [0.0, 0.5, 1.0]:
        raise ValueError(
            f"id={_id}: '{column}' must be one of 0/1/0.5 in {dataset_path}, got {val}"
        )

    return val


def save_split_csv(id_list, expert_outputs, labels, out_path, model_names):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    E, N = expert_outputs.shape

    header = ["id"] + model_names + ["label"]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(N):
            writer.writerow([str(id_list[i])] +
                            [float(expert_outputs[e, i]) for e in range(E)] +
                            [float(labels[i])])

    print(f"Saved CSV to: {out_path}")


# =========================================================
# Read expert file
# =========================================================
def expert_output(dataset_path, column, id_name="question_id", whether_train_col="whether_train"):
    """
    Return a dict:
        id(str) -> {
            "pred": float in {0, 0.5, 1},
            "whether_train": int in {0, 1}
        }
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_info = {}
    for item in dataset:
        if id_name not in item:
            raise ValueError(
                f"Missing '{id_name}' in one item of {dataset_path}. Keys: {list(item.keys())}"
            )
        if column not in item:
            raise ValueError(
                f"Missing '{column}' in one item of {dataset_path}. Keys: {list(item.keys())}"
            )
        if whether_train_col not in item:
            raise ValueError(
                f"Missing '{whether_train_col}' in one item of {dataset_path}. Keys: {list(item.keys())}"
            )

        _id = normalize_id(item[id_name])

        if _id in id_to_info:
            raise ValueError(f"Duplicate {id_name}={_id} in {dataset_path}")

        pred = value_to_float_label(item[column], dataset_path, _id, column)
        whether_train = int(item[whether_train_col])

        if whether_train not in [0, 1]:
            raise ValueError(
                f"id={_id}: '{whether_train_col}' must be 0 or 1 in {dataset_path}, got {whether_train}"
            )

        id_to_info[_id] = {
            "pred": pred,
            "whether_train": whether_train,
        }

    return id_to_info


# =========================================================
# Read ground truth file
# =========================================================
def ground_truth_info(dataset_path, label_col="label", id_name="question_id", whether_train_col="whether_train"):
    """
    Return a dict:
        id(str) -> {
            "label": float in {0, 0.5, 1},
            "whether_train": int in {0, 1}
        }
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_info = {}
    for item in dataset:
        if id_name not in item:
            raise ValueError(
                f"Missing '{id_name}' in one item of {dataset_path}. Keys: {list(item.keys())}"
            )
        if label_col not in item:
            raise ValueError(
                f"Missing '{label_col}' in one item of {dataset_path}. Keys: {list(item.keys())}"
            )
        if whether_train_col not in item:
            raise ValueError(
                f"Missing '{whether_train_col}' in one item of {dataset_path}. Keys: {list(item.keys())}"
            )

        _id = normalize_id(item[id_name])

        if _id in id_to_info:
            raise ValueError(f"Duplicate {id_name}={_id} in {dataset_path}")

        label = value_to_float_label(item[label_col], dataset_path, _id, label_col)
        whether_train = int(item[whether_train_col])

        if whether_train not in [0, 1]:
            raise ValueError(
                f"id={_id}: '{whether_train_col}' must be 0 or 1 in {dataset_path}, got {whether_train}"
            )

        id_to_info[_id] = {
            "label": label,
            "whether_train": whether_train,
        }

    return id_to_info


# =========================================================
# Load all expert files
# =========================================================
all_expert_dicts = []
all_model_name = []

for fname in sorted(os.listdir(base_path)):
    if fname.startswith(target_filename) and fname.endswith(".json"):
        json_path = os.path.join(base_path, fname)

        if os.path.isfile(json_path):
            print(f"Found expert file: {json_path}")
            clean_model_name = normalize_model_name(fname)
            all_model_name.append(clean_model_name)
            all_expert_dicts.append(
                expert_output(
                    json_path,
                    target_variable,
                    id_name=id_name,
                    whether_train_col=whether_train_col,
                )
            )

if len(all_expert_dicts) == 0:
    raise ValueError(
        f"No files starting with '{target_filename}' found under {base_path}"
    )


# =========================================================
# Strict alignment across experts
# =========================================================
base_ids = set(all_expert_dicts[0].keys())
for i, d in enumerate(all_expert_dicts):
    if set(d.keys()) != base_ids:
        missing = sorted(list(base_ids - set(d.keys())))[:10]
        extra = sorted(list(set(d.keys()) - base_ids))[:10]
        raise ValueError(
            f"ID mismatch in {all_model_name[i]}.\n"
            f"Missing (up to 10): {missing}\n"
            f"Extra (up to 10): {extra}"
        )

# check whether_train consistency across experts
for _id in base_ids:
    wt0 = all_expert_dicts[0][_id]["whether_train"]
    for i in range(1, len(all_expert_dicts)):
        wti = all_expert_dicts[i][_id]["whether_train"]
        if wti != wt0:
            raise ValueError(
                f"whether_train mismatch for id={_id} across expert files: "
                f"{all_model_name[0]} has {wt0}, {all_model_name[i]} has {wti}"
            )


# =========================================================
# Load ground truth
# =========================================================
ground_truth_json_path = os.path.join(base_path, f"{ground_truth}.json")
gt_dict = ground_truth_info(
    ground_truth_json_path,
    label_col="label",
    id_name=id_name,
    whether_train_col=whether_train_col,
)

if set(gt_dict.keys()) != base_ids:
    missing = sorted(list(base_ids - set(gt_dict.keys())))[:10]
    extra = sorted(list(set(gt_dict.keys()) - base_ids))[:10]
    raise ValueError(
        f"ID mismatch between expert files and ground truth.\n"
        f"Missing in ground truth (up to 10): {missing}\n"
        f"Extra in ground truth (up to 10): {extra}"
    )

# check whether_train consistency with ground truth
for _id in base_ids:
    wt_expert = all_expert_dicts[0][_id]["whether_train"]
    wt_gt = gt_dict[_id]["whether_train"]
    if wt_expert != wt_gt:
        raise ValueError(
            f"whether_train mismatch for id={_id} between experts and ground truth: "
            f"experts have {wt_expert}, ground truth has {wt_gt}"
        )


# =========================================================
# Split ids by whether_train and sort by id
# =========================================================
train_ids = sorted([_id for _id in base_ids if all_expert_dicts[0][_id]["whether_train"] == 1])
test_ids = sorted([_id for _id in base_ids if all_expert_dicts[0][_id]["whether_train"] == 0])

print("Train size:", len(train_ids))
print("Test size:", len(test_ids))


# =========================================================
# Build split tensors
# expert_outputs shape = [num_experts, N_split]
# label shape = [N_split]
# =========================================================
def build_split_arrays(id_list, all_expert_dicts, gt_dict):
    E = len(all_expert_dicts)
    N = len(id_list)

    expert_outputs = np.zeros((E, N), dtype=np.float32)
    for e, d in enumerate(all_expert_dicts):
        expert_outputs[e, :] = np.array([d[_id]["pred"] for _id in id_list], dtype=np.float32)

    labels = np.array([gt_dict[_id]["label"] for _id in id_list], dtype=np.float32)

    return expert_outputs, labels


train_expert_outputs, train_labels = build_split_arrays(train_ids, all_expert_dicts, gt_dict)
test_expert_outputs, test_labels = build_split_arrays(test_ids, all_expert_dicts, gt_dict)

print("train expert_outputs shape:", train_expert_outputs.shape)
print("test expert_outputs shape:", test_expert_outputs.shape)


# =========================================================
# Save train/test separately
# =========================================================
train_out_path = os.path.join(base_path, "combine_train.csv")
test_out_path = os.path.join(base_path, "combine_test.csv")

save_split_csv(
    train_ids,
    train_expert_outputs,
    train_labels,
    train_out_path,
    all_model_name,
)

save_split_csv(
    test_ids,
    test_expert_outputs,
    test_labels,
    test_out_path,
    all_model_name,
)

print("Done.")
print("Expert order:", all_model_name)
