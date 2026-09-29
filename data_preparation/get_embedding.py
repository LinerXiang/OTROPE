import os
import json
import torch

from argparse import ArgumentParser
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM
from safetensors.torch import save_file


## =========================================================
## Templates for LLM-as-a-judge embeddings
## =========================================================

PAIR_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n"
    "[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n"
    "[The End of Assistant B's Answer]"
)

A_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n"
    "[The End of Assistant A's Answer]"
)

B_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n"
    "[The End of Assistant B's Answer]"
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


## =========================================================
## Argument Parsing
## =========================================================

parser = ArgumentParser()

parser.add_argument("--path", type=str, default="~/data/embeddings")
parser.add_argument("--input_file", type=str, default="generations.json")

parser.add_argument(
    "--embedding_model",
    type=str,
    default="Qwen/Qwen3-Embedding-0.6B"
)

parser.add_argument(
    "--embed_type",
    type=str,
    default="two_prompt_answer",
    choices=["prompt_only", "prompt_and_answer", "two_prompt_answer"]
)

parser.add_argument("--id_name", type=str, default="id")
parser.add_argument("--candidate_name", type=str, default="candidate")
parser.add_argument("--opponent_name", type=str, default="opponent")
parser.add_argument("--whether_train_col", type=str, default="whether_train")
parser.add_argument("--ground_truth_col", type=str, default="ground_truth")

parser.add_argument("--train_out_name", type=str, default=None)
parser.add_argument("--test_out_name", type=str, default=None)

parser.add_argument("--max_emb_tokens", type=int, default=8150)

args = parser.parse_args()

path = os.path.expanduser(args.path)
input_file = args.input_file
data_path = os.path.join(path, input_file)

embedding_model = args.embedding_model
embedding_model_name = embedding_model.split("/")[-1]

embed_type = args.embed_type
id_name = args.id_name
candidate_name = args.candidate_name
opponent_name = args.opponent_name
whether_train_col = args.whether_train_col
ground_truth_col = args.ground_truth_col
MAX_EMB_TOKENS = args.max_emb_tokens

if args.train_out_name is None:
    train_out_name = f"embed_{embedding_model_name}_train.safetensors"
else:
    train_out_name = args.train_out_name

if args.test_out_name is None:
    test_out_name = f"embed_{embedding_model_name}_test.safetensors"
else:
    test_out_name = args.test_out_name


## =========================================================
## Load Dataset
## =========================================================

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

ds = Dataset.from_list(raw_data)
df = ds.to_pandas()

required_cols = [
    id_name,
    "prompt",
    candidate_name,
    opponent_name,
    whether_train_col,
]

missing_cols = [c for c in required_cols if c not in df.columns]
if len(missing_cols) > 0:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Canonical string order matches the score CSV loader.
if df[id_name].isna().any():
    raise ValueError("Embedding IDs must not be missing")
df[id_name] = df[id_name].astype(str)
if df[id_name].duplicated().any():
    raise ValueError("Embedding IDs must be unique")

# sort by id_name before split
df = df.sort_values(by=id_name).reset_index(drop=True)
print(df[id_name][:5])
print(f"Loaded {len(df)} examples from {data_path}")
print(f"Dataset sorted by {id_name}")

# split by whether_train
train_df = (
    df[df[whether_train_col] == 1]
    .copy()
    .sort_values(by=id_name)
    .reset_index(drop=True)
)
print(train_df[id_name])

test_df = (
    df[df[whether_train_col] == 0]
    .copy()
    .sort_values(by=id_name)
    .reset_index(drop=True)
)

print(f"Train examples (where {whether_train_col} == 1): {len(train_df)}")
print(f"Test examples  (where {whether_train_col} == 0): {len(test_df)}")


## =========================================================
## Load Embedding Model
## =========================================================

model = LLM(model=embedding_model, task="embed")
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)


def truncate_for_embed(text, max_tokens=MAX_EMB_TOKENS):
    ids = embedding_tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return embedding_tokenizer.decode(ids, skip_special_tokens=True)


def extract_embeddings(vllm_outputs):
    return [x.outputs.embedding for x in vllm_outputs]


## =========================================================
## Build embedding text
## =========================================================

def build_pair_text(prompt, candidate, opponent, mode="prompt_and_answer"):
    if mode == "prompt_only":
        text = prompt
    elif mode == "prompt_and_answer":
        text = PAIR_TEMPLATE.format(
            question=prompt,
            answer_a=candidate,
            answer_b=opponent,
        )
    else:
        raise ValueError(f"Unsupported pair mode: {mode}")

    return truncate_for_embed(text)


def build_sep_texts(prompt, candidate, opponent):
    text_a = A_TEMPLATE.format(
        question=prompt,
        answer_a=candidate,
    )
    text_b = B_TEMPLATE.format(
        question=prompt,
        answer_b=opponent,
    )

    text_a = truncate_for_embed(text_a)
    text_b = truncate_for_embed(text_b)

    return text_a, text_b


## =========================================================
## Generate embeddings
## =========================================================

@torch.no_grad()
def generate_pair_embeddings(sub_df, model):
    prompts = sub_df["prompt"].tolist()
    candidates = sub_df[candidate_name].tolist()
    opponents = sub_df[opponent_name].tolist()

    texts = []
    for p, c, o in zip(prompts, candidates, opponents):
        texts.append(build_pair_text(p, c, o, embed_type))

    outputs = model.embed(texts)
    embeddings = extract_embeddings(outputs)

    return embeddings


@torch.no_grad()
def generate_sep_embeddings(sub_df, model):
    prompts = sub_df["prompt"].tolist()
    candidates = sub_df[candidate_name].tolist()
    opponents = sub_df[opponent_name].tolist()

    texts_a = []
    texts_b = []

    for p, c, o in zip(prompts, candidates, opponents):
        ta, tb = build_sep_texts(p, c, o)
        texts_a.append(ta)
        texts_b.append(tb)

    outputs_a = model.embed(texts_a)
    outputs_b = model.embed(texts_b)

    emb_a = extract_embeddings(outputs_a)
    emb_b = extract_embeddings(outputs_b)

    return emb_a, emb_b


## =========================================================
## Save helpers
## =========================================================

def maybe_add_int_column(save_dict, sub_df, col_name, save_name=None):
    if col_name in sub_df.columns:
        out_name = save_name if save_name is not None else col_name
        save_dict[out_name] = torch.tensor(
            sub_df[col_name].tolist(),
            dtype=torch.int32
        )
        print(f"Saved column: {out_name}")


def build_save_dict(sub_df):
    """
    Return:
        save_dict: tensors to save into safetensors
        original_ids: original id list (may be strings), saved separately to json
    """
    original_ids = sub_df[id_name].tolist()

    save_dict = {
        "row_indices": torch.arange(len(sub_df), dtype=torch.long),
    }

    if embed_type in ["prompt_only", "prompt_and_answer"]:
        embeddings = generate_pair_embeddings(sub_df, model)
        save_dict["embeddings"] = torch.tensor(embeddings, dtype=torch.float32)
    else:
        emb_a, emb_b = generate_sep_embeddings(sub_df, model)
        save_dict["embeddings_A"] = torch.tensor(emb_a, dtype=torch.float32)
        save_dict["embeddings_B"] = torch.tensor(emb_b, dtype=torch.float32)

    maybe_add_int_column(save_dict, sub_df, ground_truth_col, "ground_truth")

    return save_dict, original_ids


def build_metadata(sub_df, split_name):
    metadata = {
        "embedding_model": embedding_model,
        "embedding_model_name": embedding_model_name,
        "embed_type": embed_type,
        "id_column": id_name,
        "candidate_column": candidate_name,
        "opponent_column": opponent_name,
        "whether_train_column": whether_train_col,
        "ground_truth_column": ground_truth_col if ground_truth_col in sub_df.columns else "NOT_FOUND",
        "data_path": data_path,
        "num_examples": str(len(sub_df)),
        "sorted_by": id_name,
        "split_name": split_name,
        "flip_handled": "False",
        "id_saved_separately": "True",
        "id_tensor_name": "row_indices",
    }
    return metadata

def save_ids_json(ids_list, out_path_without_json_suffix):
    with open(out_path_without_json_suffix, "w", encoding="utf-8") as f:
        json.dump(ids_list, f, ensure_ascii=False, indent=2)
    print(f"Saved ids to: {out_path_without_json_suffix}")


## =========================================================
## Save train split
## =========================================================

if len(train_df) > 0:
    train_save_dict, train_ids = build_save_dict(train_df)
    train_metadata = build_metadata(train_df, "train")
    train_out_path = os.path.join(path, train_out_name)

    save_file(train_save_dict, train_out_path, metadata=train_metadata)
    save_ids_json(train_ids, train_out_path + ".ids.json")
    print(f"Saved train embeddings to: {train_out_path}")
else:
    print("Train split is empty. No train file saved.")


## =========================================================
## Save test split
## =========================================================

if len(test_df) > 0:
    test_save_dict, test_ids = build_save_dict(test_df)
    test_metadata = build_metadata(test_df, "test")
    test_out_path = os.path.join(path, test_out_name)

    save_file(test_save_dict, test_out_path, metadata=test_metadata)
    save_ids_json(test_ids, test_out_path + ".ids.json")
    print(f"Saved test embeddings to: {test_out_path}")
else:
    print("Test split is empty. No test file saved.")


print("Done.")
