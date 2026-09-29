"""Shared embedding/label loading, sample alignment, and PCA preprocessing."""
import json
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from sklearn.decomposition import PCA


def load_pred_and_truth(
    split_path: str
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], np.ndarray]:
    """
    Load expert predictions and ground truth from a CSV file.

    Expected CSV format:
      - one column named 'id'
      - one column named 'label'
      - multiple numeric columns corresponding to expert judgements

    Behavior:
      - sort rows by id to ensure alignment
      - treat numeric columns (except 'label') as expert outputs

    Returns:
      pred_g: np.ndarray, shape [N, num_experts]
      true_g: np.ndarray, shape [N]
      model_names: list[str]
      ids: np.ndarray of shape [N]
    """
    df = pd.read_csv(split_path, dtype={"id": str})

    print("=" * 80)
    print(f"[DEBUG] load_pred_and_truth: {split_path}")
    print(f"[DEBUG] Columns: {list(df.columns)}")
    print("=" * 80)

    # ---------- check required columns ----------
    if "id" not in df.columns:
        raise ValueError("CSV must contain 'id' column")
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")

    # ---------- sort by id ----------
    if df["id"].isna().any():
        raise ValueError("CSV IDs must not be missing")
    df["id"] = df["id"].astype(str)
    df = df.sort_values("id").reset_index(drop=True)

    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("CSV IDs must be present and unique")
    ids = df["id"].values

    # ---------- select expert columns ----------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "label" not in numeric_cols:
        raise ValueError("Column 'label' must be numeric (0/1)")

    expert_cols = [c for c in numeric_cols if c != "label"]

    if len(expert_cols) == 0:
        raise ValueError(
            f"No expert columns found. Numeric columns: {numeric_cols}"
        )

    print(f"[DEBUG] Expert columns used: {expert_cols}")

    # ---------- build arrays ----------
    pred_g = df[expert_cols].to_numpy(dtype=np.float32)
    true_g = df["label"].to_numpy(dtype=np.float32)

    model_names = expert_cols

    if not np.isfinite(pred_g).all() or not np.isfinite(true_g).all():
        raise ValueError("Predictions and labels must be finite")
    if ((pred_g < 0) | (pred_g > 1)).any() or ((true_g < 0) | (true_g > 1)).any():
        raise ValueError("Predictions and labels must lie in [0, 1]")
    return pred_g, true_g, model_names, ids



def load_X(embedding_path: str) -> np.ndarray:
    """
    Load input features X from a safetensors embedding file.

    Supports:
      - pairwise format: embeddings_A + embeddings_B
      - single format: embeddings
    """
    data = load_file(embedding_path)

    if "embeddings_A" in data and "embeddings_B" in data:
        X1 = data["embeddings_A"].cpu().numpy()
        X2 = data["embeddings_B"].cpu().numpy()
        X = np.concatenate((X1, X2), axis=1).astype(np.float32)

    elif "embeddings" in data:
        X = data["embeddings"].cpu().numpy().astype(np.float32)

    else:
        raise ValueError(
            f"No valid embedding keys found in {embedding_path}. "
            f"Expected 'embeddings' or ('embeddings_A', 'embeddings_B')."
        )

    return X



def validate_embedding_ids(embedding_path, ids):
    """Verify row identity; legacy files retain their documented sorted-ID convention."""
    sidecar = embedding_path + ".ids.json"
    if not os.path.exists(sidecar):
        warnings.warn(f"No ID sidecar for {embedding_path}; assuming sorted string-ID order", stacklevel=2)
        return
    with open(sidecar, encoding="utf-8") as handle:
        embedding_ids = [str(value) for value in json.load(handle)]
    if embedding_ids != list(map(str, ids)):
        raise ValueError(f"Embedding IDs do not match sorted CSV IDs: {embedding_path}")



def load_sampled_ids(ids_csv_path: str):
    df_ids = pd.read_csv(ids_csv_path, dtype={"id": str})
    if "id" not in df_ids.columns:
        raise ValueError(f"'id' column not found in {ids_csv_path}")
    return df_ids["id"].astype(str).tolist()



def load_full_dataset(
    embedding_path: str,
    split_path: str,
):
    X = load_X(embedding_path=embedding_path)
    pred_g, true_g, model_names, ids = load_pred_and_truth(split_path)

    validate_embedding_ids(embedding_path, ids)
    n = X.shape[0]
    if pred_g.shape[0] != n or len(true_g) != n or len(ids) != n:
        raise ValueError("Length mismatch among X, pred_g, true_g, ids")

    return X, pred_g, true_g, model_names, ids



def load_dataset_by_ids(
    embedding_path: str,
    split_path: str,
    sampled_ids_path: str,
):
    X, pred_g, true_g, model_names, ids_all = load_full_dataset(
        embedding_path=embedding_path,
        split_path=split_path,
    )

    sampled_ids = load_sampled_ids(sampled_ids_path)

    X_sub, pred_g_sub, true_g_sub, ids_sub = subset_by_sampled_ids(
        X, pred_g, true_g, ids_all, sampled_ids
    )

    return X_sub, pred_g_sub, true_g_sub, model_names, ids_sub



def subset_by_sampled_ids(X, pred_g, true_g, ids_all, sampled_ids):
    """
    X:         full array [N, D]
    pred_g:    full array [N, K]
    true_g:    full array [N]
    ids_all:   full ids array [N], already aligned with X
    sampled_ids: list[str], ids selected by MOE for one replication

    Returns subset in the SAME order as sampled_ids.
    """
    ids_all = np.asarray(ids_all).astype(str)
    sampled_ids = [str(x) for x in sampled_ids]

    if len(set(ids_all)) != len(ids_all) or len(set(sampled_ids)) != len(sampled_ids):
        raise ValueError("Full and sampled IDs must be unique")
    id_to_pos = {id_: i for i, id_ in enumerate(ids_all)}

    missing = [id_ for id_ in sampled_ids if id_ not in id_to_pos]
    if missing:
        raise ValueError(f"Some sampled ids are missing from full dataset, e.g. {missing[:5]}")

    idx = np.array([id_to_pos[id_] for id_ in sampled_ids], dtype=int)

    X_sub = X[idx]
    pred_g_sub = pred_g[idx]
    true_g_sub = true_g[idx]
    ids_sub = ids_all[idx]

    return X_sub, pred_g_sub, true_g_sub, ids_sub



def apply_pca_if_needed(
    E_train: np.ndarray,
    E_test: np.ndarray,
    pca_dim: int,
    pca_mode: str
):
    """
    Apply PCA according to pca_mode:
      - none : do not apply PCA
      - train: fit PCA on train only, transform both train/test
      - pool : fit PCA on pooled train+test, transform both train/test
    """
    if pca_mode not in {"none", "train", "pool"}:
        raise ValueError(f"Unsupported pca_mode={pca_mode}. Choose from none/train/pool.")

    if pca_mode == "none" or pca_dim <= 0:
        print("PCA disabled. Using original embeddings for OT.")
        info = {
            "pca_enabled": False,
            "pca_mode": "none",
            "pca_dim": 0,
            "pca_explained_variance_ratio_sum": None
        }
        return E_train, E_test, None, info

    original_dim = E_train.shape[1]

    if pca_mode == "train":
        max_valid_dim = min(E_train.shape[0], original_dim)
        if pca_dim > max_valid_dim:
            raise ValueError(
                f"Invalid pca_dim={pca_dim}. For pca_mode='train', must be <= "
                f"min(n_train, orig_dim) = {max_valid_dim}."
            )

        print(f"Fitting PCA on TRAIN only: original_dim={original_dim}, pca_dim={pca_dim}")
        pca = PCA(n_components=pca_dim)
        E_train_pca = pca.fit_transform(E_train)
        E_test_pca = pca.transform(E_test)

    else:  # pca_mode == "pool"
        pooled = np.vstack([E_train, E_test])
        max_valid_dim = min(pooled.shape[0], original_dim)
        if pca_dim > max_valid_dim:
            raise ValueError(
                f"Invalid pca_dim={pca_dim}. For pca_mode='pool', must be <= "
                f"min(n_train + n_test, orig_dim) = {max_valid_dim}."
            )

        print(f"Fitting PCA on POOLED train+test: original_dim={original_dim}, pca_dim={pca_dim}")
        pca = PCA(n_components=pca_dim)
        pca.fit(pooled)
        E_train_pca = pca.transform(E_train)
        E_test_pca = pca.transform(E_test)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"PCA explained variance ratio sum: {explained:.6f}")
    print("Train PCA shape:", E_train_pca.shape)
    print("Test PCA shape:", E_test_pca.shape)

    info = {
        "pca_enabled": True,
        "pca_mode": pca_mode,
        "pca_dim": int(pca_dim),
        "pca_explained_variance_ratio_sum": explained
    }

    return E_train_pca, E_test_pca, pca, info
