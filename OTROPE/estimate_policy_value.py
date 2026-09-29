"""Command-line entry point for OTROPE and baseline policy-value evaluation."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

if __package__:
    from .data_io import load_dataset_by_ids, load_full_dataset, apply_pca_if_needed
    from .otrope_weights import optimize_w_strict_ot_gpu
    from .estimators import estimate_weights_by_classifier, estimate_policy_values
else:
    from data_io import load_dataset_by_ids, load_full_dataset, apply_pca_if_needed
    from otrope_weights import optimize_w_strict_ot_gpu
    from estimators import estimate_weights_by_classifier, estimate_policy_values


def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)


    # Embedding files
    parser.add_argument("--train_embedding_file", type=str, required=True)
    parser.add_argument("--test_embedding_file", type=str, required=True)

    # Prediction/ground-truth split files
    parser.add_argument("--train_pred_file", type=str, default="combine_train.csv")
    parser.add_argument("--test_pred_file", type=str, default="combine_test.csv")
    parser.add_argument("--sampled_ids_path", type=str, required=True)

    # Evaluation csvs
    parser.add_argument("--train_eval_csv", type=str, required=True)
    parser.add_argument("--test_eval_csv", type=str, required=True)
    parser.add_argument("--prediction_col", type=str, required=True)

    # PCA
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=0,
        help="Dimension after PCA for OT. Use 0 to disable PCA."
    )
    parser.add_argument(
        "--pca_mode",
        type=str,
        default="none",
        choices=["none", "train", "pool"],
        help="PCA mode: none / train / pool"
    )

    # OT hyperparameters
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--max_iter", type=int, default=200, help="outer optimization iterations")
    parser.add_argument("--sinkhorn_iters", type=int, default=200, help="inner Sinkhorn iterations")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate for logits")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)


    # Output
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--save_name", type=str, default="wasserstein_weights")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_embedding_path = os.path.join(args.train_path, args.train_embedding_file)
    test_embedding_path = os.path.join(args.test_path, args.test_embedding_file)

    train_pred_path = os.path.join(args.train_path, args.train_pred_file)
    test_pred_path = os.path.join(args.test_path, args.test_pred_file)

    print("Loading train/test embeddings...")
    print("Loading train/test prediction+label files...")

    E_d1_raw, _, train_label, _, train_ids = load_dataset_by_ids(
        embedding_path=train_embedding_path,
        split_path=train_pred_path,
        sampled_ids_path=args.sampled_ids_path,
    )

    E_d2_raw, _, test_label, _, test_ids = load_full_dataset(
        embedding_path=test_embedding_path,
        split_path=test_pred_path,
    )

    n_train_emb = E_d1_raw.shape[0]
    n_test_emb = E_d2_raw.shape[0]
    n_train_pred = train_label.shape[0]
    n_test_pred = test_label.shape[0]

    print("Train embedding size:", n_train_emb)
    print("Test embedding size:", n_test_emb)
    print("Train pred/label size:", n_train_pred)
    print("Test pred/label size:", n_test_pred)

    if n_train_emb != n_train_pred:
        raise ValueError(
            f"Train size mismatch: embeddings have {n_train_emb}, pred/label have {n_train_pred}"
        )
    if n_test_emb != n_test_pred:
        raise ValueError(
            f"Test size mismatch: embeddings have {n_test_emb}, pred/label have {n_test_pred}"
        )

    print("Raw D1 shape:", E_d1_raw.shape)
    print("Raw D2 shape:", E_d2_raw.shape)

    raw_dist_of_means = np.linalg.norm(
        np.mean(E_d1_raw, axis=0) - np.mean(E_d2_raw, axis=0)
    )
    print(f"Raw distance between dataset centers: {raw_dist_of_means:.6f}")

    # PCA for OT space
    E_d1, E_d2, _, pca_info = apply_pca_if_needed(
        E_d1_raw,
        E_d2_raw,
        args.pca_dim,
        args.pca_mode
    )

    dist_of_means = np.linalg.norm(np.mean(E_d1, axis=0) - np.mean(E_d2, axis=0))
    print(f"Distance between OT-space dataset centers: {dist_of_means:.6f}")

    print("Computing OT weights (GPU differentiable Sinkhorn)...")
    w_star = optimize_w_strict_ot_gpu(
        E_d1,
        E_d2,
        reg=args.reg,
        outer_iters=args.max_iter,
        sinkhorn_iters=args.sinkhorn_iters,
        lr=args.lr,
        device=args.device,
        dtype=torch.float32,
        verbose=True
    )

    print(f"Sum of weights: {np.sum(w_star):.6f}")

    print("Computing classifier-based importance weights (logistic regression)...")
    w_cls = estimate_weights_by_classifier(
        E_d1, E_d2,
        random_state=args.seed,
    )
    print(f"Classifier weights sum: {np.sum(w_cls):.6f}")

    # --------------------------------------------------
    # Load evaluation predictions
    # --------------------------------------------------
    train_eval = pd.read_csv(args.train_eval_csv, dtype={"id": str})
    test_eval = pd.read_csv(args.test_eval_csv, dtype={"id": str})

    for frame, expected in ((train_eval, train_ids), (test_eval, test_ids)):
        if "id" not in frame or frame["id"].tolist() != list(map(str, expected)):
            raise ValueError("Evaluation CSV IDs must match embedding/sample order; regenerate with MOE.py")

    required_cols = ["true", args.prediction_col]
    for c in required_cols:
        if c not in train_eval.columns:
            raise ValueError(f"Missing '{c}' in {args.train_eval_csv}")
        if c not in test_eval.columns:
            raise ValueError(f"Missing '{c}' in {args.test_eval_csv}")

    if len(train_eval) != n_train_emb:
        raise ValueError(
            f"Train eval csv length mismatch: expected {n_train_emb}, got {len(train_eval)}"
        )
    if len(test_eval) != n_test_emb:
        raise ValueError(
            f"Test eval csv length mismatch: expected {n_test_emb}, got {len(test_eval)}"
        )

    results = {
        "replication": args.replication,
        "sample_size": args.sample_size,
        "seed": args.seed,
        **estimate_policy_values(train_eval, test_eval, args.prediction_col, w_star, w_cls),
    }

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.save_name + ".csv")
    pd.DataFrame([results]).to_csv(csv_path, index=False)

    print("\n===== Baselines / Estimates =====")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    meta = {
        "train_path": args.train_path,
        "test_path": args.test_path,
        "train_embedding_file": args.train_embedding_file,
        "test_embedding_file": args.test_embedding_file,
        "train_pred_file": args.train_pred_file,
        "test_pred_file": args.test_pred_file,
        "train_n": int(n_train_emb),
        "test_n": int(n_test_emb),
        "raw_dim": int(E_d1_raw.shape[1]),
        "ot_dim": int(E_d1.shape[1]),
        "pca_enabled": pca_info["pca_enabled"],
        "pca_mode": pca_info["pca_mode"],
        "pca_dim": pca_info["pca_dim"],
        "pca_explained_variance_ratio_sum": pca_info["pca_explained_variance_ratio_sum"],
        "reg": args.reg,
        "outer_iter": args.max_iter,
        "sinkhorn_iters": args.sinkhorn_iters,
        "lr": args.lr,
        "device": args.device,
    }

    meta_path = os.path.join(args.output_dir, args.save_name + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta to:", meta_path)

    print("Done.")



if __name__ == "__main__":
    main()
