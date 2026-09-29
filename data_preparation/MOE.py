import os
import itertools
import json
from argparse import ArgumentParser
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from OTROPE.data_io import load_X, load_pred_and_truth, validate_embedding_ids


# =========================
# Logging
# =========================

def make_logger(log_path: str, log_flag: int):
    """
    log_flag behavior:
      0 -> print only
      1 -> print + write file
      2 -> write file only
    """
    if log_flag == 0:
        def log(msg: str):
            print(msg)
        logfile = None
        return log, logfile

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logfile = open(log_path, "w", encoding="utf-8")

    if log_flag == 1:
        def log(msg: str):
            print(msg)
            logfile.write(msg + "\n")
            logfile.flush()
    elif log_flag == 2:
        def log(msg: str):
            logfile.write(msg + "\n")
            logfile.flush()
    else:
        raise ValueError("log_flag must be 0, 1, or 2")

    return log, logfile


# =========================
# Model
# =========================

class RouterNet(nn.Module):
    """
    Router policy network:
    input: X [B, D]
    output: w [B, num_experts] after softmax
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: Optional[List[int]] = None,
        temperature: float = 1.0,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.num_experts = int(num_experts)

        if hidden_dim is None or len(hidden_dim) == 0:
            self.net = nn.Sequential(nn.Linear(input_dim, num_experts))
        else:
            layers = []
            prev_dim = input_dim
            for h in hidden_dim:
                layers.append(nn.Linear(prev_dim, int(h)))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropout_p)))
                prev_dim = int(h)
            layers.append(nn.Linear(prev_dim, num_experts))
            self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.net(X) / self.temperature
        w = torch.softmax(logits, dim=-1)
        return w


# =========================
# Data utilities
# =========================

def train_val_split(
    X: np.ndarray,
    pred_g: np.ndarray,
    true_g: np.ndarray,
    ratio: float = 0.8,
    seed: int = 42
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Split arrays into train and validation sets.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.permutation(N)
    n_train = int(N * ratio)
    if not 0 < ratio < 1 or not 0 < n_train < N:
        raise ValueError("split_ratio must produce nonempty training and validation sets")

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    return (
        (X[train_idx], pred_g[train_idx], true_g[train_idx]),
        (X[val_idx], pred_g[val_idx], true_g[val_idx])
    )





def load_dataset(
    embedding_path: str,
    split_path: str,
    sampled: bool,
    sample_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], np.ndarray]:
    """
    Load one dataset.

    IMPORTANT:
    - embeddings are already sorted by id
    - CSV is sorted by id in load_pred_and_truth()
    - after that, X and CSV rows are aligned
    - sampling must be done jointly using the same indices

    Returns:
      X:         [N, D] or [sample_size, D]
      pred_g:    [N, num_experts] or [sample_size, num_experts]
      true_g:    [N] or [sample_size]
      model_names
      ids
    """
    X = load_X(embedding_path=embedding_path)
    pred_g, true_g, model_names, ids = load_pred_and_truth(split_path)

    validate_embedding_ids(embedding_path, ids)
    n = X.shape[0]

    if pred_g.shape[0] != n:
        raise ValueError(
            f"X and pred length mismatch: X={n}, pred={pred_g.shape[0]}. "
            f"embedding_path={embedding_path}, split_path={split_path}"
        )
    if len(true_g) != n:
        raise ValueError(
            f"X and truth length mismatch: X={n}, truth={len(true_g)}. "
            f"embedding_path={embedding_path}, split_path={split_path}"
        )
    if len(ids) != n:
        raise ValueError(
            f"X and ids length mismatch: X={n}, ids={len(ids)}. "
            f"embedding_path={embedding_path}, split_path={split_path}"
        )

    if sampled:
        if sample_size is None:
            raise ValueError("sample_size cannot be None when sampled=True")
        if sample_size > n:
            raise ValueError(f"sample_size={sample_size} exceeds dataset size n={n}")

        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        print(idx[0:5])
        X = X[idx]
        pred_g = pred_g[idx]
        true_g = true_g[idx]
        ids = ids[idx]

    print("[DEBUG] Final X shape:", X.shape)
    print("[DEBUG] Final pred_g shape:", pred_g.shape)
    print("[DEBUG] Final true_g shape:", true_g.shape)
    print("[DEBUG] Final ids preview:", ids[: min(5, len(ids))])

    return X, pred_g, true_g, model_names, ids



# =========================
# Preference utilities
# =========================

def prob_to_binary_label(
    p: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Convert probability to binary label in {0, 1}.
    """
    return (p >= threshold).astype(np.float32)


def summarize_preference_predictions(
    true_g: np.ndarray,
    p_pred: np.ndarray,
    g_pred: np.ndarray,
    prefix: str,
    log,
):
    """
    Log evaluation metrics suitable for human preference targets in {0, 0.5, 1}.
    Main metric: Brier score (same as MSE for probability targets).
    """
    true_g = np.asarray(true_g, dtype=np.float32)
    p_pred = np.asarray(p_pred, dtype=np.float32)

    avg_acc = float((g_pred == true_g).mean())

    log(f"{prefix}: sample size            = {len(true_g)}")
    log(f"{prefix}: true preference mean   = {true_g.mean():.4f}")
    log(f"{prefix}: predicted value mean    = {p_pred.mean():.4f}")
    log(f"{prefix}: predicted accuracy     = {avg_acc:.4f}")

    for val in [0.0, 0.5, 1.0]:
        mask = np.isclose(true_g, val)
        n = int(mask.sum())
        if n == 0:
            continue
        group_pred_mean = float(p_pred[mask].mean())
        group_brier = float(((p_pred[mask] - true_g[mask]) ** 2).mean())
        log(
            f"{prefix}: label={val:.1f}, n={n}, "
            f"pred_mean={group_pred_mean:.4f}, brier={group_brier:.6f}"
        )

@torch.no_grad()
def evaluate_router_objective(
    router: RouterNet,
    X: np.ndarray,
    pred_g: np.ndarray,
    true_g: np.ndarray,
    *,
    loss_type: str,
    batch_size: int = 2048,
    alpha: float = 0.0,
    device: Optional[str] = None,
) -> float:
    """
    Evaluate the unpenalized prediction loss on a validation set.
    Returns the average validation loss.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    router.eval()

    X_t = torch.as_tensor(X, dtype=torch.float32)
    pred_t = torch.as_tensor(pred_g, dtype=torch.float32)
    true_t = torch.as_tensor(true_g, dtype=torch.float32)

    ds = TensorDataset(X_t, pred_t, true_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    bce = nn.BCELoss(reduction="sum")
    mse = nn.MSELoss(reduction="sum")

    total_main_loss = 0.0
    total_entropy = 0.0
    total_n = 0

    for Xb, Eb, yb in loader:
        Xb = Xb.to(device)
        Eb = Eb.to(device)
        yb = yb.to(device)

        w = router(Xb)
        p = (w * Eb).sum(dim=1)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        if loss_type == "top-1":
            max_p, _ = (w * Eb).max(dim=1)
            max_p = torch.clamp(max_p, 1e-6, 1 - 1e-6)
            main_loss = -(torch.log(max_p)).sum()

        elif loss_type == "bce":
            main_loss = bce(p, yb)

        elif loss_type == "mse":
            main_loss = mse(p, yb)

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        entropy = -(w * torch.log(w + 1e-9)).sum(dim=1).sum()

        total_main_loss += float(main_loss.item())
        total_entropy += float(entropy.item())
        total_n += Xb.size(0)

    avg_main_loss = total_main_loss / max(total_n, 1)
    avg_entropy = total_entropy / max(total_n, 1)
    # Compare hyperparameters using the unpenalized validation loss.
    avg_total_loss = avg_main_loss
    return avg_total_loss

# =========================
# Training / Prediction
# =========================

def train_router(
    X_train: np.ndarray,
    pred_g_train: np.ndarray,
    true_g_train: np.ndarray,
    *,
    hidden_dim: Optional[List[int]],
    temperature: float,
    dropout_p: float,
    alpha: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    loss_type: str,
    log,
    device: Optional[str] = None,
    return_losses: bool = True,
) -> Tuple[RouterNet, List[float]]:
    """
    Train the router. No evaluation metrics are printed here.
    Only training loss per epoch is logged.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    pred_train_t = torch.as_tensor(pred_g_train, dtype=torch.float32, device=device)
    true_train_t = torch.as_tensor(true_g_train, dtype=torch.float32, device=device)

    D = X_train_t.shape[1]
    num_experts = pred_train_t.shape[1]

    router = RouterNet(
        input_dim=D,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        temperature=temperature,
        dropout_p=dropout_p,
    ).to(device)

    dataset = TensorDataset(X_train_t, pred_train_t, true_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    train_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        router.train()
        total_loss = 0.0
        n_steps = 0

        for Xb, Eb, yb in loader:
            optimizer.zero_grad()
            w = router(Xb)
            p = (w * Eb).sum(dim=1)
            p = torch.clamp(p, 1e-6, 1 - 1e-6)

            if loss_type == "top-1":
                max_p, _ = (w * Eb).max(dim=1)
                max_p = torch.clamp(max_p, 1e-6, 1 - 1e-6)
                main_loss = -(torch.log(max_p)).mean()

            elif loss_type == "bce":
                main_loss = bce(p, yb)

            elif loss_type == "mse":
                main_loss = mse(p, yb)

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            entropy = -(w * torch.log(w + 1e-9)).sum(dim=1).mean()
            loss = main_loss + alpha * entropy

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        train_losses.append(avg_loss)
        log(f"[Epoch {epoch}] Train Loss = {avg_loss:.6f}")

    return router, train_losses if return_losses else []


@torch.no_grad()
def router_predict(
    router: RouterNet,
    X: np.ndarray,
    pred_g: np.ndarray,
    *,
    batch_size: int = 1024,
    low: float = 1.0 / 3.0,
    high: float = 2.0 / 3.0,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict with router:
      - p_weighted: weighted sum probability
      - y_weighted: binary preference label in {0, 1}
      - p_top1: top-1 expert probability
      - y_top1: binary preference label in {0, 1}

    Note:
      Main evaluation should use p_weighted / p_top1 directly.
      The discretized labels are auxiliary outputs only.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    router.eval()

    X_t = torch.as_tensor(X, dtype=torch.float32)
    pred_t = torch.as_tensor(pred_g, dtype=torch.float32)

    ds = TensorDataset(X_t, pred_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    p1_all, p2_all = [], []

    for Xb, Eb in loader:
        Xb = Xb.to(device)
        Eb = Eb.to(device)
        w = router(Xb)

        p1 = (w * Eb).sum(dim=1)
        p1_all.append(p1.detach().cpu())

        idx = w.argmax(dim=1)
        batch_idx = torch.arange(Eb.size(0), device=device)
        p2 = Eb[batch_idx, idx]
        p2_all.append(p2.detach().cpu())

    p1 = torch.cat(p1_all, dim=0).numpy()
    p2 = torch.cat(p2_all, dim=0).numpy()

    y1 = prob_to_binary_label(p1, threshold= 0.5)
    y2 = prob_to_binary_label(p2, threshold= 0.5)

    return p1, y1, p2, y2


# =========================
# Saving artifacts
# =========================

def save_full_results_csv(
    save_path: str,
    true_g,
    pred_g,
    model_names,
    p1, y1, p2, y2,
    extra_meta: Optional[Dict[str, Any]] = None,
    ids=None,
):
    """
    Save all results into a CSV for preview,
    and save metadata into a side JSON file.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    n = len(true_g)
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "true": true_g,
    })

    if model_names is None:
        model_names = [f"expert_{i}" for i in range(pred_g.shape[1])]

    for i, name in enumerate(model_names):
        df[f"expert_{name}"] = pred_g[:, i]

    df["p_weighted"] = p1
    df["y_weighted"] = y1
    df["p_top1"] = p2
    df["y_top1"] = y2

    if ids is not None:
        df.insert(0, "id", ids)
    df.to_csv(save_path, index=False)
    if extra_meta is not None:
        with open(save_path + ".meta.json", "w", encoding="utf-8") as handle:
            json.dump(extra_meta, handle, indent=2)


def plot_training_loss(train_losses: List[float], save_path: str, title: str):
    """
    Plot and save the training loss curve as a PNG.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# Grid Search
# =========================

def grid_search_best_config(
    X: np.ndarray,
    pred_g: np.ndarray,
    true_g: np.ndarray,
    *,
    alpha_list: List[float],
    temperature_list: List[float],
    hidden_dim: Optional[List[int]],
    dropout_p: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    split_ratio: float,
    seed: int,
    loss_type: str,
    log,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Automatic grid search over alpha and temperature.
    Best config is selected by the unpenalized validation loss.
    No per-epoch validation metrics are printed.
    """
    (X_tr, pred_tr, true_tr), (X_val, pred_val, true_val) = train_val_split(
        X, pred_g, true_g, ratio=split_ratio, seed=seed
    )

    best_score = float("inf")
    best_config: Optional[Dict[str, Any]] = None

    total = len(alpha_list) * len(temperature_list)
    cnt = 0

    for alpha, temperature in itertools.product(alpha_list, temperature_list):
        cnt += 1
        log(f"[Grid {cnt}/{total}] alpha={alpha}, temperature={temperature}")

        router, _ = train_router(
            X_train=X_tr,
            pred_g_train=pred_tr,
            true_g_train=true_tr,
            hidden_dim=hidden_dim,
            temperature=float(temperature),
            dropout_p=float(dropout_p),
            alpha=float(alpha),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            epochs=int(epochs),
            loss_type=loss_type,
            log=lambda msg: None,
            device=device,
            return_losses=False,
        )

        score = evaluate_router_objective(
            router=router,
            X=X_val,
            pred_g=pred_val,
            true_g=true_val,
            loss_type=loss_type,
            batch_size=2048,
            alpha=float(alpha),
            device=device,
        )
        log(f"[Grid {cnt}/{total}] val_objective={score:.6f}")

        if score < best_score:
            best_score = score
            best_config = {
                "alpha": float(alpha),
                "temperature": float(temperature),
                "hidden_dim": hidden_dim,
                "dropout_p": float(dropout_p),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "loss_type": loss_type,
                "best_val_objective": float(best_score),
            }
            log(f"[Grid {cnt}/{total}] New best config found.")

    if best_config is None:
        raise RuntimeError("Grid search failed to produce any configuration.")

    return best_config


# =========================
# Main
# =========================

def main():
    parser = ArgumentParser()
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--embedding_model", "--embded_model", dest="embded_model", type=str, required=True)
    parser.add_argument("--train_split_path", type=str, required=True)
    parser.add_argument("--test_split_path", type=str, default="")

    parser.add_argument("--train_embedding_path", type=str, required=True)
    parser.add_argument("--test_embedding_path", type=str, default="")

    parser.add_argument("--alpha_list", type=float, nargs="+", default=[0.0])
    parser.add_argument("--temperature_list", type=float, nargs="+", default=[1.0])

    parser.add_argument("--hidden_dim", type=int, nargs="*", default=[1024, 512])
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_times", type=int, default=10)

    parser.add_argument("--replication", type=int, default=1)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["top-1", "bce", "mse"]
    )

    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_path", type=str, default="train_router_MOE.txt")
    parser.add_argument("--log_flag", type=int, default=1)

    args = parser.parse_args()
    if args.epoch_times < 1 or args.batch_size < 1 or args.sample_size < 2:
        parser.error("Require epoch_times >= 1, batch_size >= 1, sample_size >= 2")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log_path = os.path.join(args.output_dir, args.log_path)
    log, logfile = make_logger(log_path, args.log_flag)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(args.output_dir, exist_ok=True)

        log("=" * 80)
        log(f"Train split:       {args.train_split_path}")
        log(f"Train embedding:   {args.train_embedding_path}")
        if args.test_split_path:
            log(f"Test split:        {args.test_split_path}")
        if args.test_embedding_path:
            log(f"Test embedding:    {args.test_embedding_path}")
        log(f"Device:            {device}")
        log("=" * 80)

        log(f"Loss type:         {args.loss_type}")

        # 1) Load training dataset
        X_trainset, pred_trainset, true_trainset, model_names, ids = load_dataset(
            embedding_path=args.train_embedding_path,
            split_path=args.train_split_path,
            sampled=True,
            sample_size = args.sample_size,
            seed=args.seed
        )

        os.makedirs(args.output_dir, exist_ok=True)
        sampled_ids_path = f"{args.output_dir}/ids_{args.sample_size}_{args.replication}.csv"
        pd.DataFrame({"id": ids}).to_csv(sampled_ids_path, index=False)

        # 2) Automatic grid search
        log("Starting automatic grid search over alpha x temperature ...")
        best_config = grid_search_best_config(
            X_trainset,
            pred_trainset,
            true_trainset,
            alpha_list=args.alpha_list,
            temperature_list=args.temperature_list,
            hidden_dim=args.hidden_dim if len(args.hidden_dim) > 0 else None,
            dropout_p=args.dropout_p,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epoch_times,
            split_ratio=args.split_ratio,
            seed=args.seed,
            loss_type=args.loss_type,
            log=log,
            device=device,
        )
        log("Grid search completed.")
        log(f"Best config: {best_config}")

        # 3) Full training
        router_final, train_losses_full = train_router(
            X_train=X_trainset,
            pred_g_train=pred_trainset,
            true_g_train=true_trainset,
            hidden_dim=best_config["hidden_dim"],
            temperature=best_config["temperature"],
            dropout_p=best_config["dropout_p"],
            alpha=best_config["alpha"],
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
            batch_size=best_config["batch_size"],
            epochs=best_config["epochs"],
            loss_type=best_config["loss_type"],
            log=log,
            device=device,
            return_losses=True,
        )

        # 4) Save train predictions
        p1_tr, y1_tr, p2_tr, y2_tr = router_predict(
            router_final,
            X_trainset,
            pred_trainset,
            batch_size=2048,
            device=device,
        )

        train_result_csv_path = os.path.join(args.output_dir, f"train_MOE_{args.embded_model}_{args.sample_size}_{args.replication}.csv")

        summarize_preference_predictions(
            true_g=np.asarray(true_trainset, dtype=np.float32),
            p_pred=np.asarray(p1_tr, dtype=np.float32),
            g_pred=np.asarray(y1_tr, dtype=np.float32),
            prefix="Train",
            log=log,
        )

        save_full_results_csv(
            train_result_csv_path,
            true_g=true_trainset,
            ids=ids,
            pred_g=pred_trainset,
            model_names=model_names,
            p1=p1_tr,
            y1=y1_tr,
            p2=p2_tr,
            y2=y2_tr,
            extra_meta={
                "split": "train",
                "split_path": args.train_split_path,
                "embedding_path": args.train_embedding_path,
                **best_config,
            }
        )

        log("Saved FULL-train router predictions to:")
        log(f"  CSV         : {train_result_csv_path}")

        # 5) Save training loss plot
        loss_fig_path = os.path.join(args.output_dir, f"training_loss_{args.embded_model}_{args.sample_size}_{args.replication}.png")
        plot_training_loss(
            train_losses_full,
            loss_fig_path,
            title="Training Loss"
        )
        log(f"Saved training loss figure to: {loss_fig_path}")

        # 6) Run on test dataset
        if args.test_split_path and args.test_embedding_path:
            X_test, pred_test, true_test, test_model_names, test_ids = load_dataset(
                embedding_path=args.test_embedding_path,
                split_path=args.test_split_path,
                sampled=False,
                sample_size=1000000,
                seed=args.seed
            )

            if test_model_names != model_names:
                raise ValueError("Train and test expert columns must match in order")

            p1_te, y1_te, p2_te, y2_te = router_predict(
                router_final,
                X_test,
                pred_test,
                batch_size=2048,
                device=device,
            )

            test_result_csv_path = os.path.join(args.output_dir, f"test_MOE_{args.embded_model}_{args.sample_size}_{args.replication}.csv")

            summarize_preference_predictions(
                true_g=np.asarray(true_test, dtype=np.float32),
                p_pred=np.asarray(p1_te, dtype=np.float32),
                g_pred=np.asarray(y1_te, dtype=np.float32),
                prefix="Test",
                log=log,
            )

            save_full_results_csv(
                test_result_csv_path,
                true_g=true_test,
                ids=test_ids,
                pred_g=pred_test,
                model_names=model_names,
                p1=p1_te,
                y1=y1_te,
                p2=p2_te,
                y2=y2_te,
                extra_meta={
                    "split": "test",
                    "split_path": args.test_split_path,
                    "embedding_path": args.test_embedding_path,
                    **best_config,
                }
            )

            log("Saved test router predictions to:")
            log(f"  CSV         : {test_result_csv_path}")

        log("All done.")

    finally:
        if logfile is not None:
            logfile.close()


if __name__ == "__main__":
    main()
