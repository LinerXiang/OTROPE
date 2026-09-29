import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Shared colors and labels for MoE and majority-vote plots.
_ROLE_COLORS = {
    "direct":    "#1f77b4",
    "is":        "#ff7f0e",
    "dr":        "#9467bd",
    "PPI":       "#2ca02c",
    "tuned PPI": "#bcbd22",
    "ot_raw":    "#17becf",
    "ot":        "#d62728",
}

_ROLE_LABELS = {
    "direct":    "DM",
    "is":        "IS",
    "dr":        "DR",
    "PPI":       "PPI",
    "tuned PPI": "PPI++",
    "ot_raw":    "OT",
    "ot":        "OTROPE",
}

_MOE_METHOD_ROLES = {
    "dm":      "direct",
    "is":      "is",
    "dr":      "dr",
    "ppi":     "PPI",
    "ppi++":   "tuned PPI",
    "ot":      "ot_raw",
    "otrope":  "ot",
}

_MAJORITY_METHOD_ROLES = {
    "dm_majority":      "direct",
    "is":               "is",
    "dr_majority":      "dr",
    "ppi_majority":     "PPI",
    "ppi++_majority":   "tuned PPI",
    "ot":               "ot_raw",
    "otrope_majority":  "ot",
}

_MOE_CSV_METHODS      = ("dm", "dr", "ppi", "ppi++", "otrope")
_MAJORITY_CSV_METHODS = ("dm_majority", "dr_majority", "ppi_majority", "ppi++_majority", "otrope_majority")
_SHARED_METHODS       = ("ot", "is")
_ALL_METHODS          = _MOE_CSV_METHODS + _MAJORITY_CSV_METHODS + _SHARED_METHODS


def load_replication_csvs(results_dir: str, pattern: str = "ot_*.csv") -> pd.DataFrame:
    """Read all replication-level CSVs from a directory and concatenate them."""
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if len(files) == 0:
        raise ValueError(f"No CSV files found in {results_dir} with pattern={pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "sample_size" in df.columns:
        df["sample_size"] = df["sample_size"].astype(int)
    return df


def merge_replications_by_sample_size(
    df: pd.DataFrame,
    merged_dir: str,
):
    """Merge replication-level CSVs and save one merged CSV per sample size."""
    os.makedirs(merged_dir, exist_ok=True)
    for sample_size, g in df.groupby("sample_size"):
        out_path = os.path.join(merged_dir, f"merged_{sample_size}.csv")
        g.reset_index(drop=True).to_csv(out_path, index=False)
        print(f"[INFO] Saved merged file for sample_size={sample_size}: {out_path}")


def summarize_by_sample_size(
    df: pd.DataFrame,
    methods=("ppi", "ppi_majority", "otrope"),
    ci_level: float = 0.95,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """
    Summarize estimates by sample size and compute Monte Carlo confidence intervals.

    CI formula: mean ± z * sd / sqrt(n_rep)
    """
    if "sample_size" not in df.columns:
        raise ValueError("Input dataframe must contain column 'sample_size'")
    if "true" not in df.columns:
        raise ValueError("Input dataframe must contain column 'true'")

    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    if ci_level not in z_map:
        raise ValueError("Currently supported ci_level values: 0.90, 0.95, 0.99")
    z = z_map[ci_level]

    summary_rows = []

    for sample_size, g in df.groupby("sample_size"):
        row = {
            "sample_size": int(sample_size),
            "n_rep":       len(g),
            "true_value":  g["true"].mean(),
        }

        for method in methods:
            if method not in g.columns:
                if skip_missing:
                    continue
                raise ValueError(f"Missing column: {method}")

            vals = g[method].dropna()
            true_vals = g["true"].loc[vals.index]
            n = len(vals)
            mean_val = vals.mean()
            sd_val   = vals.std(ddof=1) if n > 1 else 0.0
            se_val   = sd_val / np.sqrt(n) if n > 0 else np.nan
            ci_lower = mean_val - z * se_val if n > 0 else np.nan
            ci_upper = mean_val + z * se_val if n > 0 else np.nan
            ae_vals  = (vals - true_vals).abs()
            ae_mean  = ae_vals.mean() if n > 0 else np.nan
            ae_sd    = ae_vals.std(ddof=1) if n > 1 else 0.0
            ae_se    = ae_sd / np.sqrt(n) if n > 0 else np.nan

            row[f"{method}_mean"]     = mean_val
            row[f"{method}_sd"]       = sd_val
            row[f"{method}_se"]       = se_val
            row[f"{method}_ci_lower"] = ci_lower
            row[f"{method}_ci_upper"] = ci_upper
            row[f"{method}_ae_mean"]  = ae_mean
            row[f"{method}_ae_sd"]    = ae_sd
            row[f"{method}_ae_se"]    = ae_se

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df.sort_values("sample_size").reset_index(drop=True)


def plot_dual_panel_by_sample_size(
    summary_df: pd.DataFrame,
    moe_title: str = "MoE",
    majority_title: str = "Majority Voting",
    ylim: tuple = None,
    sample_size_from: int = None,
    output_path: str = None,
    show: bool = True,
):
    """
    Two-panel chart: left = MOE methods, right = Majority Voting methods.
    Colors are consistent across panels; a single legend sits at the bottom.

    sample_size_from : if set, only plot sample sizes >= this value.
    """
    if sample_size_from is not None:
        summary_df = summary_df[summary_df["sample_size"] >= sample_size_from].reset_index(drop=True)

    fig, (ax_moe, ax_maj) = plt.subplots(1, 2, figsize=(15, 5))

    x_labels = summary_df["sample_size"].astype(int).tolist()
    x = np.arange(len(x_labels))

    legend_handles = []
    legend_labels  = []

    for ax, title, method_roles in (
        (ax_moe, moe_title,      _MOE_METHOD_ROLES),
        (ax_maj, majority_title, _MAJORITY_METHOD_ROLES),
    ):
        ax.set_title(title, fontsize=17)

        for method, role in method_roles.items():
            mean_col = f"{method}_mean"
            low_col  = f"{method}_ci_lower"
            high_col = f"{method}_ci_upper"

            if mean_col not in summary_df.columns:
                continue

            y      = summary_df[mean_col].to_numpy()
            y_low  = summary_df[low_col].to_numpy()
            y_high = summary_df[high_col].to_numpy()

            color  = _ROLE_COLORS[role]

            ax.fill_between(x, y_low, y_high, color=color, alpha=0.18, linewidth=0, zorder=1)
            (line,) = ax.plot(
                x, y, linewidth=2.0, color=color, zorder=2
            )

            if ax is ax_moe:
                legend_handles.append(line)
                legend_labels.append(_ROLE_LABELS[role])

        if "true_value" in summary_df.columns:
            (true_line,) = ax.plot(
                x,
                summary_df["true_value"].to_numpy(),
                linestyle="--",
                linewidth=2.0,
                color="black",
                alpha=0.85,
                zorder=3,
            )
            if ax is ax_moe:
                legend_handles.append(true_line)
                legend_labels.append("true value")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("n", fontsize=15)
        ax.set_ylabel("Estimated Value", fontsize=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    # Align y-axes: union of both panels' auto ranges; override with ylim=(ymin, ymax)
    if ylim is not None:
        y_range = ylim
    else:
        lo = min(ax_moe.get_ylim()[0], ax_maj.get_ylim()[0])
        hi = max(ax_moe.get_ylim()[1], ax_maj.get_ylim()[1])
        y_range = (lo, hi)
    ax_moe.set_ylim(y_range)
    ax_maj.set_ylim(y_range)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=True,
        fontsize=15,
        bbox_to_anchor=(0.5, -0.08),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=400, bbox_inches="tight")
        print(f"[INFO] Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dual_panel_mse(
    summary_df: pd.DataFrame,
    moe_title: str = "MoE",
    majority_title: str = "Majority Voting",
    sample_size_from: int = None,
    output_path: str = None,
    show: bool = True,
):
    """
    Two-panel MSE chart: left = MOE methods, right = Majority Voting methods.
    MSE = bias² + variance = (mean - true_value)² + sd²
    """
    if sample_size_from is not None:
        summary_df = summary_df[summary_df["sample_size"] >= sample_size_from].reset_index(drop=True)

    fig, (ax_moe, ax_maj) = plt.subplots(1, 2, figsize=(15, 5))

    x_labels = summary_df["sample_size"].astype(int).tolist()
    x = np.arange(len(x_labels))
    true_val = summary_df["true_value"].to_numpy()

    legend_handles = []
    legend_labels  = []

    for ax, title, method_roles in (
        (ax_moe, moe_title,      _MOE_METHOD_ROLES),
        (ax_maj, majority_title, _MAJORITY_METHOD_ROLES),
    ):
        ax.set_title(title, fontsize=17)

        for method, role in method_roles.items():
            mean_col = f"{method}_mean"
            sd_col   = f"{method}_sd"
            se_col   = f"{method}_se"

            if mean_col not in summary_df.columns:
                continue

            bias = summary_df[mean_col].to_numpy() - true_val
            sd   = summary_df[sd_col].to_numpy() if sd_col in summary_df.columns else 0.0
            se   = summary_df[se_col].to_numpy() if se_col in summary_df.columns else 0.0
            mse  = bias ** 2 + sd ** 2
            half = 2 * np.abs(bias) * se  # delta-method band half-width

            color  = _ROLE_COLORS[role]

            ax.fill_between(x, mse - half, mse + half, color=color, alpha=0.18, linewidth=0, zorder=1)
            (line,) = ax.plot(
                x, mse, linewidth=2.0, color=color, zorder=2
            )

            if ax is ax_moe:
                legend_handles.append(line)
                legend_labels.append(_ROLE_LABELS[role])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("n", fontsize=15)
        ax.set_ylabel("MSE", fontsize=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    lo = min(ax_moe.get_ylim()[0], ax_maj.get_ylim()[0])
    hi = max(ax_moe.get_ylim()[1], ax_maj.get_ylim()[1])
    ax_moe.set_ylim(lo, hi)
    ax_maj.set_ylim(lo, hi)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=True,
        fontsize=15,
        bbox_to_anchor=(0.5, -0.08),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=400, bbox_inches="tight")
        print(f"[INFO] Saved MSE plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dual_panel_mae(
    summary_df: pd.DataFrame,
    moe_title: str = "MoE",
    majority_title: str = "Majority Voting",
    sample_size_from: int = None,
    output_path: str = None,
    output_csv_moe: str = None,
    output_csv_majority: str = None,
    show: bool = True,
):
    """
    Two-panel MAE chart: left = MOE methods, right = Majority Voting methods.
    MAE = mean of per-replication absolute errors
    """
    if sample_size_from is not None:
        summary_df = summary_df[summary_df["sample_size"] >= sample_size_from].reset_index(drop=True)

    fig, (ax_moe, ax_maj) = plt.subplots(1, 2, figsize=(15, 5))

    x_labels = summary_df["sample_size"].astype(int).tolist()
    x = np.arange(len(x_labels))

    legend_handles = []
    legend_labels  = []
    csv_rows_moe = [{"sample_size": s} for s in x_labels]
    csv_rows_maj = [{"sample_size": s} for s in x_labels]

    for ax, title, method_roles, csv_rows in (
        (ax_moe, moe_title,      _MOE_METHOD_ROLES,      csv_rows_moe),
        (ax_maj, majority_title, _MAJORITY_METHOD_ROLES, csv_rows_maj),
    ):
        ax.set_title(title, fontsize=17)

        for method, role in method_roles.items():
            ae_mean_col = f"{method}_ae_mean"
            ae_se_col   = f"{method}_ae_se"
            ae_sd_col   = f"{method}_ae_sd"

            if ae_mean_col not in summary_df.columns:
                continue

            mae = summary_df[ae_mean_col].to_numpy()
            sd  = summary_df[ae_sd_col].to_numpy() if ae_sd_col in summary_df.columns else np.zeros_like(mae)
            se  = summary_df[ae_se_col].to_numpy() if ae_se_col in summary_df.columns else np.zeros_like(mae)

            color = _ROLE_COLORS[role]
            label = _ROLE_LABELS[role]

            ax.fill_between(x, mae - se, mae + se, color=color, alpha=0.18, linewidth=0, zorder=1)
            (line,) = ax.plot(
                x, mae, linewidth=2.0, color=color, zorder=2
            )

            if ax is ax_moe:
                legend_handles.append(line)
                legend_labels.append(label)

            for i, row in enumerate(csv_rows):
                row[f"{label}_MAE"]   = float(mae[i])
                row[f"{label}_AE_SD"] = float(sd[i])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("n", fontsize=15)
        ax.set_ylabel("MAE", fontsize=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    lo = min(ax_moe.get_ylim()[0], ax_maj.get_ylim()[0])
    hi = max(ax_moe.get_ylim()[1], ax_maj.get_ylim()[1])
    ax_moe.set_ylim(lo, hi)
    ax_maj.set_ylim(lo, hi)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=True,
        fontsize=15,
        bbox_to_anchor=(0.5, -0.08),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=400, bbox_inches="tight")
        print(f"[INFO] Saved MAE plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    import csv as _csv
    for path, rows in ((output_csv_moe, csv_rows_moe), (output_csv_majority, csv_rows_maj)):
        if path is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO] Saved MAE summary CSV to {path}")


def summarize_and_plot(
    results_dir: str,
    csv_pattern: str = "ot_*.csv",
    merged_dir: str = None,
    moe_csv_path: str = None,
    majority_csv_path: str = None,
    plot_path: str = None,
    mse_plot_path: str = None,
    mae_plot_path: str = None,
    mae_csv_moe_path: str = None,
    mae_csv_majority_path: str = None,
    ci_level: float = 0.95,
    size_from: int = 0,
):
    """Summarize replication CSVs, export tables, and plot estimates, MSE, and MAE."""
    df_raw = load_replication_csvs(results_dir, csv_pattern)
    df_summary = summarize_by_sample_size(
        df=df_raw,
        methods=_ALL_METHODS,
        ci_level=ci_level,
        skip_missing=True,
    )
    if merged_dir is None:
        merged_dir = os.path.join(results_dir, "merged_by_sample_size")
    merge_replications_by_sample_size(df_raw, merged_dir)

    base_cols = ["sample_size", "n_rep", "true_value"]

    def _save_split(methods, path):
        cols = list(base_cols)
        for m in methods:
            for suf in ("_mean", "_sd", "_se", "_ci_lower", "_ci_upper", "_ae_mean", "_ae_sd", "_ae_se"):
                c = f"{m}{suf}"
                if c in df_summary.columns:
                    cols.append(c)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df_summary[cols].to_csv(path, index=False)
        print(f"[INFO] Saved summary CSV to {path}")

    if moe_csv_path is not None:
        _save_split(_MOE_CSV_METHODS + _SHARED_METHODS, moe_csv_path)

    if majority_csv_path is not None:
        _save_split(_MAJORITY_CSV_METHODS + _SHARED_METHODS, majority_csv_path)

    plot_dual_panel_by_sample_size(
        summary_df=df_summary,
        sample_size_from=size_from,
        output_path=plot_path,
        show=False,
    )

    plot_dual_panel_mse(
        summary_df=df_summary,
        sample_size_from=size_from,
        output_path=mse_plot_path,
        show=False,
    )

    plot_dual_panel_mae(
        summary_df=df_summary,
        sample_size_from=size_from,
        output_path=mae_plot_path,
        output_csv_moe=mae_csv_moe_path,
        output_csv_majority=mae_csv_majority_path,
        show=False,
    )

    return df_raw, df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip processing; read --moe-csv and --majority-csv and replot")
    parser.add_argument("--results-dir",   type=str, default=None, help="Replication CSV directory")
    parser.add_argument("--moe-csv",       type=str, default=None)
    parser.add_argument("--majority-csv",  type=str, default=None)
    parser.add_argument("--plot-path",     type=str, default=None)
    parser.add_argument("--mse-path",      type=str, default=None)
    parser.add_argument("--mae-path",      type=str, default=None)
    parser.add_argument("--size-from",     type=int, default=0)
    args = parser.parse_args()

    if args.plot_only:
        if args.moe_csv is None or args.majority_csv is None:
            raise ValueError("--plot-only requires both --moe-csv and --majority-csv")
        df_moe = pd.read_csv(args.moe_csv)
        df_maj = pd.read_csv(args.majority_csv)
        # Keep shared OT and IS columns only from the MoE table.
        shared_base = ["sample_size", "n_rep", "true_value"]
        shared_cols = {f"{m}{s}" for m in _SHARED_METHODS
                       for s in ("_mean", "_sd", "_se", "_ci_lower", "_ci_upper", "_ae_mean", "_ae_sd", "_ae_se")}
        maj_extra = [c for c in df_maj.columns
                     if c not in shared_base and c not in shared_cols]
        df_summary = df_moe.merge(df_maj[["sample_size"] + maj_extra], on="sample_size", how="outer")
        df_summary["sample_size"] = df_summary["sample_size"].astype(int)
        df_summary = df_summary.sort_values("sample_size").reset_index(drop=True)
        print(f"[INFO] Loaded {len(df_summary)} rows from {args.moe_csv} + {args.majority_csv}")

        plot_dual_panel_by_sample_size(
            summary_df=df_summary,
            sample_size_from=args.size_from,
            output_path=args.plot_path,
            show=args.plot_path is None,
        )
        plot_dual_panel_mse(
            summary_df=df_summary,
            sample_size_from=args.size_from,
            output_path=args.mse_path,
            show=args.mse_path is None,
        )
        plot_dual_panel_mae(
            summary_df=df_summary,
            sample_size_from=args.size_from,
            output_path=args.mae_path,
            show=args.mae_path is None,
        )
    else:
        if args.results_dir is None:
            raise ValueError("Provide --results-dir (replication CSV folder)")
        base        = args.results_dir.rstrip("/")
        summary_dir = os.path.join(os.path.dirname(base), "summary")
        merged_dir  = os.path.join(summary_dir, "merged_by_sample_size")
        moe_csv      = args.moe_csv      or os.path.join(summary_dir, "summary_moe.csv")
        majority_csv = args.majority_csv or os.path.join(summary_dir, "summary_majority.csv")
        plot_path    = args.plot_path    or os.path.join(summary_dir, "estimate_by_sample_size.png")
        mse_path     = args.mse_path     or os.path.join(summary_dir, "mse_by_sample_size.png")
        mae_path     = args.mae_path     or os.path.join(summary_dir, "mae_by_sample_size.png")
        summarize_and_plot(
            results_dir=base,
            merged_dir=merged_dir,
            moe_csv_path=moe_csv,
            majority_csv_path=majority_csv,
            plot_path=plot_path,
            mse_plot_path=mse_path,
            mae_plot_path=mae_path,
            mae_csv_moe_path=os.path.join(summary_dir, "mae_table_moe.csv"),
            mae_csv_majority_path=os.path.join(summary_dir, "mae_table_majority.csv"),
            size_from=args.size_from,
        )
