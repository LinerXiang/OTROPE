import math
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0) Utilities
# ============================================================
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_SQRT2     = math.sqrt(2.0)
_SQRT2_INV = 1.0 / _SQRT2


def _phi(x):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + torch.erf(x * _SQRT2_INV))


def _phi_inv(u):
    """Standard normal quantile via erfinv."""
    return _SQRT2 * torch.erfinv(2.0 * u - 1.0)


def trunc_normal_sample(mu, sigma, lo, hi):
    """Sample from TruncNormal(mu, sigma, lo, hi) via inverse-CDF trick."""
    cdf_lo = _phi((lo - mu) / sigma)
    cdf_hi = _phi((hi - mu) / sigma)
    u = (torch.rand_like(mu) * (cdf_hi - cdf_lo) + cdf_lo).clamp(1e-6, 1 - 1e-6)
    return mu + sigma * _phi_inv(u)


# ============================================================
# 1) Global DGP constants  (change here to reconfigure)
# ============================================================
X_MU, X_SIGMA, X_LO, X_HI = 0.0, 1.0, -3.0, 3.0   # X ~ TruncNormal
Y_SIGMA, Y_LO,  Y_HI       = 0.5, -4.0, 4.0          # Y|X components ~ TruncNormal
R_W0, R_WX, R_WS           = 2.0, 3.0, 2.0          # reward linear coefficients
R_SIGMA                     = 1.0                    # reward noise std


# ============================================================
# 2) DGP helpers
# ============================================================
def make_gamma(d, device, val=0.5):
    """mu_ref_j(x) = val * x for all j."""
    return torch.full((d,), val, device=device)


def make_a(d, device, seed=123):
    """Unit vector defining the reward-relevant direction in Y-space."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    a = torch.randn(d, generator=g, device=device)
    return a / (a.norm() + 1e-12)


def sample_X(n, device):
    """X ~ TruncNormal(X_MU, X_SIGMA, X_LO, X_HI)"""
    mu    = torch.full((n,), X_MU,    device=device)
    sigma = torch.full((n,), X_SIGMA, device=device)
    return trunc_normal_sample(mu, sigma, X_LO, X_HI)


def sample_Y_ref(X, gamma):
    """Y_j | x ~ TruncNormal(gamma_j * x, Y_SIGMA, Y_LO, Y_HI)  for each j."""
    mu    = X.unsqueeze(1) * gamma.unsqueeze(0)          # (n, d)
    sigma = torch.full_like(mu, Y_SIGMA)
    return trunc_normal_sample(mu, sigma, Y_LO, Y_HI)


def sample_Y_tgt(X, gamma, a, delta):
    """Shift mu along reward-relevant direction a: mu_tgt = mu_ref + delta * a."""
    mu    = X.unsqueeze(1) * gamma.unsqueeze(0) + delta * a.unsqueeze(0)
    sigma = torch.full_like(mu, Y_SIGMA)
    return trunc_normal_sample(mu, sigma, Y_LO, Y_HI)


def sample_reward(X, Y, a):
    """R = R_W0 + R_WX*x + R_WS*(a^T y) + N(0, R_SIGMA^2)"""
    s  = (Y * a).sum(dim=1)
    mu = R_W0 + R_WX * X + R_WS * s
    return mu + R_SIGMA * torch.randn_like(mu)


# ============================================================
# 3) True density ratio  r(x,y) = p_tgt(y|x) / p_ref(y|x)
#    Each component j is independent TruncNormal; ratio is exact.
# ============================================================
@torch.no_grad()
def true_density_ratio(X, Y, gamma, a, delta, clip_max=None):
    s      = Y_SIGMA
    mu_ref = X.unsqueeze(1) * gamma.unsqueeze(0)     # (n, d)
    mu_tgt = mu_ref + delta * a.unsqueeze(0)          # (n, d)

    # Gaussian kernel ratio (log scale)
    log_gauss = (0.5 / s**2) * ((Y - mu_ref)**2 - (Y - mu_tgt)**2).sum(dim=1)

    # Normalizing-constant ratio: prod_j Z_ref_j / Z_tgt_j
    Z_ref = (_phi((Y_HI - mu_ref) / s) - _phi((Y_LO - mu_ref) / s)).clamp(1e-10)
    Z_tgt = (_phi((Y_HI - mu_tgt) / s) - _phi((Y_LO - mu_tgt) / s)).clamp(1e-10)
    log_norm = (torch.log(Z_ref) - torch.log(Z_tgt)).sum(dim=1)

    r = torch.exp(log_gauss + log_norm)
    if clip_max is not None:
        r = r.clamp(0.0, clip_max)
    return r


# ============================================================
# 4) Misspecified predictor — uses only x, ignores Y
#    True reward depends on both x and a^T y, so this is wrong by design.
# ============================================================
class MisspecPredictor(nn.Module):
    def __init__(self, hidden=(32,)):
        super().__init__()
        layers, d0 = [], 1
        for h in hidden:
            layers += [nn.Linear(d0, h), nn.ReLU()]
            d0 = h
        layers += [nn.Linear(d0, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)


def train_predictor(model, X, R, lr=2e-3, epochs=120, batch_size=512, weight_decay=1e-4):
    model.to(X.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n   = X.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=X.device)
        for start in range(0, n, batch_size):
            j    = idx[start:start + batch_size]
            loss = F.mse_loss(model(X[j]), R[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ============================================================
# 5) OT reweighting (logits + Adam + differentiable log-domain Sinkhorn)
#    Same algorithm as OTROPE/otrope_weights.py :: optimize_w_strict_ot_gpu
# ============================================================
def _sinkhorn_cost_from_logits(logits, b, C, reg, sinkhorn_iters=200, tol=1e-6):
    """Differentiable entropic OT cost. w = softmax(logits)."""
    w     = torch.softmax(logits, dim=0)
    log_a = torch.log(w + 1e-32)
    log_b = torch.log(b + 1e-32)
    M     = -C / reg

    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    for _ in range(sinkhorn_iters):
        f_prev = f
        f = log_a - torch.logsumexp(M + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(M + f.unsqueeze(1), dim=0)
        if torch.max(torch.abs(f - f_prev)).item() < tol:
            break

    P    = torch.exp(f.unsqueeze(1) + g.unsqueeze(0) + M)
    cost = torch.sum(P * C)
    return cost, w


def optimize_w_ot(E1, E2, reg=0.05, outer_iters=300, sinkhorn_iters=200,
                  lr=0.1, device="cuda", verbose=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    E1 = torch.as_tensor(E1, dtype=torch.float32, device=device)
    E2 = torch.as_tensor(E2, dtype=torch.float32, device=device)
    n, N = E1.shape[0], E2.shape[0]

    C      = torch.cdist(E1, E2, p=2)
    C      = C / (C.max() + 1e-12)
    b      = torch.full((N,), 1.0 / N, dtype=torch.float32, device=device)
    logits = torch.zeros(n, dtype=torch.float32, device=device, requires_grad=True)
    opt    = torch.optim.Adam([logits], lr=lr)

    best_cost, best_w = None, None
    for t in range(outer_iters):
        opt.zero_grad()
        cost, w = _sinkhorn_cost_from_logits(logits, b, C, reg, sinkhorn_iters)
        cost.backward()
        opt.step()

        with torch.no_grad():
            if best_cost is None or cost.item() < best_cost:
                best_cost = cost.item()
                best_w    = w.detach().clone()
            if verbose and (t % 20 == 0 or t == outer_iters - 1):
                ess = float((w.sum()**2 / (w**2).sum()).item())
                print(f"  iter {t:3d} | cost={cost.item():.6f} | ESS={ess:.1f}")

    return best_w


# ============================================================
# 6) Runner
# ============================================================
def run_sim(
    n=1000, N=2000,
    d=5,
    seed=0,
    device="cuda",

    delta=4.0,
    a_seed=123,

    ratio_clip_max=50.0,

    ot_reg=0.05, ot_outer_iters=300, ot_sinkhorn_iters=200, ot_lr=0.1,
    verbose_ot=False,
    v_true=None,
    return_samples=False,
):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    gamma = make_gamma(d, device)
    a     = make_a(d, device, seed=a_seed)

    # --- reference data ---
    X1 = sample_X(n, device)
    Y1 = sample_Y_ref(X1, gamma=gamma)
    R1 = sample_reward(X1, Y1, a)

    # --- target data ---
    X2 = sample_X(N, device)
    Y2 = sample_Y_tgt(X2, gamma=gamma, a=a, delta=delta)
    R2 = sample_reward(X2, Y2, a)

    # A) Misspecified predictor: fixed linear g(x) = R_W0 + R_WX * x, ignores Y
    with torch.no_grad():
        # ghat1 = R_W0 + (R_WX) * X1
        # ghat2 = R_W0 + (R_WX) * X2
        ghat1 = R_W0-2 + -3 * X1
        ghat2 = R_W0-2 + -3 * X2
    V_dm = float(ghat2.mean())

    # B) True density ratio
    r1         = true_density_ratio(X1, Y1, gamma=gamma, a=a, delta=delta, clip_max=ratio_clip_max)
    V_ipw_true = float((r1 * R1).mean())
    V_dr_true  = float(V_dm + (r1 * (R1 - ghat1)).mean())
    ess_r      = float(r1.sum()**2 / (r1**2).sum())
    r_mean     = float(r1.mean())
    r_min      = float(r1.min())
    r_max      = float(r1.max())

    # C) PPI (uniform correction)
    V_ppi = float(V_dm + (R1 - ghat1).mean())

    # D) PPI++ (tuned lambda)
    R1_np  = R1.cpu().numpy()
    g1_np  = ghat1.cpu().numpy()
    g2_np  = ghat2.cpu().numpy()
    n_tr, n_te = len(R1_np), len(g2_np)

    cov_rg      = np.cov(R1_np, g1_np, ddof=1)[0, 1]
    var_g       = np.var(np.concatenate([g1_np, g2_np]), ddof=1)
    lam         = cov_rg / ((1 + n_tr / n_te) * var_g + 1e-12)
    V_ppi_tuned = R1_np.mean() + lam * (g2_np.mean() - g1_np.mean())

    # E) OT reweighting
    s1     = (Y1 * a).sum(dim=1)
    s2     = (Y2 * a).sum(dim=1)
    E1     = torch.stack([X1, s1], dim=1)
    E2     = torch.stack([X2, s2], dim=1)
    w_star = optimize_w_ot(E1, E2, reg=ot_reg, outer_iters=ot_outer_iters,
                           sinkhorn_iters=ot_sinkhorn_iters, lr=ot_lr,
                           device=str(device), verbose=verbose_ot)

    V_ot    = float((w_star * R1).sum())
    V_ot_dr = float(V_dm + (w_star * (R1 - ghat1)).sum())
    ess_ot  = float(w_star.sum()**2 / (w_star**2).sum())

    result = {
        "n": int(n), "N": int(N), "d": int(d), "delta": float(delta),

        "V_true":      v_true,
        "V_naive":     float(R1.mean()),
        "V_dm":        V_dm,
        "V_ipw_true":  V_ipw_true,
        "V_dr_true":   V_dr_true,
        "V_ppi":       V_ppi,
        "V_ppi_tuned": float(V_ppi_tuned),
        "V_ot":        V_ot,
        "V_ot_dr":     V_ot_dr,

        "lambda":      float(lam),
        "ratio_mean":  r_mean,
        "ratio_min":   r_min,
        "ratio_max":   r_max,
        "ratio_ESS":   ess_r,
        "OT_ESS":      ess_ot,
        "mse_pred":    float(F.mse_loss(ghat1, R1)),
    }

    if return_samples:
        samples = {"X1": X1, "Y1": Y1, "R1": R1,
                   "X2": X2, "Y2": Y2, "R2": R2, "a": a}
        return result, samples
    return result


# ============================================================
# 7) Main: 200 replications × n sweep (fixed delta=3)
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="PyTorch device, e.g. cuda:0 / cuda:1 / cpu")
    parser.add_argument("--n_rep", type=int, default=200)
    args = parser.parse_args()
    if args.n_rep < 1:
        parser.error("n_rep must be positive")
    set_seed(2026)

    N_REP  = args.n_rep
    DELTA  = 1
    N_LIST = [200, 400, 600, 800, 1000]
    _device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # compute V_true once for the fixed delta
    with torch.no_grad():
        _gamma = make_gamma(5, _device)
        _a     = make_a(5, _device, seed=123)
        _X_mc  = sample_X(500_000, _device)
        _Y_mc  = sample_Y_tgt(_X_mc, gamma=_gamma, a=_a, delta=DELTA)
        _R_mc  = sample_reward(_X_mc, _Y_mc, _a)
        v_true = float(_R_mc.mean())
    del _X_mc, _Y_mc, _R_mc
    print(f"delta={DELTA}  V_true (MC 500k) = {v_true:.6f}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir  = os.path.join(base_dir, "simulation")
    os.makedirs(sim_dir, exist_ok=True)

    total = N_REP * len(N_LIST)
    done  = 0

    for n in N_LIST:
        n_results = []
        for rep in range(N_REP):
            out = run_sim(
                n=n, N=2000, d=5, seed=rep, device=args.device,
                delta=DELTA,
                ratio_clip_max=None,
                ot_reg=0.1, ot_outer_iters=500, ot_sinkhorn_iters=1000, ot_lr=0.05,
                v_true=v_true,
            )
            out["rep"] = rep
            n_results.append(out)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done:4d}/{total}] rep={rep}  n={n}")

        # save per-n results and free GPU memory
        n_csv      = os.path.join(sim_dir, f"results_{n}_{DELTA}.csv")
        fieldnames = ["rep"] + [k for k in n_results[0] if k != "rep"]
        with open(n_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(n_results)
        print(f"  Saved n={n} ({len(n_results)} rows) -> {n_csv}")
        del n_results
        torch.cuda.empty_cache()

    # reload all per-n CSVs for summary
    results = []
    for n in N_LIST:
        n_csv = os.path.join(sim_dir, f"results_{n}_{DELTA}.csv")
        with open(n_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({k: (int(v) if k in ("rep", "n", "N", "d") else float(v))
                                 for k, v in row.items()})

    # --- full results ---
    csv_path = os.path.join(base_dir, f"results_size_{DELTA}.csv")
    fieldnames = ["rep"] + [k for k in results[0] if k != "rep"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDone. {len(results)} rows saved to {csv_path}")

    # --- summary: mean absolute error and std of absolute error, grouped by n ---
    ESTIMATORS = ["V_dm", "V_ipw_true", "V_dr_true",
                  "V_ppi", "V_ppi_tuned", "V_ot", "V_ot_dr"]

    from collections import defaultdict
    buckets = defaultdict(list)   # n -> list of row dicts
    for row in results:
        buckets[row["n"]].append(row)

    summary_rows = []
    for n_val in sorted(buckets):
        rows  = buckets[n_val]
        v_arr = np.array([r["V_true"] for r in rows])
        entry = {"n": n_val, "n_rep": len(rows)}
        for est in ESTIMATORS:
            v_est = np.array([r[est] for r in rows])
            ae    = np.abs(v_est - v_arr)
            entry[f"{est}_MAE"] = float(np.mean(ae))
            entry[f"{est}_AE_std"] = float(np.std(ae, ddof=1))
        summary_rows.append(entry)

    sum_path = os.path.join(base_dir, f"summary_size_{DELTA}.csv")
    sum_fields = list(summary_rows[0].keys())
    with open(sum_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to {sum_path}")

    # print summary table
    print(f"\n{'n':>6}  {'n_rep':>5}  " +
          "  ".join(f"{e+'_MAE':>16}" for e in ESTIMATORS))
    for row in summary_rows:
        print(f"{row['n']:>6}  {row['n_rep']:>5}  " +
              "  ".join(f"{row[e+'_MAE']:>16.6f}" for e in ESTIMATORS))

    # --- MAE line plot ---
    import matplotlib.pyplot as plt

    PLOT_ESTIMATORS = ["V_ipw_true", "V_dr_true",
                       "V_ppi", "V_ppi_tuned", "V_ot_dr"]
    LABELS = {
        "V_ppi":       "PPI",
        "V_ppi_tuned": "PPI++",
        "V_ipw_true":  "IPW",
        "V_dr_true":   "DR",
        "V_ot_dr":     "OTROPE",
    }
    COLORS = ["#ff7f0e", "#9467bd",
              "#2ca02c", "#bcbd22", "#d62728"]

    n_vals = [row["n"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    for est, color in zip(PLOT_ESTIMATORS, COLORS):
        maes = [row[f"{est}_MAE"] for row in summary_rows]
        ax.plot(n_vals, maes, marker="o", markersize=4, color=color, lw=0.9, alpha=0.75, label=LABELS[est])
    ax.set_xlabel(r"$n$", fontsize=13)
    ax.set_ylabel("MAE", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(base_dir, "summary_mae_size.png")
    fig.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"MAE plot saved to {plot_path}")


# python simulation_size.py --device cuda:1 --n_rep 10
