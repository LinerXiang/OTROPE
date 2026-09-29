"""Semantic transport costs, differentiable Sinkhorn, and source-weight optimization."""
import torch


def build_cost_matrix_torch(E1, E2, device="cuda", dtype=torch.float32):
    """
    Build normalized pairwise Euclidean cost matrix on device.
    E1: [n, d]
    E2: [N, d]
    """
    E1 = torch.as_tensor(E1, dtype=dtype, device=device)
    E2 = torch.as_tensor(E2, dtype=dtype, device=device)

    C = torch.cdist(E1, E2, p=2)  # [n, N]
    C = C / (C.max() + 1e-12)
    return C



def sinkhorn_cost_torch_from_logits(
    logits,
    b,
    C,
    reg=0.05,
    sinkhorn_iters=200,
    tol=1e-6
):
    """
    Compute entropic OT cost on device in a differentiable way.

    logits: [n] unconstrained parameters; source weights are softmax(logits)
    b:      [N] target weights, usually uniform
    C:      [n, N] cost matrix
    reg:    entropy regularization epsilon

    Returns:
        cost: scalar
        w:    [n] source weights
        P:    [n, N] transport plan
    """
    # source weights on simplex
    w = torch.softmax(logits, dim=0)  # [n]

    log_a = torch.log(w + 1e-32)      # [n]
    log_b = torch.log(b + 1e-32)      # [N]
    M = -C / reg                      # [n, N]

    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    for _ in range(sinkhorn_iters):
        f_prev = f

        # f_i = log a_i - logsumexp_j(M_ij + g_j)
        f = log_a - torch.logsumexp(M + g.unsqueeze(0), dim=1)

        # g_j = log b_j - logsumexp_i(M_ij + f_i)
        g = log_b - torch.logsumexp(M + f.unsqueeze(1), dim=0)

        err = torch.max(torch.abs(f - f_prev))
        if err.item() < tol:
            break

    logP = f.unsqueeze(1) + g.unsqueeze(0) + M
    P = torch.exp(logP)

    cost = torch.sum(P * C)
    return cost, w, P



def optimize_w_strict_ot_gpu(
    E1,
    E2,
    reg=0.05,
    outer_iters=300,
    sinkhorn_iters=200,
    lr=0.1,
    device="cuda",
    dtype=torch.float32,
    verbose=False
):
    """
    Fully device-based optimization of source weights:
        min_w OT_reg((E1, w), (E2, uniform))

    Uses:
      - logits parameterization: w = softmax(logits)
      - differentiable log-domain Sinkhorn
      - Adam optimizer

    Returns:
      w_star as numpy array on CPU
    """
    if reg <= 0 or outer_iters < 1 or sinkhorn_iters < 1 or lr <= 0:
        raise ValueError("Require positive reg, iteration counts, and learning rate")
    if len(E1) == 0 or len(E2) == 0:
        raise ValueError("OT requires nonempty source and target samples")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    device = torch.device(device)

    C = build_cost_matrix_torch(E1, E2, device=device, dtype=dtype)
    n, N = C.shape

    # uniform target marginal
    b = torch.full((N,), 1.0 / N, dtype=dtype, device=device)

    # initialize to uniform source weights
    logits = torch.zeros(n, dtype=dtype, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([logits], lr=lr)

    best_cost = None
    best_w = None

    for t in range(outer_iters):
        optimizer.zero_grad()

        cost, w, _ = sinkhorn_cost_torch_from_logits(
            logits=logits,
            b=b,
            C=C,
            reg=reg,
            sinkhorn_iters=sinkhorn_iters,
            tol=1e-6
        )

        cost.backward()
        optimizer.step()

        with torch.no_grad():
            ess = (w.sum() ** 2 / (w ** 2).sum()).item()

            if best_cost is None or cost.item() < best_cost:
                best_cost = cost.item()
                best_w = w.detach().clone()

            if verbose and (t % 20 == 0 or t == outer_iters - 1):
                print(
                    f"outer {t:3d} | OT_cost={cost.item():.6f} | "
                    f"ESS={ess:.1f} | "
                    f"w_min/max={w.min().item():.2e}/{w.max().item():.2e}"
                )

    result = best_w.detach().cpu().numpy()
    del C, b, logits, best_w
    torch.cuda.empty_cache()
    return result
