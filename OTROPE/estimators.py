"""Classifier importance weights and OTROPE/DM/IS/DR/PPI policy-value estimates."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def estimate_weights_by_classifier(
    E_train: np.ndarray,
    E_test: np.ndarray,
    random_state: int = 42,
):
    """
    Estimate importance weights w(x) = (π₀/π₁) * η(x) / (1 - η(x))
    via logistic regression trained to distinguish train (label=0) from test (label=1).

    π₀/π₁ is set to n_train/n_test to correct for unbalanced class sizes.
    Returns density-ratio weights divided by n_train (not self-normalized).
    """
    X = np.vstack([E_train, E_test])
    y = np.concatenate([np.zeros(len(E_train)), np.ones(len(E_test))])

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X, y)

    n_iter = int(clf.n_iter_[0])
    if n_iter >= 1000:
        print(f"[WARNING] LogisticRegression did NOT converge (n_iter={n_iter} == max_iter). Consider increasing max_iter.")
    else:
        print(f"[INFO] LogisticRegression converged in {n_iter} iterations.")

    eta = clf.predict_proba(E_train)[:, 1]  # P(l=1 | x) for train samples
    odds = eta / np.clip(1.0 - eta, 1e-12, None)
    prior_ratio = len(E_train) / len(E_test)
    w = prior_ratio * odds
    w = w / len(E_train)
    return w



def estimate_policy_values(train_eval, test_eval, prediction_col, w_star, w_cls):
    """Compute the original estimators from aligned predictions and source weights.

    Target labels are used only for the reported benchmark ground truth.
    """
    y_tr = train_eval["true"].to_numpy(dtype=float)
    y_te = test_eval["true"].to_numpy(dtype=float)

    yhat_tr = train_eval[prediction_col].to_numpy(dtype=float)
    yhat_te = test_eval[prediction_col].to_numpy(dtype=float)

    expert_cols = [col for col in test_eval.columns if col.startswith("expert_")]
    te_votes = test_eval[expert_cols].sum(axis=1)
    tr_votes = train_eval[expert_cols].sum(axis=1)

    K = len(expert_cols)
    if K == 0:
        raise ValueError("Evaluation CSV must include expert_ columns")

    maj_te = (te_votes > K / 2).astype(int)
    maj_tr = (tr_votes > K / 2).astype(int)

    est_dm_maj = maj_te.mean()

    ## PPI++
    n = len(y_tr)
    N = len(y_te)

    cov_pred = np.cov(y_tr, yhat_tr, ddof=1)[0, 1]
    yhat_all = np.concatenate([yhat_tr, yhat_te])
    var_pred = np.var(yhat_all, ddof=1)
    lambda_pred = cov_pred / ((1 + n / N) * var_pred) if var_pred > 0 and n > 1 else 0.0
    print("lambda_hat =", lambda_pred)

    cov_maj = np.cov(y_tr, maj_tr, ddof=1)[0, 1]
    maj_all = np.concatenate([maj_tr, maj_te])
    var_maj = np.var(maj_all, ddof=1)
    lambda_maj = cov_maj / ((1 + n / N) * var_maj) if var_maj > 0 and n > 1 else 0.0
    print("lambda_hat =", lambda_maj)

    # --------------------------------------------------
    # Baselines / estimates
    # --------------------------------------------------
    est_dm = yhat_te.mean()
    est_naive = yhat_tr.mean()

    w_ot = np.asarray(w_star, dtype=float)
    w_ot = w_ot / w_ot.sum()

    w_uniform = np.ones_like(w_ot) / len(w_ot)

    est_ot = np.sum(w_ot * y_tr)
    est_otrope = est_dm + np.sum(w_ot * (y_tr - yhat_tr))
    est_otrope_maj = est_dm_maj + np.sum(w_ot * (y_tr - maj_tr))
    est_ppi = est_dm + np.sum(w_uniform * (y_tr - yhat_tr))
    est_ppi_maj = est_dm_maj + np.sum(w_uniform * (y_tr - maj_tr))

    est_is = np.sum(w_cls * y_tr)
    est_dr = est_dm + np.sum(w_cls * (y_tr - yhat_tr))
    est_dr_maj = est_dm_maj + np.sum(w_cls * (y_tr - maj_tr))

    est_ppi_pp = y_tr.mean() + lambda_pred * (yhat_te.mean() - yhat_tr.mean())
    est_ppi_pp_maj = y_tr.mean() + lambda_maj * (maj_te.mean() - maj_tr.mean())

    true_target = y_te.mean()

    results = {
        "n_obs": len(w_ot),
        "naive": float(est_naive),
        "ot": float(est_ot),
        "dm": float(est_dm),
        "dm_majority": float(est_dm_maj),
        "ppi": est_ppi,
        "ppi_majority": est_ppi_maj,
        "ppi++": est_ppi_pp,
        "ppi++_majority": est_ppi_pp_maj,
        "otrope": est_otrope,
        "otrope_majority": est_otrope_maj,
        "is": float(est_is),
        "dr": float(est_dr),
        "dr_majority": float(est_dr_maj),
        "true": true_target,
    }

    return results
