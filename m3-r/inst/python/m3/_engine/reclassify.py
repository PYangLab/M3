# AdaSampling-style self-training for classification (PyTorch)
# ------------------------------------------------------------
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional

def preprocess_with_pca(X_np: np.ndarray, var_target: float = 0.70, max_pcs: int = 20, min_pcs: int = 10):
    """
    X_np: (N, D) numpy array
    returns: Z_np (N, d'), scaler, pca
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_np)

    pca_tmp = PCA(n_components=min(max_pcs, Xs.shape[1]), svd_solver="full", random_state=0)
    Xp = pca_tmp.fit_transform(Xs)
    cum = np.cumsum(pca_tmp.explained_variance_ratio_)
    d_choose = np.searchsorted(cum, var_target) + 1
    d_choose = int(np.clip(d_choose, min_pcs, min(max_pcs, Xs.shape[1])))

    pca = PCA(n_components=d_choose, svd_solver="full", random_state=0)
    Z = pca.fit_transform(Xs)
    return Z.astype(np.float32), scaler, pca

class LinearCls(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

def train_one_model(model, X, y, sample_w=None, epochs=100, batch_size=256, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    if sample_w is None:
        sample_w = torch.ones(len(y), dtype=torch.float32, device=device)
    else:
        sample_w = sample_w.to(device).float()

    ds = TensorDataset(X, y, sample_w)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for xb, yb, wb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss_vec = F.cross_entropy(logits, yb, reduction="none")
            loss = (loss_vec * wb).mean()
            loss.backward()
            opt.step()

@torch.no_grad()
def predict_all(model, X, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X = X.to(device)
    logits = model(X)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return probs.cpu().numpy(), pred.cpu().numpy(), conf.cpu().numpy()

def stratified_weighted_sample(class_indices, weights, m_total, rng):
    """
    class_indices: dict{c: np.array(indices_of_class_c)}
    weights: np.array shape (N,), per-sample sampling weights (normalised within each class)
    m_total: target total number of samples
    rng: np.random.Generator
    return: np.array of selected indices
    """
    N = sum(len(v) for v in class_indices.values())
    picks = []
    allocated = 0
    per_class_target = {}
    for c, idx_c in class_indices.items():
        target_c = max(1, int(round(m_total * len(idx_c) / N)))
        per_class_target[c] = min(target_c, len(idx_c))
        allocated += per_class_target[c]
    if allocated > m_total:
        overflow = allocated - m_total
        for c in sorted(class_indices, key=lambda k: len(class_indices[k]), reverse=True):
            if overflow == 0: break
            if per_class_target[c] > 1:
                per_class_target[c] -= 1
                overflow -= 1
    elif allocated < m_total:
        gap = m_total - allocated
        for c in sorted(class_indices, key=lambda k: len(class_indices[k]), reverse=True):
            if gap == 0: break
            if per_class_target[c] < len(class_indices[c]):
                per_class_target[c] += 1
                gap -= 1

    for c, idx_c in class_indices.items():
        w_c = weights[idx_c].clip(1e-8)
        w_c = w_c / w_c.sum()
        k_c = per_class_target[c]
        take = idx_c if len(idx_c) <= k_c else rng.choice(idx_c, size=k_c, replace=False, p=w_c)
        picks.append(take)
    return np.concatenate(picks)

def ada_self_training(
    X_tensor: torch.Tensor,
    y0_tensor: torch.Tensor,
    num_classes: Optional[int] = None, #num_classes: int | None = None,
    n_iters: int = 5,
    seed_ratio: float = 0.2,
    sample_ratio: float = 0.8,
    alpha_start: float = 1.5,
    alpha_end: float = 1.0,
    tau: float = 0.3,
    relabeled_weight: float = 0.5,
    use_pca: bool = True,
    pca_var: float = 0.70,
    max_pcs: int = 20,
    min_pcs: int = 10,
    epochs_per_iter: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    random_state: int = 42,
    device: Optional[str] = None, #device: str | None = None,
    print_progress: bool = True,
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    rng = np.random.default_rng(random_state)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_np = X_tensor.detach().cpu().numpy().astype(np.float32)
    y0_np = y0_tensor.detach().cpu().numpy().astype(int)
    N, D = X_np.shape
    if num_classes is None:
        num_classes = int(np.unique(y0_np).size)

    if use_pca:
        Z_np, scaler, pca = preprocess_with_pca(X_np, var_target=pca_var, max_pcs=max_pcs, min_pcs=min_pcs)
    else:
        scaler, pca = None, None
        Z_np = X_np

    Z = torch.from_numpy(Z_np).to(torch.float32)

    model = LinearCls(Z.shape[1], num_classes)

    class_indices = {c: np.where(y0_np == c)[0] for c in np.unique(y0_np)}
    seed_k = max(1, int(round(seed_ratio * N)))
    seed_idx_list = []
    allocated = 0
    for c, idx_c in class_indices.items():
        k_c = max(1, int(round(seed_k * len(idx_c) / N)))
        take_c = idx_c if len(idx_c) <= k_c else rng.choice(idx_c, size=k_c, replace=False)
        seed_idx_list.append(take_c)
        allocated += len(take_c)
    seed_idx = np.concatenate(seed_idx_list)
    train_one_model(model,
                    X=Z[seed_idx],
                    y=torch.from_numpy(y0_np[seed_idx]),
                    sample_w=None,
                    epochs=epochs_per_iter,
                    batch_size=batch_size,
                    lr=lr,
                    device=device)

    history = []

    for it in range(n_iters):
        probs, pred_all, conf_all = predict_all(model, Z, device=device)
        p_true = probs[np.arange(N), y0_np]

        alpha_t = alpha_start + (alpha_end - alpha_start) * (it / max(1, n_iters - 1))
        w = np.clip(p_true, 1e-6, 1.0) ** float(alpha_t)

        m_total = max(1, int(round(sample_ratio * N)))
        class_indices = {c: np.where(y0_np == c)[0] for c in np.unique(y0_np)}
        train_idx = stratified_weighted_sample(class_indices, w, m_total, rng)

        train_y = y0_np[train_idx].copy()
        low_conf_mask = (p_true[train_idx] < tau)
        n_relabel = int(low_conf_mask.sum())
        train_y[low_conf_mask] = pred_all[train_idx][low_conf_mask]

        sample_w = np.ones(train_idx.size, dtype=np.float32)
        if n_relabel > 0 and relabeled_weight is not None:
            sample_w[low_conf_mask] = float(relabeled_weight)

        train_one_model(model,
                        X=Z[train_idx],
                        y=torch.from_numpy(train_y),
                        sample_w=torch.from_numpy(sample_w),
                        epochs=epochs_per_iter,
                        batch_size=batch_size,
                        lr=lr,
                        device=device)

        probs_after, pred_after, conf_after = predict_all(model, Z, device=device)
        acc_all = (pred_after == y0_np).mean()
        history.append({
            "iter": it+1,
            "alpha": alpha_t,
            "train_size": int(train_idx.size),
            "mean_ptrue_train": float(p_true[train_idx].mean()),
            "relabeled": n_relabel,
            "acc_all_vs_y0": float(acc_all),
        })
        if print_progress:
            print(f"[iter {it+1}/{n_iters}] "
                  f"train={train_idx.size}, relabeled={n_relabel}, "
                  f"alpha={alpha_t:.2f}, mean_ptrue_train={p_true[train_idx].mean():.4f}, "
                  f"acc_all_vs_y0={acc_all:.4f}")

    final_probs, final_pred, final_conf = predict_all(model, Z, device=device)

    return {
        "model": model,
        "preprocess": {"scaler": scaler, "pca": pca},
        "history": history,
        "final_pred": final_pred,          # N
        "final_conf": final_conf,          # N
        "final_probs": final_probs,        # N x K
    }

