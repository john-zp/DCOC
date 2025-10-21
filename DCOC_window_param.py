# -*- coding: utf-8 -*-
"""
DCOC with sliding window (single-parameter version)
--------------------------------------------------
This script exposes the window size as a runtime parameter instead of sweeping.

Example:
    python dcoc_window_param.py \
        --label data/YelpZip_label.csv \
        --features data/YelpZip_X_array.npy \
        --window 1000 \
        --train-ratio 0.8 \
        --rounds 5 \
        --tag YelpZip_0

Outputs (tagged by `--tag` and `--window`):
    <tag>_DCOC_w<window>_Performance.csv
    <tag>_DCOC_w<window>_train.pdf
    <tag>_DCOC_w<window>_roc.png
    <tag>_DCOC_w<window>_pr.png
    <tag>_DCOC_w<window>.pkl
"""
import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# =========================
# Kernel: prefer pynolca, otherwise fall back to a simple RBF implementation
# =========================
try:
    import pynolca  # type: ignore
    _KERNEL = pynolca.kernel.RBF_Kernel()
except Exception:
    class RBF_Kernel_Fallback:
        def __init__(self, gamma=None):
            self.gamma = gamma  # If None, will be set to 1/d at first call
        def _ensure_gamma(self, X):
            if self.gamma is None:
                d = X.shape[1] if X.ndim == 2 else X.size
                self.gamma = 1.0 / max(d, 1)
        def compute_kernel(self, A, B):
            A = np.atleast_2d(A)
            B = np.atleast_2d(B)
            self._ensure_gamma(A)
            A2 = np.sum(A*A, axis=1)[:, None]
            B2 = np.sum(B*B, axis=1)[None, :]
            K = np.exp(-self.gamma * (A2 + B2 - 2.0 * A.dot(B.T)))
            if K.shape[1] == 1:
                return K[:, 0]
            if K.shape[0] == 1:
                return K[0]
            return K
    _KERNEL = RBF_Kernel_Fallback()

# =========================
# Memory monitoring (tracemalloc required, psutil optional)
# =========================
import tracemalloc
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
    _PROC = psutil.Process(os.getpid())
except Exception:
    _HAS_PSUTIL = False
    _PROC = None

class MemoryMonitor:
    def __init__(self):
        self._start_rss = None
        self.peak_tracemalloc = 0  # bytes
    def __enter__(self):
        if _HAS_PSUTIL:
            self._start_rss = _PROC.memory_info().rss
        tracemalloc.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        _, peak = tracemalloc.get_traced_memory()
        self.peak_tracemalloc = peak
        tracemalloc.stop()
    @property
    def rss_delta_bytes(self):
        if not _HAS_PSUTIL or self._start_rss is None:
            return None
        now = _PROC.memory_info().rss
        return max(0, int(now) - int(self._start_rss))

def bytes_to_mb(b):
    return (b or 0) / (1024.0 * 1024.0)

# =========================
# Utilities
# =========================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def shuffle_data(X, y, seed=None):
    assert len(X) == len(y), "X and y must have the same length"
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    return X[idx], y[idx]

def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    TP = np.sum((y_pred ==  1) & (y_true ==  1))
    FP = np.sum((y_pred ==  1) & (y_true == -1))
    TN = np.sum((y_pred == -1) & (y_true == -1))
    FN = np.sum((y_pred == -1) & (y_true ==  1))
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    acc       = (TP + TN) / max((TP + FP + TN + FN), 1)
    return acc, precision, recall, f1

# =========================
# Base class with SV storage/decision
# =========================
class KernelOnlineBase:
    def __init__(self, kernel):
        self.kernel = kernel
        self.X_sv = None     # (m, d)
        self.coef = None     # (m,)
        self.errors = 0
        self.error_rates = []
    def _append_sv(self, x, c):
        if self.X_sv is None:
            self.X_sv = x[None, ...]
            self.coef = np.array([c], dtype=float)
        else:
            self.X_sv = np.vstack([self.X_sv, x])
            self.coef = np.append(self.coef, c)
    def decision_function(self, X, chunk=1024):
        if self.X_sv is None or self.coef is None or len(self.coef) == 0:
            return np.zeros(len(X), dtype=float)
        scores = np.zeros(len(X), dtype=float)
        for i in range(0, len(X), chunk):
            K = self.kernel.compute_kernel(self.X_sv, X[i:i+chunk])
            scores[i:i+chunk] = np.dot(self.coef, K)
        return scores
    def predict_scores(self, X):
        return self.decision_function(X)
    def predict(self, X):
        scores = self.predict_scores(X)
        return np.where(scores >= 0.0, 1, -1).astype(int)
    def predict_proba(self, X):
        return sigmoid(self.predict_scores(X))
    def get_num_sv(self):
        return 0 if self.coef is None else int(np.sum(self.coef != 0))
    def model_nbytes(self):
        total = 0
        if self.X_sv is not None:
            total += getattr(self.X_sv, 'nbytes', 0)
        if self.coef is not None:
            total += getattr(self.coef, 'nbytes', 0)
        return int(total)

# =========================
# DCOC with sliding window (budget)
# =========================
class DCOCWindow(KernelOnlineBase):
    """
    Windowed (budget) DCOC:
      - psi = clip(alpha*(1 - y f(x)), 0, 1 - s), s <= 0
      - If 0 < psi < (1 - s): append SV with theta = y * psi / K(x,x)
      - After appending, if SV count exceeds the window, keep the most recent `window` SVs (FIFO)
    """
    def __init__(self, kernel, s=-0.5, alpha=1.0, window=1000):
        super().__init__(kernel)
        assert s <= 0.0, "s must be <= 0"
        assert int(window) >= 1, "window must be a positive integer"
        self.s = float(s)
        self.alpha = float(alpha)
        self.window = int(window)
    def _psi(self, y, score):
        val = self.alpha * (1.0 - y*score)
        return min(max(val, 0.0), 1.0 - self.s)
    def _evict_if_needed(self):
        if self.X_sv is not None and len(self.coef) > self.window:
            self.X_sv = self.X_sv[-self.window:, :]
            self.coef = self.coef[-self.window:]
    def fit(self, X, y, X_val=None, y_val=None, eval_every=8):
        self.errors, self.error_rates = 0, []
        for t in range(len(y)):
            x_i, y_i = X[t], y[t]
            score = 0.0 if self.X_sv is None else self.kernel.compute_kernel(self.X_sv, x_i).dot(self.coef)
            psi = self._psi(y_i, score)
            if 0.0 < psi < (1.0 - self.s):
                kxx = float(self.kernel.compute_kernel(x_i, x_i))
                theta = (y_i * psi) / (kxx if kxx > 0 else 1.0)
                self._append_sv(x_i, theta)
                self._evict_if_needed()
            if np.sign(score) != y_i:
                self.errors += 1
            self.error_rates.append(self.errors/(t+1))

# =========================
# CV for DCOCWindow
# =========================

def select_param_dcoc_window(X, y, kernel, grid_dcoc, window, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    best_val, best_params = -np.inf, None
    for params in grid_dcoc:
        accs = []
        for tr, va in kf.split(X):
            m = DCOCWindow(kernel, window=window, **params)
            m.fit(X[tr], y[tr])
            y_pred = m.predict(X[va])
            accs.append((y_pred == y[va]).mean())
        mean_acc = float(np.mean(accs))
        if mean_acc > best_val:
            best_val, best_params = mean_acc, params
    return best_val, best_params

# =========================
# Plot helpers
# =========================

def plot_roc(y_true, scores, title, filename):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--',lw=1,color='gray',alpha=0.5)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(filename, bbox_inches='tight'); plt.close()

def plot_pr(y_true, scores, title, filename):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f'AP={ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(filename, bbox_inches='tight'); plt.close()

# =========================
# Main experiment (single window value)
# =========================

def run_dcoc_with_window(
    X_clean, y_clean, kernel, grids_DCOC, window=1000,
    training_ratio=0.8, T=5, seed=1234, tag="YelpZip_0"
):
    metrics = {k: [] for k in ['acc','prec','rec','f1','time','nsv','mem_peak_mb','rss_delta_mb','model_bytes']}
    loss_traces = []

    for t in range(T):
        # Shuffle and split (TEST kept clean)
        X_all, y_all = shuffle_data(X_clean, y_clean, seed=seed+t)
        n_tr = int(training_ratio * len(y_all))
        X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]
        X_te, y_te = X_all[n_tr:], y_all[n_tr:]

        # CV on a small subset for speed
        sub_n = max(10, int(0.3 * len(X_tr)))
        X_sub, y_sub = X_tr[:sub_n], y_tr[:sub_n]
        cv, best = select_param_dcoc_window(X_sub, y_sub, kernel, grids_DCOC, window=window, folds=5)
        print(f"[{tag}] window={window} | round {t+1}/{T}: best={best}, CV-acc={cv:.4f}")

        clf = DCOCWindow(kernel, window=window, **best)
        with MemoryMonitor() as mm:
            t0 = time.time(); clf.fit(X_tr, y_tr); tr_time = time.time() - t0
        peak_mb = bytes_to_mb(mm.peak_tracemalloc)
        rss_mb  = bytes_to_mb(mm.rss_delta_bytes) if mm.rss_delta_bytes is not None else np.nan
        model_b = clf.model_nbytes()

        y_sc = clf.predict_scores(X_te)
        y_pd = np.where(y_sc >= 0, 1, -1).astype(int)
        acc, pr, rc, f1 = compute_metrics(y_te, y_pd)

        metrics['acc'].append(acc); metrics['prec'].append(pr); metrics['rec'].append(rc); metrics['f1'].append(f1)
        metrics['time'].append(tr_time); metrics['nsv'].append(clf.get_num_sv())
        metrics['mem_peak_mb'].append(peak_mb); metrics['rss_delta_mb'].append(rss_mb); metrics['model_bytes'].append(model_b)
        loss_traces.append(np.array(clf.error_rates))

        # Plots for this round (will be overwritten each round; last one kept)
        plot_roc(y_te, y_sc, title=f'ROC (DCOCWindow={window})', filename=f'{tag}_DCOC_w{window}_roc.png')
        plot_pr (y_te, y_sc, title=f'PR (DCOCWindow={window})',  filename=f'{tag}_DCOC_w{window}_pr.png')

    # Export CSV: mean±std
    row = {
        'Window': window,
        'Accuracy':     f"{np.mean(metrics['acc']):.4f}±{np.std(metrics['acc']):.4f}",
        'Precision':    f"{np.mean(metrics['prec']):.4f}±{np.std(metrics['prec']):.4f}",
        'Recall':       f"{np.mean(metrics['rec']):.4f}±{np.std(metrics['rec']):.4f}",
        'F1':           f"{np.mean(metrics['f1']):.4f}±{np.std(metrics['f1']):.4f}",
        'Time(s)':      f"{np.mean(metrics['time']):.4f}±{np.std(metrics['time']):.4f}",
        'SV Num':       f"{np.mean(metrics['nsv']):.2f}±{np.std(metrics['nsv']):.2f}",
        'PeakHeap(MB)': f"{np.nanmean(metrics['mem_peak_mb']):.2f}±{np.nanstd(metrics['mem_peak_mb']):.2f}",
        'RSSΔ(MB)':     f"{np.nanmean(metrics['rss_delta_mb']):.2f}±{np.nanstd(metrics['rss_delta_mb']):.2f}",
        'ModelBytes':   f"{np.mean(metrics['model_bytes']):.0f}±{np.std(metrics['model_bytes']):.0f}",
    }
    pd.DataFrame([row]).to_csv(f'{tag}_DCOC_w{window}_Performance.csv', index=False)

    # Training misclassification rate PDF (mean±std across T rounds)
    plt.style.use('seaborn-v0_8-whitegrid')
    pp = PdfPages(f'{tag}_DCOC_w{window}_train.pdf')
    plt.figure()
    if len(loss_traces) > 0:
        min_len = min(len(t) for t in loss_traces)
        arr = np.vstack([t[:min_len] for t in loss_traces])
        mean = arr.mean(axis=0); std = arr.std(axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=f'window={window}')
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel('Steps'); plt.ylabel('Misclassification rate'); plt.legend(); plt.grid(True, alpha=0.2)
    pp.savefig(bbox_inches='tight'); plt.close(); pp.close()

    # Persist objects
    with open(f'{tag}_DCOC_w{window}.pkl', 'wb') as f:
        pickle.dump({'metrics': metrics, 'loss_traces': loss_traces, 'window': window}, f)

    print(f"[DCOC-Window] Done: tag={tag}, window={window} -> "
          f"{tag}_DCOC_w{window}_Performance.csv / {tag}_DCOC_w{window}_train.pdf / "
          f"{tag}_DCOC_w{window}_roc.png / {tag}_DCOC_w{window}_pr.png / {tag}_DCOC_w{window}.pkl")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DCOC with a fixed sliding window size")
    parser.add_argument('--label', type=str, default='data/YelpZip_label.csv', help='Path to label CSV (last column is y)')
    parser.add_argument('--features', type=str, default='data/YelpZip_X_array.npy', help='Path to feature .npy file')
    parser.add_argument('--window', type=int, default=1000, help='Sliding window (budget) size')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training ratio (rest used for test)')
    parser.add_argument('--rounds', type=int, default=5, help='Number of repeated runs for mean±std')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed base')
    parser.add_argument('--tag', type=str, default='YelpZip_0', help='Prefix for output files')
    args = parser.parse_args()

    # Load data
    df_label = pd.read_csv(args.label)
    X_all = np.load(args.features)
    y_all = np.array(df_label[df_label.columns[-1]], dtype=np.int16)

    # Kernel (pynolca if available; otherwise fallback)
    kernel = _KERNEL

    # DCOC grid (window is fixed via --window)
    grids_DCOC = [
        {'s': s, 'alpha': a}
        for s in np.linspace(-1.0, 0, 10)
        for a in np.linspace(0.1, 1.0, 10)
    ]

    run_dcoc_with_window(
        X_clean=X_all,
        y_clean=y_all,
        kernel=kernel,
        grids_DCOC=grids_DCOC,
        window=int(args.window),
        training_ratio=float(args.train_ratio),
        T=int(args.rounds),
        seed=int(args.seed),
        tag=args.tag,
    )
