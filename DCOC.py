# -*- coding: utf-8 -*-
"""
DCOC runner
=================
This script keeps only the DCOC algorithm (with ramp loss) and the minimal utilities/experiment loop.
Save as `DCOC.py` and run directly.

Dependencies: numpy, pandas, scikit-learn, matplotlib
Optional: psutil (for RSS monitoring), pynolca (preferred kernel if available)
"""
import os
import time
import pickle
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
    HAS_PYNOLCA = True
except Exception:
    HAS_PYNOLCA = False

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
            A2 = np.sum(A * A, axis=1)[:, None]
            B2 = np.sum(B * B, axis=1)[None, :]
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
            self._start_rss = _PROC.memory_info().rss  # bytes
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        _, peak = tracemalloc.get_traced_memory()  # bytes
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
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == -1))
    TN = np.sum((y_pred == -1) & (y_true == -1))
    FN = np.sum((y_pred == -1) & (y_true == 1))
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (TP + TN) / max((TP + FP + TN + FN), 1)
    return acc, precision, recall, f1

def plot_roc(y_true, scores, label, filename):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1, color="gray", alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(filename, bbox_inches="tight"); plt.close()

def plot_pr(y_true, scores, label, filename):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(filename, bbox_inches="tight"); plt.close()

# =========================
# Kernel online base (no window)
# =========================
class KernelOnlineBase:
    def __init__(self, kernel):
        self.kernel = kernel
        self.X_sv = None  # (m, d)
        self.coef = None  # (m,)
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
            K = self.kernel.compute_kernel(self.X_sv, X[i : i + chunk])
            scores[i : i + chunk] = np.dot(self.coef, K)
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
            total += getattr(self.X_sv, "nbytes", 0)
        if self.coef is not None:
            total += getattr(self.coef, "nbytes", 0)
        return int(total)

# =========================
# DCOC algorithm (kept only)
# =========================
class DCOC(KernelOnlineBase):
    """
    Ramp loss with adjustable slope:
    psi_{s,alpha}(y,f(x)) = min(max(alpha*(1 - y f(x)), 0), 1 - s),  s <= 0
    If 0 < psi < 1 - s:
        theta = y * psi / K(x,x)
        f <- f + theta K(x,·)
    """

    def __init__(self, kernel, s=-0.5, alpha=1.0):
        super().__init__(kernel)
        assert s <= 0.0, "s must be <= 0"
        self.s = float(s)
        self.alpha = float(alpha)

    def _psi(self, y, score):
        val = self.alpha * (1.0 - y * score)
        return min(max(val, 0.0), 1.0 - self.s)

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
            if np.sign(score) != y_i:
                self.errors += 1
            self.error_rates.append(self.errors / (t + 1))

# =========================
# Hyperparameter selection (grid: list[dict])
# =========================

def select_param(model_ctor, X, y, grid, folds=5):
    best_val, best_params = -np.inf, None
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for params in grid:
        accs = []
        for tr, va in kf.split(X):
            m = model_ctor(**params)
            m.fit(X[tr], y[tr])
            y_pred = m.predict(X[va])
            accs.append((y_pred == y[va]).mean())
        mean_acc = np.mean(accs)
        if mean_acc > best_val:
            best_val, best_params = mean_acc, params
    return best_val, best_params

# =========================
# Experiment with noise injected only into the TRAIN set (DCOC only)
# =========================

def run_experiment_with_noise(
    X_clean,
    y_clean,
    X_noise,
    y_noise,
    training_ratio=0.8,
    T=5,
    grids=None,
    kernel=None,
    seed=1234,
    level_tag="0",
    dataset_tag="pos",
):
    name = "DCOC"
    metrics = {
        name: {
            "acc": [],
            "prec": [],
            "rec": [],
            "f1": [],
            "time": [],
            "nsv": [],
            "mem_peak_mb": [],
            "rss_delta_mb": [],
            "model_bytes": [],
        }
    }
    loss_traces = {name: []}
    last_scores = {}
    dcoc_best_history = []

    def _run_dcoc(X_tr, y_tr, X_te, y_te, grid, round_idx=0):
        sub_n = max(10, int(0.3 * len(X_tr)))
        X_sub, y_sub = X_tr[:sub_n], y_tr[:sub_n]
        cv_score, best = select_param(lambda **kw: DCOC(kernel, **kw), X_sub, y_sub, grid)
        print(
            f"[{dataset_tag} | Level {level_tag}] [Round {round_idx + 1}/{T}] DCOC best params: {best} (CV acc={cv_score:.4f})"
        )
        dcoc_best_history.append(best)
        clf = DCOC(kernel, **best)
        with MemoryMonitor() as mm:
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            tr_time = time.time() - t0
        peak_mb = bytes_to_mb(mm.peak_tracemalloc)
        rss_mb = bytes_to_mb(mm.rss_delta_bytes) if mm.rss_delta_bytes is not None else np.nan
        model_b = clf.model_nbytes()
        y_sc = clf.predict_scores(X_te)
        y_pd = np.where(y_sc >= 0, 1, -1).astype(int)
        acc, pr, rc, f1 = compute_metrics(y_te, y_pd)
        for k, v in [
            ("acc", acc),
            ("prec", pr),
            ("rec", rc),
            ("f1", f1),
            ("time", tr_time),
            ("nsv", clf.get_num_sv()),
            ("mem_peak_mb", peak_mb),
            ("rss_delta_mb", rss_mb),
            ("model_bytes", model_b),
        ]:
            metrics["DCOC"][k].append(v)
        loss_traces["DCOC"].append(np.array(clf.error_rates))
        last_scores["DCOC"] = y_sc

    for t in range(T):
        # Shuffle the clean data first
        X_all, y_all = shuffle_data(X_clean, y_clean, seed=seed + t)
        # Split; keep TEST clean
        n_tr = int(training_ratio * len(y_all))
        X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]
        X_te, y_te = X_all[n_tr:], y_all[n_tr:]
        # Inject noise only into TRAIN
        if X_noise is not None and y_noise is not None and len(X_noise) > 0:
            X_tr = np.vstack([X_tr, X_noise])
            y_tr = np.hstack([y_tr, y_noise])
            X_tr, y_tr = shuffle_data(X_tr, y_tr, seed=seed + 10000 + t)

        # Run DCOC only
        _run_dcoc(X_tr, y_tr, X_te, y_te, grids["DCOC"], round_idx=t)

        # On the last round, draw ROC/PR (with level + dataset tags)
        if t == T - 1 and "DCOC" in last_scores:
            scores = last_scores["DCOC"]
            plot_roc(y_te, scores, label="DCOC", filename=f"{dataset_tag}_{level_tag}_roc_curve_dcoc.png")
            plot_pr(y_te, scores, label="DCOC", filename=f"{dataset_tag}_{level_tag}_pr_curve_dcoc.png")

    if len(dcoc_best_history) > 0:
        print(f"[{dataset_tag} | Level {level_tag}] DCOC best params by round: {dcoc_best_history}")
        print(f"[{dataset_tag} | Level {level_tag}] DCOC last-round best params: {dcoc_best_history[-1]}")

    return metrics, loss_traces

# =========================
# Load data & run batches across datasets + noise levels (DCOC only)
# =========================
if __name__ == "__main__":
    kernel = _KERNEL  # from pynolca or fallback

    # Hyperparameter grid for DCOC only
    grids = {
        "DCOC": [
            {"s": s, "alpha": a}
            for s in np.linspace(-1.0, 0, 10)
            for a in np.linspace(0.1, 1.0, 10)
        ]
    }

    datasets = ["pos", "neg", "deception", "muti"]
    levels = ["0", "0.1", "0.2", "0.3"]

    for ds in datasets:
        for lev in levels:
            base_label = f"data/{ds}_y_{lev}_label.csv"
            base_feat = f"data/{ds}_X_{lev}.npy"
            if not (os.path.exists(base_label) and os.path.exists(base_feat)):
                print(f"[Skip] Missing base data for {ds} level {lev}: {base_label} or {base_feat}")
                continue

            print(f"========== Running {ds} | level {lev} ==========")
            df_label = pd.read_csv(base_label)
            X_all = np.load(base_feat)
            y_all = np.array(df_label.iloc[:, -1], dtype=np.int16)

            noise_label_path = f"data/{ds}_y_noise_{lev}_label.csv"
            noise_feat_path = f"data/{ds}_X_noise_{lev}.npy"
            if os.path.exists(noise_label_path) and os.path.exists(noise_feat_path):
                df_noise = pd.read_csv(noise_label_path)
                X_noise = np.load(noise_feat_path)
                y_noise = np.array(df_noise.iloc[:, -1], dtype=np.int16)
                print(
                    f"Loaded noise set for {ds} level {lev}: {len(y_noise)} samples (appended to TRAIN only)"
                )
            else:
                X_noise, y_noise = None, None
                print(
                    f"No explicit noise files for {ds} level {lev}; training without extra noise samples."
                )

            metrics, loss_traces = run_experiment_with_noise(
                X_clean=X_all,
                y_clean=y_all,
                X_noise=X_noise,
                y_noise=y_noise,
                training_ratio=0.8,
                T=5,
                grids=grids,
                kernel=kernel,
                seed=1234,
                level_tag=lev,
                dataset_tag=ds,
            )

            # Summaries (tagged with dataset + level)
            summary = []
            dic = metrics["DCOC"]
            row = {
                "Algorithm": "DCOC",
                "Precision": f"{np.mean(dic['prec']):.4f}±{np.std(dic['prec']):.4f}",
                "Recall": f"{np.mean(dic['rec']):.4f}±{np.std(dic['rec']):.4f}",
                "F1": f"{np.mean(dic['f1']):.4f}±{np.std(dic['f1']):.4f}",
                "Accuracy": f"{np.mean(dic['acc']):.4f}±{np.std(dic['acc']):.4f}",
                "Time(s)": f"{np.mean(dic['time']):.4f}±{np.std(dic['time']):.4f}",
                "SV Num": f"{np.mean(dic['nsv']):.2f}±{np.std(dic['nsv']):.2f}",
                "PeakHeap(MB)": f"{np.nanmean(dic['mem_peak_mb']):.2f}±{np.nanstd(dic['mem_peak_mb']):.2f}",
                "RSSΔ(MB)": f"{np.nanmean(dic['rss_delta_mb']):.2f}±{np.nanstd(dic['rss_delta_mb']):.2f}",
                "ModelBytes": f"{np.mean(dic['model_bytes']):.0f}±{np.std(dic['model_bytes']):.0f}",
            }
            summary.append(row)
            pd.DataFrame(summary).to_csv(f"{ds}_{lev}_Performance.csv", index=False)

            # Training misclassification rate PDF (tagged with dataset + level)
            plt.style.use("seaborn-v0_8-whitegrid")
            pp = PdfPages(f"{ds}_{lev}_train.pdf")
            plt.figure()
            traces = loss_traces["DCOC"]
            if len(traces) > 0:
                arr = np.vstack([np.array(t) for t in traces])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                x = np.arange(len(mean))
                plt.plot(x, mean, label="DCOC")
                plt.fill_between(x, mean - std, mean + std, alpha=0.2)
            plt.xlabel("Steps")
            plt.ylabel("Misclassification rate")
            plt.legend()
            plt.grid(True, alpha=0.2)
            pp.savefig(bbox_inches="tight")
            plt.close()
            pp.close()

            # Persist objects
            with open(f"{ds}_{lev}.pkl", "wb") as f:
                pickle.dump({"metrics": metrics, "loss_traces": loss_traces}, f)

            print(
                f"Done {ds} | level {lev}. Files saved: "
                f"{ds}_{lev}_Performance.csv, {ds}_{lev}_train.pdf, "
                f"{ds}_{lev}_roc_curve_dcoc.png, {ds}_{lev}_pr_curve_dcoc.png, "
                f"{ds}_{lev}.pkl"
            )
