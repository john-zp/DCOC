# DCOC: Dynamic Kernel Online Classification for Fake‑Review Detection

This repo contains a clean, **DCOC‑only** implementation for fake‑review detection in text streams, plus a minimal preprocessing pipeline. DCOC is a kernel online learner with a **slope‑adjusted ramp loss** and an optional **sliding‑window budget** over support vectors.

> Why DCOC? The slope‑adjusted ramp loss keeps the penalty **bounded** for hard outliers (robustness) while letting you **tune the slope** near the margin (adapt to different noise levels). The windowed variant caps model size for streaming use.

---

## Repository structure

```
.
├─ data/                      # processed datasets produced by preprocessing (e.g., *_X.npy, *_y.csv, *_X_noise.npy ...)
├─ pynolca/                   # (optional) local pynolca package/build if you use it
├─ DCOC.py                    # DCOC (non-windowed) runner, if you keep a baseline variant
├─ DCOC_window_param.py       # DCOC (windowed) trainer; window is a single CLI parameter
└─ text_preprocess_generic.py # generic TF‑IDF + optional noise; no hard‑coded paths
```

---

## Installation

Requirements (CPU‑only is fine):

* Python ≥ 3.10 (paper used 3.11.5)
* `numpy`, `pandas`, `scikit‑learn`, `matplotlib`
* Optional: `psutil` (for RSS tracking); if missing, only tracemalloc peak is reported
* Optional: `pynolca` (provides a fast RBF kernel). If not present, we use a lightweight RBF fallback.

```bash
pip install numpy pandas scikit-learn matplotlib psutil
# optional
pip install pynolca  # if available in your environment
```

---

## Public datasets (summary)

The paper evaluates on public review corpora including **TripAdvisor**, **Yelp**, and **Amazon**:

* **TripAdvisor (Chicago hotels)**: 800 positive (400 truthful + 400 Turker fakes) and 800 negative (400 truthful from Expedia/Hotels.com/Orbitz/Priceline/TripAdvisor/Yelp + 400 Turker fakes).
* **TripAdvisor + Yelp (hotel/restaurant/doctor)**: 1200 truthful + 1636 fake (mix of Turker and domain‑expert generated).
* **Amazon**: multiple product categories (e.g., Books, DVDs, Kitchen, Electronics), with domain sizes varying by category.

> Noise injection in experiments: additional **10% / 20% / 30%** pseudo‑reviews generated from the in‑corpus vocabulary with **random labels** to simulate noisy streams.

> **Note:** This repo does **not** redistribute datasets. Please follow the original sources’ licenses and obtain the data from their official releases.

---

## Using the provided `/data` layout (your screenshot)

Your `data/` folder already contains split files per **dataset** and **noise level** with the following patterns:

* Clean features/labels:

  * `<dataset>_X_<level>.npy` (e.g., `deception_X_0.npy`, `deception_X_0.2.npy`)
  * `<dataset>_y_<level>_label.csv` (e.g., `deception_y_0_label.csv`, `deception_y_0.2_label.csv`)
* Noise to be appended to **TRAIN only** (optional):

  * `<dataset>_X_noise_<level>.npy` (e.g., `deception_X_noise_0.3.npy`)
  * `<dataset>_y_noise_<level>_label.csv`

> The training script reads **the last column** of the label CSV as `y` (column name is not required).

### Train on a single split (clean only)

```bash
# Example: deception, level 0.2
python DCOC_window_param.py \
  --label data/deception_y_0.2_label.csv \
  --features data/deception_X_0.2.npy \
  --window 1000 --train-ratio 0.8 --rounds 5 --seed 1234 --tag deception_0.2
```

### Train on multiple splits in a loop

**bash**

```bash
for ds in deception muti; do
  for lev in 0 0.1 0.2 0.3; do
    python DCOC_window_param.py \
      --label data/${ds}_y_${lev}_label.csv \
      --features data/${ds}_X_${lev}.npy \
      --window 1000 --train-ratio 0.8 --rounds 5 --seed 1234 --tag ${ds}_${lev}
  done
done
```

**PowerShell**

```powershell
$datasets = @('deception','muti')
$levels   = @( '0','0.1','0.2','0.3')
foreach ($ds in $datasets) {
  foreach ($lev in $levels) {
    python DCOC_window_param.py `
      --label    "data/${ds}_y_${lev}_label.csv" `
      --features "data/${ds}_X_${lev}.npy" `
      --window 1000 --train-ratio 0.8 --rounds 5 --seed 1234 --tag "${ds}_${lev}"
  }
}
```

### If you want to include noise 

`DCOC_window_param.py` expects one feature file and one label file. To **append noise to TRAIN only**, first combine the clean and noise sets into a new pair before training (example below keeps the test set clean by passing the combined file as the *training source* when the script internally splits):

```bash
python - << 'PY'
import numpy as np, pandas as pd
clean_X = np.load('data/deception_X_0.2.npy')
clean_y = pd.read_csv('data/deception_y_0.2_label.csv')['y' if 'y' in pd.read_csv('data/deception_y_0.2_label.csv').columns else pd.read_csv('data/deception_y_0.2_label.csv').columns[-1]].to_numpy()
noise_X = np.load('data/deception_X_noise_0.2.npy')
noise_y = pd.read_csv('data/deception_y_noise_0.2_label.csv').iloc[:,-1].to_numpy()
X = np.vstack([clean_X, noise_X])
y = np.concatenate([clean_y, noise_y])
np.save('data/deception_X_0.2_combined.npy', X)
pd.DataFrame({'y': y}).to_csv('data/deception_y_0.2_combined.csv', index=False)
print('Saved combined: data/deception_X_0.2_combined.npy / data/deception_y_0.2_combined.csv')
PY

python DCOC_window_param.py \
  --label data/deception_y_0.2_combined.csv \
  --features data/deception_X_0.2_combined.npy \
  --window 1000 --train-ratio 0.8 --rounds 5 --seed 1234 --tag deception_0.2_noise
```

If you prefer, you can script this combination for every dataset/level.

---

## Data preprocessing (TF‑IDF + optional noise)

`text_preprocess_generic.py` is path‑agnostic and accepts any number of labeled sources via CLI. It lowercases, strips punctuation/digits, removes stop‑words, and builds a TF‑IDF representation.

### Example

```bash
python text_preprocess_generic.py \
  --src "./raw/truth/*/,truthful" \
  --src "./raw/deceptive/*/,deceptive" \
  --label-map "truthful=1" "deceptive=-1" \
  --max-features 1000 \
  --noise-rate 0.3 \
  --seed 42 \
  --save-prefix data/reviews
```

**Outputs into `data/`** (when `--save-prefix` points to `data/...`):

* `data/reviews_X.npy`, `data/reviews_y.csv` (clean)
* `data/reviews_X_noise.npy`, `data/reviews_y_noise.csv` (noise; only if `--noise-rate>0`)
* `data/reviews_X_combined.npy`, `data/reviews_y_combined.csv` (clean + noise)
* `data/reviews_vocab.csv`

---

## Training DCOC (window as a parameter)

Use `DCOC_window_param.py`. The **window** (budget) limits the number of most‑recent support vectors.

```bash
python DCOC_window_param.py \
  --label data/reviews_y.csv \
  --features data/reviews_X.npy \
  --window 1000 \
  --train-ratio 0.8 \
  --rounds 5 \
  --seed 1234 \
  --tag YelpZip_0
```

> If you produced noise files and want to mix them into training yourself, use the `_combined` outputs from `data/`.

### Outputs 

* `<tag>_DCOC_w<window>_Performance.csv` — mean±std over `--rounds` for Accuracy / Precision / Recall / F1, time, SV count, memory, and model bytes.
* `<tag>_DCOC_w<window>_train.pdf` — training misclassification rate (mean±std across rounds).
* `<tag>_DCOC_w<window>_roc.png`, `<tag>_DCOC_w<window>_pr.png` — last‑round ROC/PR.
* `<tag>_DCOC_w<window>.pkl` — raw metrics and loss traces.

---

## Hyperparameters

**Loss parameters**

* Truncation level **s**: typically in **[−2.0, −0.1]** (coarser to finer steps acceptable)
* Slope factor **α**: typically in **[0.25, 2.0]**

**Capacity parameter**

* **window**: integer ≥ 1 (e.g., 500, 1000, 2000, 4000). Larger window → higher capacity (more SVs), higher memory/time.

**Kernel**

* RBF kernel via `pynolca` when available; otherwise, a small pure‑NumPy RBF fallback (γ defaults to 1/d unless provided).

### Grids used in the paper

| Model                        | Parameter ranges                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| **DCOC (ours)**              | `s ∈ {−1.0, −0.9, …, 0}`; `α ∈ {0.25, 0.5, …, 2.0}`                              |
| Perceptron                   | learning rate `η ∈ {1e−5, 1e−4, 1e−3, 1e−2}`                                         |
| Pegasos                      | regularization `λ ∈ {1e−4, 1e−3, 1e−2}`                                              |
| OnlineSVM                    | `C ∈ {1e−4, 1e−3, 1e−2, 1e−1, 1, 10}`                                                |
| Passive‑aggressive (PA)      | `PA ∈ {−1.0, −0.9, …, −0.1}`                                                               |
| Ahpatron (budget perceptron) | `budget ∈ {200, 500, 800}`; `η ∈ {1e−3, 1e−2, 1e−1}`; `removal ∈ {oldest, smallest}` |

---

## Model selection protocol 

* **5‑fold cross‑validation** on a stratified **20% subset of the training data** to pick `(s, α)` for the chosen `window`.
* Retrain DCOC on the **full training split** with the best params; evaluate on the **held‑out test**.
* Repeat for `--rounds` to report mean±std.

> Tip: for speed on large `X`, we also sub‑sample the training set when cross‑validating (`dcoc_window_param.py` uses ~30% of the training split for CV). Adjust if you prefer.

---

## Metrics

We report **Accuracy**, **Precision**, **Recall**, and **F1** and also provide ROC/PR curves. See the CSV/plots per run under your chosen `--tag`.

---

## Reproducibility & environment

* OS/CPU: any modern CPU works; the paper used Windows 10 (64‑bit), **Intel® Core™ i7‑12650H**, **16 GB RAM**, **CPU‑only** training/inference.
* Python: ≥ 3.10 (paper used 3.11.5)

For strict reproducibility, pin package versions (example):

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib==3.8.4
psutil==5.9.8
```

---

## How to cite

If you use this codebase in academic work, please cite the accompanying paper describing DCOC, its slope‑adjusted ramp loss, experimental setup on TripAdvisor/Yelp/Amazon, and the noise protocol.

---


## Acknowledgments

Thanks to the maintainers of the original public datasets (TripAdvisor, Yelp, Amazon) and the libraries used in this implementation.



