# 🌿 AmazonNet — Multi-Label Satellite Image Classification for Deforestation Detection

Can a deep learning model learn to detect illegal logging, slash-and-burn agriculture, and artisanal mining directly from satellite imagery — before any human analyst flags them?

This project builds a full computer vision pipeline — from exploratory data analysis to a deployed, production-optimized inference API with Grad-CAM visual explanations — to classify land use and atmospheric conditions in chips of Amazon rainforest imagery. It is a real-world multi-label classification problem where a single image can carry up to 9 simultaneous labels, and where missing a threat (false negative) is far more costly than a false alarm.

---

## 🚀 Live Demo

| | |
|---|---|
| **Frontend (Streamlit)** | [amazonnet.streamlit.app](https://pedrosall-amazonnet.streamlit.app) |
| **Backend API (FastAPI)** | [amazon-deforestation-cv.onrender.com](https://amazon-deforestation-cv.onrender.com) |
| **API docs (Swagger)** | [/docs](https://amazon-deforestation-cv.onrender.com/docs) |

> ⚠️ **Cold start notice:** the backend runs on a free-tier instance that sleeps after 15 minutes of inactivity. The first prediction after a period of inactivity can take up to 1–2 minutes while the instance wakes up and reloads the model. This is a conscious trade-off for a zero-cost portfolio deployment, not a bug — see [Deployment Architecture](#-deployment-architecture) below for why.

---

## 🎯 Motivation

Deforestation of the Amazon is one of the most critical environmental challenges of our time. Satellite imagery provides continuous, large-scale coverage — but the volume of data far exceeds human review capacity. This project asks: how much can a CNN learn about land use threats from 256×256 pixel chips, using only visual signal?

The answer has direct real-world implications: automated flagging of suspicious zones for ranger follow-up, trend monitoring at scale, and early-warning systems for conservation agencies.

**Key design constraint:** every modeling decision is grounded in the EDA. No architecture choice, loss function, or augmentation strategy was made without first understanding the data.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | Planet: Understanding the Amazon from Space (Kaggle Competition) |
| **Mirror used** | nikitarom/planets-dataset |
| **Size** | 40,479 labeled training images + 61,191 test images |
| **Resolution** | 256 × 256 pixels, RGB (converted from CMYK at load time) |
| **Labels** | 17 classes, multi-label (a single image can have multiple simultaneous labels) |
| **Class imbalance** | Extreme — ratio of most to least frequent label is 383× |

### Label taxonomy

| Category | Labels |
|---|---|
| Atmospheric conditions | `clear`, `cloudy`, `haze`, `partly_cloudy` |
| Base cover | `primary` (primary rainforest) |
| Land use | `agriculture`, `cultivation`, `habitation`, `road`, `water` |
| Threats & events | `selective_logging`, `slash_burn`, `artisinal_mine`, `blow_down`, `conventional_mine`, `bare_ground`, `blooming` |

### Key EDA findings that drove every modeling decision

| Finding | Decision |
|---|---|
| All images are CMYK format | `.convert("RGB")` on every load |
| Label imbalance up to 383× | `pos_weight` in `BCEWithLogitsLoss` |
| Mean 2.87 labels per image | Decision threshold set at 0.3 |
| `primary` present in 92.7% of images | Treat as background signal, not discriminative |
| `cloudy` never co-occurs with surface labels | Physically motivated — clouds block the sensor |
| No natural "up" orientation in satellite imagery | Flips + 90/180/270° rotations are valid augmentations |
| `blow_down` ≈ `selective_logging` visually | Predicted hardest class before training — confirmed |

---

## 🔬 Methodology

The project follows a five-phase pipeline, each documented in its own notebook:

### 1. EDA & Cleaning (`01_eda.ipynb`)
- Image format audit (CMYK discovery, resolution confirmation)
- Label frequency distribution (log scale — 383× imbalance makes linear scale misleading)
- Labels-per-image distribution (complexity profiling)
- Co-occurrence heatmap (Jaccard index) revealing ecological structure: `cloudy` is mutually exclusive with all surface labels; `artisinal_mine` co-occurs with `water` at 0.88 (gold mining requires rivers); all threats co-occur with `primary` at 1.0
- Visual grid of real images per label category — including direct comparison of pristine forest vs. human-disturbed zones

### 2. Dataset & Augmentation (`02_dataset.ipynb`)
- Custom `AmazonDataset` class (PyTorch `Dataset` interface)
- Training transforms: resize → random horizontal/vertical flip → random 90° rotation → color jitter → normalize (ImageNet stats)
- Validation transforms: resize → normalize only (no stochastic augmentation)
- `DataLoader` configuration for GPU-optimized throughput

### 3. CNN Baseline (`03_cnn_baseline.ipynb`)
- Custom CNN architecture trained from scratch (2.39M parameters, 15 epochs)
- `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance
- F2-score as primary metric (recall weighted 2× over precision — false negatives are costlier)
- Global decision threshold set at 0.3

### 4. Transfer Learning (`04_finetuning.ipynb`)
- EfficientNet-B3 backbone with multi-label classification head (10.72M parameters)
- Two-phase fine-tuning: frozen backbone (7 epochs) → gradual unfreezing of layers 5–8 with discriminative learning rates 1e-4/1e-5 (10 epochs)
- Largest per-class improvement: `artisinal_mine` (+0.279 F2), driven by transfer of color/texture features from ImageNet

### 5. Interpretability (`05_gradcam.ipynb`)
- Grad-CAM applied per label using hooks on `model_eff.features[-1]` (last convolutional block)
- CAM energy analysis across 8 representative classes to characterize model attention patterns
- Misclassification analysis for threat classes, including confusion matrix between `blow_down` and `selective_logging`

---

## 📈 Results

### Overall metrics

| Model | F2-macro | F2-micro | Parameters |
|---|---|---|---|
| CNN Baseline | 0.4749 | 0.7364 | 2.39M |
| EfficientNet-B3 Phase 1 (frozen backbone) | 0.4937 | — | 10.72M |
| EfficientNet-B3 Fine-tuned | **0.5435** | **0.8130** | 10.72M |
| Absolute improvement (baseline → fine-tuned) | +0.0686 | +0.0766 | |

### Per-class F2 — EfficientNet-B3 fine-tuned

| Class | F2 | TP | FP | FN |
|---|---|---|---|---|
| `blow_down` | 0.045 | 13 | 1363 | 5 |
| `slash_burn` | 0.081 | 27 | 1512 | 4 |
| `selective_logging` | 0.119 | 55 | 2018 | 4 |
| `blooming` | 0.123 | 61 | 2143 | 6 |
| `conventional_mine` | 0.143 | 17 | 507 | 1 |
| `bare_ground` | 0.301 | 180 | 2033 | 14 |
| `artisinal_mine` | 0.412 | 70 | 500 | 0 |
| `habitation` | 0.614 | 650 | 1871 | 43 |
| `cultivation` | 0.642 | 832 | 2062 | 64 |
| `haze` | 0.652 | 501 | 1263 | 18 |
| `cloudy` | 0.780 | 410 | 547 | 8 |
| `water` | 0.789 | 1395 | 1340 | 131 |
| `road` | 0.841 | 1518 | 1052 | 96 |
| `agriculture` | 0.882 | 2301 | 859 | 170 |
| `partly_cloudy` | 0.911 | 1413 | 544 | 36 |
| `primary` | 0.946 | 7004 | 50 | 492 |
| `clear` | 0.958 | 5483 | 283 | 227 |

The performance gap between atmospheric/common labels (F2 > 0.78) and rare threat labels (F2 < 0.15) reflects the core challenge of the problem: threat classes have few training examples, weak visual signal, and similar appearance to non-threat classes.

---

## 🔍 Key Insights (Grad-CAM)

### 1. Attention patterns depend on label semantics

CAM energy (mean activation fraction across the image) varies systematically by label type:

| Label | CAM energy (μ) | Pattern |
|---|---|---|
| `cloudy` | 0.55 | Fully diffuse — the model looks everywhere, which is correct: clouds cover the whole frame |
| `artisinal_mine` | 0.29 | High energy, high variance (σ=0.09) — detects exposed reddish soil but inconsistently |
| `blow_down` | 0.21 | Moderately diffuse with high variance — erratic signal |
| `agriculture` | 0.19 | Moderate — activates on geometric field patterns |
| `road` | 0.12 | Focused — linear structures are a clear localized signal |
| `water` | 0.11 | Focused — spectral signature of water is localized and distinctive |
| `primary` | 0.10 | Focused and consistent (σ low) — dense canopy texture is reliable |

### 2. The blow_down problem is not class confusion — it is threshold sensitivity

The initial hypothesis was that `blow_down` and `selective_logging` would be confused with each other due to visual similarity. The data tells a more precise story:

- Direct confusion (GT=`blow_down`, predicted as `selective_logging`): **3 cases**
- Direct confusion (GT=`selective_logging`, predicted as `blow_down`): **24 cases**
- False positives of `blow_down` at threshold=0.3: **1363** against only 18 true positives in validation

The model is not systematically confusing these two classes. It produces weak, noisy probability scores for `blow_down` across many unrelated images, and a global threshold of 0.3 activates them all. **Per-class threshold optimization would be the highest-ROI next step — no retraining required.**

### 3. artisinal_mine — perfect recall, low precision

`artisinal_mine` achieved Recall=1.0 (FN=0) with F2=0.412. The model learned a strong visual signal — exposed reddish-brown soil, characteristic of open-pit artisanal gold mining — that transfers well from ImageNet features. The 500 FPs occur because the same signal appears in agricultural clearings and unpaved roads. This explains why it was the largest per-class improvement over the baseline (+0.279 F2): the signal exists and is transferable, but it is not exclusive to mines.

---

## 🏗️ Deployment Architecture

```
┌─────────────────────┐      HTTPS POST /predict      ┌──────────────────────┐
│   Streamlit Cloud    │ ─────────────────────────────▶│   Render (Docker)    │
│   (frontend, free)   │ ◀───────────────────────────── │   FastAPI backend    │
└─────────────────────┘         JSON response          └───────────┬──────────┘
                                                                     │
                                                          hf_hub_download()
                                                          on startup
                                                                     ▼
                                                         ┌──────────────────────┐
                                                         │  HuggingFace Hub      │
                                                         │  (model weights,      │
                                                         │  ~40MB .pth file)     │
                                                         └──────────────────────┘
```

**Why this split, and not a single service?** The model (EfficientNet-B3, PyTorch) needs a real Python runtime with `torch` installed — that rules out static hosting. The frontend (Streamlit) and the inference backend (FastAPI) have very different resource profiles, so they're deployed independently:

- **Frontend → Streamlit Community Cloud.** Free, permanent, no credit card. Deploys directly from the `app/` subfolder of this repo.
- **Backend → Render (Docker, free tier).** Free, permanent, no credit card, 512MB RAM. Deploys directly from `backend/Dockerfile`.
- **Model weights → HuggingFace Hub.** Kept out of the Docker image entirely — the backend downloads the `.pth` checkpoint at startup via `hf_hub_download()`, so the image stays lightweight and the model can be updated without rebuilding the container.

**Platforms evaluated and rejected**, and why — because "it changed since last week" is a real constraint in this space:
- ~~HuggingFace Spaces (Docker)~~ — used to offer a free CPU tier; as of mid-2026, Docker Spaces require a paid plan for new accounts.
- ~~Koyeb~~ — free tier existed at the start of this project; discontinued for new signups following Koyeb's acquisition by Mistral AI.
- ~~Railway~~ / ~~Fly.io~~ — no longer offer a genuine permanent free tier.
- ~~Google Cloud Run~~ / ~~Oracle Cloud~~ — generous free tiers, but require a credit card for identity verification, which was a hard constraint for this deployment.

---

## ⚙️ Production Optimization: Fitting a CNN into 512MB

Render's free tier caps memory at 512MB. A straightforward PyTorch deployment of EfficientNet-B3 with Grad-CAM support peaked at **644MB** — over budget. Getting it to fit (with real headroom, verified in production, not just locally) required iterating through several optimization strategies:

| Technique | Peak memory | Result |
|---|---|---|
| Baseline (float32, full PyTorch) | 644.3 MiB | ❌ Over budget |
| `torch.set_num_threads(1)` | 630.6 MiB | Marginal (~2%) — affects parallelism, not memory footprint |
| Dynamic quantization (`nn.Linear` only) | 599.7 MiB | Insufficient (~7%) — EfficientNet-B3 is >95% convolutional, and conv layers can't be quantized without breaking `autograd` |
| **`float16` (model + input tensors)** | **433.3 MiB (local)** | ✅ Halves memory while staying differentiable |
| + Remove OpenCV, replace with PIL/NumPy | *(further reduction)* | Cuts import overhead from a heavy native dependency |
| + Explicit `gc.collect()` between forward passes | *(further reduction)* | Frees memory between the two forward passes (probabilities + Grad-CAM) |

**Why not ONNX or full `int8` quantization?** Both were seriously considered and rejected for the same structural reason: **Grad-CAM requires a live autograd graph** (`.backward()` on the top predicted class). ONNX Runtime and quantized convolutional layers are inference-only — using either would have broken the interpretability feature that is core to this project's value. `float16` was the sweet spot: still a normal floating-point format, so gradients flow through it exactly as they do in `float32`, at half the memory cost.

**A local-vs-production gap that mattered:** the first deploy attempt, measured at 433MB locally via `docker stats`, still crashed on Render with `Ran out of memory (used over 512MB)`. Local Docker Desktop and Render's underlying `cgroups`-based limits don't account for memory identically — the only way to validate a fix was to reproduce the exact 512MB cap locally (`docker-compose.yml` → `deploy.resources.limits.memory: 512M`) and confirm against the real Render deployment, not trust local numbers alone.

---

## ⚠️ Limitations & Future Work

### Modeling
**Per-class threshold optimization.** A global threshold of 0.3 creates massive FP rates for rare classes. Optimizing a separate threshold per class over the validation set is the highest-ROI next step — no retraining required.

**Binary success metric is a ceiling, not a floor.** The F2-score optimization assumes equal cost across all threat labels — in practice, `slash_burn` and `artisinal_mine` may warrant higher penalties than `bare_ground`.

**Dataset is a snapshot.** The Kaggle dataset covers a specific time window. Temporal dynamics (seasonality, trend detection) are not captured.

**256×256 chips lose spatial context.** At this resolution, small artisanal mines may be indistinguishable from natural clearings without surrounding context.

**blow_down is structurally hard.** With only 101 examples and strong visual similarity to disturbed vegetation in general, this class underperforms regardless of architecture. Few-shot learning or synthetic augmentation could help.

**No multispectral bands.** The dataset provides only RGB. Real-world deforestation detection systems use near-infrared (NIR) and SWIR bands, which dramatically improve vegetation health assessment.

### Deployment
**Cold starts on the free tier.** Render's free instances sleep after 15 minutes of inactivity; waking up (container start + model download + PyTorch init) can take 1–2 minutes. A paid always-on tier would eliminate this, at a small monthly cost.

**512MB is a tight ceiling.** The current setup has limited headroom for a larger backbone (e.g. EfficientNet-B5+) without further optimization or a paid tier with more RAM.

**Single-request inference.** The backend processes one image at a time with `WEB_CONCURRENCY=1` — appropriate for a portfolio demo, not for production-scale concurrent traffic.

---

## 🛠️ How to Run

### Try the live demo (easiest)

Just visit the [Streamlit app](https://pedrosall-amazonnet.streamlit.app) — no setup required. First prediction may be slow due to backend cold start (see [Live Demo](#-live-demo) notice above).

### Run the notebooks (model training) — on Kaggle (recommended)

1. Fork this notebook on Kaggle
2. Add the dataset: `nikitarom/planets-dataset`
3. Enable GPU accelerator (Tesla T4 or better)
4. Run notebooks in order: `01 → 02 → 03 → 04 → 05`
5. Before running notebook 05: save notebook 04's output and add it as an input dataset to notebook 05 — the `.pth` checkpoint does not persist across sessions automatically

### Run the full app locally (backend + frontend)

```bash
# Clone the repo
git clone https://github.com/pedrosall/amazonnet.git
cd amazonnet

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

In a second terminal:

```bash
# Frontend
cd app
python3 -m venv frontend_venv
source frontend_venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The frontend defaults to `http://127.0.0.1:8000` for the backend URL when no `API_URL` secret is set — no extra configuration needed for local development.

### Run the backend via Docker

```bash
docker compose up --build
```

Serves the API at `http://localhost:8000` (mapped from the container's internal port 7860, matching the Render/HuggingFace Docker Space convention).

> Note: model training requires the dataset, which is not tracked in this repo (see `.gitignore`). Download it from Kaggle separately as described above.

---

## 📁 Repository Structure

```
amazonnet/
├── README.md
├── requirements.txt
├── .gitignore
├── docker-compose.yml         # Local Docker orchestration for the backend
├── notebooks/
│   ├── 01_eda.ipynb           # EDA: image audit, label distribution, co-occurrence
│   ├── 02_dataset.ipynb       # PyTorch Dataset, DataLoader, augmentation
│   ├── 03_cnn_baseline.ipynb  # Custom CNN from scratch
│   ├── 04_finetuning.ipynb    # EfficientNet-B3 transfer learning
│   └── 05_gradcam.ipynb       # Grad-CAM visual interpretability
├── backend/                   # FastAPI inference API (deployed on Render)
│   ├── main.py                 # API routes: /health, /predict
│   ├── model.py                 # Model loading, preprocessing, Grad-CAM, memory optimizations
│   ├── requirements.txt
│   └── Dockerfile
└── app/                       # Streamlit frontend (deployed on Streamlit Cloud)
    ├── streamlit_app.py
    └── requirements.txt
```

---

## 🔗 Related Projects

**CineAI** — Movie commercial success classifier using pre-production tabular data. Logistic Regression, Random Forest, and MLP with SHAP interpretability.

---

## 📜 License

This project is for educational and portfolio purposes.
Dataset © Planet Labs / Kaggle competition, distributed under their respective terms.
