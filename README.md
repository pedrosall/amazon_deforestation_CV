# 🌿 AmazonNet — Multi-Label Satellite Image Classification for Deforestation Detection

Can a deep learning model learn to detect illegal logging, slash-and-burn agriculture, and artisanal mining **directly from satellite imagery** — before any human analyst flags them?

This project builds a full computer vision pipeline — from exploratory data analysis to Grad-CAM visual explanations — to classify land use and atmospheric conditions in chips of Amazon rainforest imagery. It is a real-world multi-label classification problem where **a single image can carry up to 9 simultaneous labels**, and where missing a threat (false negative) is far more costly than a false alarm.

---

## 🎯 Motivation

Deforestation of the Amazon is one of the most critical environmental challenges of our time. Satellite imagery provides continuous, large-scale coverage — but the volume of data far exceeds human review capacity. This project asks: **how much can a CNN learn about land use threats from 256×256 pixel chips**, using only visual signal?

The answer has direct real-world implications: automated flagging of suspicious zones for ranger follow-up, trend monitoring at scale, and early-warning systems for conservation agencies.

**Key design constraint:** every modeling decision is grounded in the EDA. No architecture choice, loss function, or augmentation strategy was made without first understanding the data.

---

## 📊 Dataset

- **Source:** [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) (Kaggle Competition)
- **Mirror used:** [nikitarom/planets-dataset](https://www.kaggle.com/datasets/nikitarom/planets-dataset)
- **Size:** 40,479 labeled training images + 61,191 test images
- **Resolution:** 256 × 256 pixels, RGB (converted from CMYK at load time)
- **Labels:** 17 classes, multi-label (a single image can have multiple simultaneous labels)
- **Class imbalance:** extreme — ratio of most to least frequent label is **383×**

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
| Mean 2.87 labels per image | Decision threshold search between 0.2–0.4 |
| `primary` present in 92.7% of images | Treat as background signal, not discriminative |
| `cloudy` never co-occurs with surface labels | Physically motivated — clouds block the sensor |
| No natural "up" orientation in satellite imagery | Flips + 90/180/270° rotations are valid augmentations |
| `blow_down` ≈ `selective_logging` visually | Predicted hardest class before training |

---

## 🔬 Methodology

The project follows a five-phase pipeline, each documented in its own notebook:

### 1. EDA & Cleaning (`01_eda.ipynb`)
- Image format audit (CMYK discovery, resolution confirmation)
- Label frequency distribution (log scale — 383× imbalance makes linear scale misleading)
- Labels-per-image distribution (complexity profiling)
- **Co-occurrence heatmap** (Jaccard index) revealing ecological structure: `cloudy` is mutually exclusive with all surface labels; `artisinal_mine` co-occurs with `water` at 0.88 (gold mining requires rivers); all threats co-occur with `primary` at 1.0
- Visual grid of real images per label category — including direct comparison of pristine forest vs. human-disturbed zones

### 2. Dataset & Augmentation (`02_dataset.ipynb`)
- Custom `AmazonDataset` class (PyTorch `Dataset` interface)
- Training transforms: resize → random horizontal/vertical flip → random 90° rotation → color jitter → normalize (ImageNet stats)
- Validation transforms: resize → normalize only (no stochastic augmentation)
- `DataLoader` configuration for GPU-optimized throughput

### 3. CNN Baseline (`03_cnn_baseline.ipynb`)
- Custom CNN architecture trained from scratch
- `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance
- **F2-score** as primary metric (recall weighted 2× over precision — false negatives are costlier)
- Threshold optimization over validation set

### 4. Transfer Learning (`04_finetuning.ipynb`)
- EfficientNet-B3 backbone with multi-label classification head
- Fine-tuning strategy: frozen backbone → gradual unfreezing
- Compared against baseline on identical test split

### 5. Interpretability (`05_gradcam.ipynb`)
- **Grad-CAM** applied per label — the visual equivalent of SHAP for CNNs
- Answers: *which pixels drove the model to predict `selective_logging`?*
- High-confidence misclassification analysis (especially `blow_down` vs `selective_logging`)

---

## 📈 Results

*To be updated after training.*

| Model | F2-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| CNN Baseline | — | — | — | — |
| EfficientNet-B3 (fine-tuned) | — | — | — | — |

---

## 🔍 Key Insights (Grad-CAM)

*To be updated after interpretability analysis.*

---

## 🛠️ How to Run

### On Kaggle (recommended)

1. Fork this notebook on Kaggle
2. Add the dataset: [nikitarom/planets-dataset](https://www.kaggle.com/datasets/nikitarom/planets-dataset)
3. Enable GPU accelerator (Tesla T4 or better)
4. Run notebooks in order: `01` → `02` → `03` → `04` → `05`

### Locally

```bash
# Clone the repo
git clone https://github.com/pedrosall/amazonnet.git
cd amazonnet

# Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download nikitarom/planets-dataset
unzip planets-dataset.zip -d data/raw/

# Run notebooks in order
jupyter lab notebooks/
```

> **Note:** dataset files are not tracked in this repo (see `.gitignore`). Download them separately as described above.

---

## 📁 Repository Structure

```
amazonnet/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── 01_eda.ipynb              # EDA: image audit, label distribution, co-occurrence
    ├── 02_dataset.ipynb          # PyTorch Dataset, DataLoader, augmentation
    ├── 03_cnn_baseline.ipynb     # Custom CNN from scratch
    ├── 04_finetuning.ipynb       # EfficientNet-B3 transfer learning
    └── 05_gradcam.ipynb          # Grad-CAM visual interpretability
```

---

## ⚠️ Limitations & Future Work

- **Binary success metric is a ceiling, not a floor.** The F2-score optimization assumes equal cost across all threat labels — in practice, `slash_burn` and `artisinal_mine` may warrant higher penalties than `bare_ground`.
- **Dataset is a snapshot.** The Kaggle dataset covers a specific time window. Temporal dynamics (seasonality, trend detection) are not captured.
- **256×256 chips lose spatial context.** At this resolution, small artisanal mines may be indistinguishable from natural clearings without surrounding context.
- **`blow_down` is structurally hard.** With only 101 examples and strong visual similarity to `selective_logging`, this class is expected to underperform regardless of architecture. Few-shot learning or synthetic augmentation could help.
- **No multispectral bands.** The dataset provides only RGB. Real-world deforestation detection systems use near-infrared (NIR) and SWIR bands, which dramatically improve vegetation health assessment.

---

## 🔗 Related Projects

- [**CineAI**](https://github.com/pedrosall/cineai) — Movie commercial success classifier using pre-production tabular data. Logistic Regression, Random Forest, and MLP with SHAP interpretability.

---

## 📜 License

This project is for educational and portfolio purposes.  
Dataset © Planet Labs / Kaggle competition, distributed under their respective terms.
