# How to Run

**Source:** Derived from code (pyproject.toml, environment analysis)
**Last Verified:** 2025-12-28

---

## Prerequisites (Code-Verified)

### Required Software
```
Python: 3.11.9 (verified via runtime)
Poetry: >=2.0.0 (build system requirement)
```

### Operating System
- **Tested:** Windows 10/11 (based on file paths in notebooks)
- **Expected:** Cross-platform (pure Python dependencies)

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/shah-data-scientist/Medical-Images---Cancer.git
cd Medical-Images---Cancer
```

### 2. Install Dependencies (via Poetry)

**Dependencies (from pyproject.toml):**
```toml
[project]
requires-python = ">=3.11,<3.13"

dependencies = [
    "torch (>=2.9.1,<3.0.0)",
    "torchvision (>=0.24.1,<0.25.0)",
    "numpy (>=2.4.0,<3.0.0)",
    "pandas (>=2.3.3,<3.0.0)",
    "matplotlib (>=3.10.8,<4.0.0)",
    "seaborn (>=0.13.2,<0.14.0)",
    "plotly (>=6.5.0,<7.0.0)",
    "scikit-learn (>=1.8.0,<2.0.0)",
    "pillow (>=12.0.0,<13.0.0)",
    "tqdm (>=4.67.1,<5.0.0)",
    "jupyter (>=1.1.1,<2.0.0)",
    "ipython (>=9.8.0,<10.0.0)",
    "notebook (>=7.5.1,<8.0.0)",
    "ipykernel (>=7.1.0,<8.0.0)",
    "statsmodels (>=0.14.6,<0.15.0)",
    "mlflow (>=3.8.0,<4.0.0)"
]
```

**Install Command:**
```bash
poetry install
```

This creates a virtual environment (`.venv/`) with all dependencies.

---

## Running the System

### Execution Order (Based on Code Dependencies)

The notebooks must be run in sequence:

#### Step 1: Feature Extraction
```bash
poetry run jupyter notebook 1_feature_extraction.ipynb
```

**What it does:**
- Loads images from `data/labelled/` and `data/unlabelled/`
- Extracts ResNet50 features (2048D)
- Applies PCA (2048D → 50D)
- Saves features to `features/` directory

**Requirements:**
- `data/` directory must exist with images
- Images must be in JPEG format
- Directory structure: `data/labelled/{no,yes}/` and `data/unlabelled/{no,yes}/`

**Outputs:**
- `features/resnet50_features.npy`
- `features/features_pca_50.npy`
- `features/labels.npy`
- `features/metadata.csv`

**Expected Runtime:** ~10-15 minutes (for 2,824 images)

---

#### Step 2: Unsupervised Analysis
```bash
poetry run jupyter notebook 2_unsupervised_analysis.ipynb
```

**What it does:**
- Loads PCA features from Step 1
- Performs K-means clustering (k=2)
- Generates weak labels
- Filters by confidence threshold
- Creates visualizations

**Requirements:**
- Must have completed Step 1
- `features/features_pca_50.npy` must exist

**Outputs:**
- `features/weak_labels.csv`
- `features/weak_labels_filtered.csv`
- `features/weak_labels_high_confidence.csv`
- `features/clustering_summary.json`

**Expected Runtime:** ~5-10 minutes

---

#### Step 3: Semi-Supervised Learning
```bash
poetry run jupyter notebook 3_semi_supervised_learning.ipynb
```

**What it does:**
- Loads features and weak labels from Steps 1-2
- Splits data (80/20: 80 training pool, 20 test)
- Trains 3 scenarios with 5-fold cross-validation
- Evaluates performance
- Generates validation results

**Requirements:**
- Must have completed Steps 1-2
- `features/features_pca_50.npy` must exist
- `features/weak_labels.csv` must exist
- Requires 100 manually labeled samples in dataset

**Outputs:**
- `detailed_cv_results.json`
- `scenario_comparison.csv`
- In-notebook visualizations

**Expected Runtime:** ~45-60 minutes (3 scenarios × 5 folds × 50 epochs)

---

### Standalone Validation Analysis

#### Option A: Run Individual Functions
```bash
poetry run python
>>> from advanced_validation_analysis import analyze_feature_importance, analyze_tsne_separation, test_noise_robustness
>>> # Use functions programmatically
```

#### Option B: Run Complete Validation
```bash
poetry run python run_validation_analysis.py
```

**What it does:**
- Trains a model on full training pool (80 samples)
- Runs all 3 validation analyses
- Generates visualizations
- Saves results

**Requirements:**
- Must have completed Steps 1-2
- Features must be extracted

**Outputs:**
- `feature_importance_analysis.png`
- `tsne_visualization.png`
- `noise_robustness_test.png`
- `validation_analysis_results.json`

**Expected Runtime:** ~10-15 minutes

---

## Configuration (Code-Observed)

### Environment Variables
**None required.** All configuration is hardcoded in notebooks.

### Configurable Parameters (Found in Code)

#### 1_feature_extraction.ipynb
```python
# Configurable (but hardcoded)
IMAGE_SIZE = (224, 224)  # ResNet50 input size
PCA_COMPONENTS = 50      # Dimensionality after PCA
RANDOM_STATE = 42        # Reproducibility seed
```

#### 2_unsupervised_analysis.ipynb
```python
# K-means configuration
N_CLUSTERS = 2           # Binary classification
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.7  # For weak label filtering
```

#### 3_semi_supervised_learning.ipynb
```python
# Data split
TRAIN_SIZE = 0.80        # 80% for training pool
TEST_SIZE = 0.20         # 20% for final test
RANDOM_STATE = 42

# Cross-validation
N_FOLDS = 5
STRATIFIED = True

# Model architecture
INPUT_SIZE = 50          # PCA components
HIDDEN_SIZE = 64         # Hidden layer neurons
DROPOUT_RATE = 0.70      # Dropout probability

# Training
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.1
```

**Note:** These are **hardcoded constants** in the notebooks, not configurable via environment variables or config files.

---

## Data Requirements

### Expected Directory Structure
```
Medical-Images---Cancer/
├── data/
│   ├── labelled/
│   │   ├── no/         # Normal brain scans (JPEG)
│   │   └── yes/        # Cancer brain scans (JPEG)
│   └── unlabelled/
│       ├── no/         # Unlabeled normal scans
│       └── yes/        # Unlabeled cancer scans
├── features/           # Generated by Step 1
├── 1_feature_extraction.ipynb
├── 2_unsupervised_analysis.ipynb
├── 3_semi_supervised_learning.ipynb
├── advanced_validation_analysis.py
└── run_validation_analysis.py
```

### Data Verification (Pre-Flight Check)
```bash
# Check data directory exists
ls data/labelled/no data/labelled/yes

# Count images (should be 100 labeled)
find data/labelled -name "*.jpg" -o -name "*.jpeg" | wc -l

# Count unlabeled images (should be 2724)
find data/unlabelled -name "*.jpg" -o -name "*.jpeg" | wc -l
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError
```
Error: ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Ensure you're using Poetry environment
poetry install
poetry shell
jupyter notebook
```

### Issue 2: FileNotFoundError (features/)
```
Error: FileNotFoundError: [Errno 2] No such file or directory: 'features/features_pca_50.npy'
```

**Solution:** Run notebooks in sequence. Step 2 and 3 require Step 1 outputs.
```bash
# Run Step 1 first
poetry run jupyter notebook 1_feature_extraction.ipynb
# Execute all cells
```

### Issue 3: BatchNorm Error
```
Error: ValueError: Expected more than 1 value per channel when training
```

**Solution:** Already fixed in code. Notebooks use `drop_last=True` in DataLoader:
```python
DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
```

### Issue 4: Insufficient Memory
```
Error: RuntimeError: Out of memory
```

**Solution:** Reduce batch size in notebook:
```python
# Change from:
BATCH_SIZE = 16
# To:
BATCH_SIZE = 8
```

---

## Verification

### Test Installation
```bash
# Verify Python version
poetry run python --version
# Expected: Python 3.11.9

# Verify key packages
poetry run python -c "import torch; print('PyTorch:', torch.__version__)"
# Expected: PyTorch: 2.9.1+cpu

poetry run python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
# Expected: scikit-learn: 1.8.0
```

### Test Feature Extraction
```bash
# Quick test: extract features from one image
poetry run python -c "
from torchvision import models, transforms
from PIL import Image
import torch

model = models.resnet50(pretrained=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load sample image (replace with actual path)
# img = Image.open('data/labelled/no/sample.jpg')
# features = model(transform(img).unsqueeze(0))
print('ResNet50 loaded successfully')
"
```

---

## Performance Benchmarks (Observed)

**Hardware:** CPU (no GPU)
**Environment:** Windows 10/11, Python 3.11.9

| Task | Runtime | Output Size |
|------|---------|-------------|
| Feature extraction (2,824 images) | ~10-15 min | ~12 MB |
| Unsupervised analysis | ~5-10 min | ~300 KB |
| Semi-supervised training (Scenario A) | ~2 min | In-memory |
| Semi-supervised training (Scenario B/C) | ~47 min | In-memory |
| Validation analysis | ~10-15 min | ~1 MB |

**Total Pipeline Runtime:** ~75-90 minutes

---

## Next Steps After Running

After successful execution, you should have:

1. ✅ Trained models (in-memory, not persisted)
2. ✅ Performance metrics (`detailed_cv_results.json`)
3. ✅ Validation results (`validation_analysis_results.json`)
4. ✅ Visualizations (3 PNG files)

See `03_MODULE_API.md` for detailed function documentation.

---

**Source:** This documentation is derived from:
- `pyproject.toml` (dependencies)
- Runtime environment inspection
- Notebook code analysis
- Configuration constants in code
