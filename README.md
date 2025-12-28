# BrainScanAI: Semi-Supervised Learning for Brain Tumor Detection

> **A complete machine learning pipeline for automated brain tumor detection using limited labeled data**

---

## 📋 Project Overview

This project explores the feasibility of automating brain tumor detection for **CurelyticsIA**, an e-health startup, using:
- **1,500 total brain MRI images** (512×512 JPEG)
- **100 expert-labeled images** (normal/cancer)
- **1,400 unlabeled images**
- **€300 current labeling budget** (€3/image)

### Business Objective
Evaluate scaling to **4 million images** with a **€5,000 budget**.

---

## 📚 **Documentation Hub**

**→ All project documentation is now centralized in [`docs_generated/`](docs_generated/)**

**Quick Links:**
- ⭐ **[Quick Start Guide](docs_generated/QUICKSTART.md)** - Get running in 5 minutes
- 📖 **[Complete Documentation Index](docs_generated/00_INDEX.md)** - Master navigation
- 🏗️ **[System Overview](docs_generated/01_SYSTEM_OVERVIEW.md)** - Architecture & design
- 🔧 **[How to Run](docs_generated/02_HOW_TO_RUN.md)** - Installation & execution
- 📊 **[Latest Audit Report](docs_generated/POST_COMMIT_AUDIT_2025-12-28.md)** - Production readiness: 9.2/10

**Why docs_generated/?**
- ✅ Single source of truth for all documentation
- ✅ Code-first approach (no documentation drift)
- ✅ Comprehensive (13 files covering all aspects)
- ✅ Maintained and up-to-date

---

## 📁 Project Structure

```
Medical Images - Cancer/
│
├── data/                                    # Dataset directory
│   ├── labelled/
│   │   ├── cancer/                         # Cancer brain scans
│   │   └── normal/                         # Normal brain scans
│   └── unlabelled/                         # Unlabeled brain scans
│
├── 1_feature_extraction.ipynb              # Notebook 1: Data exploration & feature extraction
├── 2_unsupervised_analysis.ipynb           # Notebook 2: Clustering & weak labeling
├── 3_semi_supervised_learning.ipynb        # Notebook 3: Model training & scaling analysis
│
├── features/                                # Generated features (created by notebooks)
│   ├── resnet50_features.npy
│   ├── labels.npy
│   ├── metadata.csv
│   ├── weak_labels.csv
│   ├── features_pca_50.npy
│   ├── features_tsne.npy
│   └── clustering_summary.json
│
├── model_supervised_best.pth               # Trained models (created by Notebook 3)
├── model_semisup_best.pth
├── final_results.json
├── model_comparison.csv
│
└── README.md                                # This file
```

---

## 🚀 Getting Started

### Prerequisites

**Python 3.8+** with the following libraries:

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pillow tqdm
pip install plotly jupyter
```

**GPU Recommended** (but not required):
- Training will be significantly faster with CUDA-enabled GPU
- CPU training is possible but may take 2-3x longer

### Quick Start

1. **Ensure your data is in the correct directory structure** (as shown above)

2. **Run the notebooks in order:**
   ```bash
   jupyter notebook 1_feature_extraction.ipynb
   ```

3. **Follow the sequence:**
   - Notebook 1 → Notebook 2 → Notebook 3
   - Each notebook depends on outputs from the previous one

---

## 📓 Notebook Descriptions

### Notebook 1: Feature Extraction
**File**: `1_feature_extraction.ipynb`

**What it does:**
- ✅ Loads and explores 1,500 brain MRI images
- ✅ Visualizes samples from labeled and unlabeled datasets
- ✅ Preprocesses images (resize, normalize)
- ✅ Extracts 2048-dimensional features using pretrained ResNet50
- ✅ Saves features for downstream analysis

**Runtime**: ~10-20 minutes (depending on hardware)

**Outputs**:
- `features/resnet50_features.npy` (2048D features)
- `features/labels.npy`
- `features/metadata.csv`

**Key Visualizations**:
- Sample brain scans (cancer vs normal vs unlabeled)
- Feature distribution statistics

---

### Notebook 2: Unsupervised Analysis
**File**: `2_unsupervised_analysis.ipynb`

**What it does:**
- ✅ Loads extracted features from Notebook 1
- ✅ Applies PCA dimensionality reduction (2048D → 50D)
- ✅ Visualizes data with t-SNE (2D embeddings)
- ✅ Performs K-Means clustering (K=2 clusters)
- ✅ Generates weak labels for 1,400 unlabeled images
- ✅ Evaluates clustering quality with ARI score
- ✅ Tests DBSCAN as alternative clustering method

**Runtime**: ~5-15 minutes

**Outputs**:
- `features/weak_labels.csv` (Weak labels for all images)
- `features/features_pca_50.npy`
- `features/features_tsne.npy`
- `features/clustering_summary.json`

**Key Visualizations**:
- t-SNE scatter plots (true labels vs clusters)
- PCA explained variance
- Confusion matrix (clusters vs true labels)
- Elbow curve and Silhouette analysis

**Expected ARI Score**: 0.1 - 0.4 (typical for medical imaging)
- Perfect clustering (ARI=1.0) is unrealistic
- Even modest ARI indicates useful weak labels

---

### Notebook 3: Semi-Supervised Learning & Scaling
**File**: `3_semi_supervised_learning.ipynb`

**What it does:**
- ✅ Trains **Approach 1**: Fully supervised (70 labeled images only)
- ✅ Trains **Approach 2**: Semi-supervised (1,400 weak + 70 strong labels)
  - Phase 1: Pre-train on weak labels
  - Phase 2: Fine-tune on strong labels
- ✅ Compares both approaches with comprehensive metrics
- ✅ Evaluates with F-beta score (β=2, emphasizing Recall)
- ✅ Analyzes scaling feasibility (€300 → €5,000 → 4M images)
- ✅ Provides business recommendations

**Runtime**: ~30-60 minutes (GPU), ~2-4 hours (CPU)

**Outputs**:
- `model_supervised_best.pth` (Best fully supervised model)
- `model_semisup_best.pth` (Best semi-supervised model)
- `final_results.json` (Performance metrics)
- `model_comparison.csv` (Side-by-side comparison)

**Key Visualizations**:
- Training curves (loss & accuracy)
- Confusion matrices
- ROC curves
- Metric comparison bar charts
- Scaling scenarios visualization

**Target**: >90% accuracy on test set

---

## 🎯 Key Findings & Results

### Model Performance
*(Will be populated after running notebooks)*

**Fully Supervised** (Baseline):
- Training data: 70 expert-labeled images
- Accuracy: TBD
- F2-Score: TBD

**Semi-Supervised** (Proposed):
- Training data: 1,400 weak + 70 strong labels
- Accuracy: TBD
- F2-Score: TBD

### Scaling Feasibility

**Current Situation:**
- Budget: €300
- Images labeled: 100
- Cost: €3/image

**Proposed Scaling:**
- Budget: €5,000
- Target: 4,000,000 images

**Verdict**: ❌ **NOT FEASIBLE** with manual labeling at current rates

**Recommended Hybrid Strategy**:
1. **Expert Labels**: 5,000 images @ €3/image = €15,000 (€3,000 from budget)
2. **Crowdsourcing**: ~6,600 images @ €0.30/image = €2,000
3. **Self-Supervised**: 4M unlabeled images (€0/image)
4. **Active Learning**: Optimize labeling efficiency (50-70% savings)

---

## 📊 Evaluation Metrics

### Primary Metrics
- **F-beta Score (β=2)**: Emphasizes Recall over Precision
  - Why? False Negatives (missing cancer) are more dangerous than False Positives
  - β=2 weighs Recall 4× more than Precision

### Secondary Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Clustering Quality
- **ARI (Adjusted Rand Index)**: Measures clustering-label agreement
  - Range: [-1, 1]
  - Expected: 0.1 - 0.4 for medical imaging

---

## 💡 Technical Approach

### Transfer Learning
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Feature Extraction**: 2048D embeddings from penultimate layer
- **Fine-tuning**: Binary classification head (normal/cancer)

### Semi-Supervised Learning Strategy
```
Phase 1: Pre-training
┌─────────────────────────────┐
│ ResNet50 (ImageNet)         │
│          ↓                  │
│ Train on 1,400 weak labels  │
│          ↓                  │
│ Learn general patterns      │
└─────────────────────────────┘

Phase 2: Fine-tuning
┌─────────────────────────────┐
│ Pre-trained model           │
│          ↓                  │
│ Fine-tune on 70 strong      │
│          ↓                  │
│ Refine decision boundary    │
└─────────────────────────────┘
```

### Weak Labeling
- **Method**: K-Means clustering (K=2)
- **Input**: 2048D ResNet50 features
- **Output**: Binary pseudo-labels
- **Quality**: ~60-80% agreement with expert labels

---

## ⚠️ Important Notes

### Critical Rules
1. **NEVER mix weak and strong labels** in the same training batch
2. **Test set must be held out** from all training phases
3. **Weak labels are noisy** - only use for pre-training

### Data Quality
- All images are 512×512 JPEG (already standardized)
- Both grayscale and RGB images present (converted to RGB in pipeline)
- No missing or corrupted images detected

### Reproducibility
- Random seed: 42 (set in all notebooks)
- GPU determinism enabled (when using CUDA)
- All hyperparameters documented

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (GPU)**
```python
# Reduce batch size in Notebook 3
BATCH_SIZE = 8  # Instead of 16
```

**2. Slow Training (CPU)**
- Expected - consider using Google Colab or AWS with GPU
- Reduce number of epochs temporarily for testing

**3. Module Not Found**
```bash
pip install <missing-module>
```

**4. Data Not Found**
- Ensure `data/` directory is in the same folder as notebooks
- Check file paths in Notebook 1

---

## 📈 Expected Outcomes

After running all notebooks, you should have:

✅ **Understanding of dataset**:
- Visual patterns distinguishing cancer vs normal
- Feature distributions
- Class balance

✅ **Weak labeling pipeline**:
- Automated pseudo-labels for unlabeled data
- Quality metrics (ARI score)

✅ **Trained models**:
- Fully supervised baseline
- Semi-supervised approach
- Performance comparison

✅ **Business insights**:
- Scaling feasibility analysis
- Cost-benefit trade-offs
- Recommended strategy

✅ **Deliverables for CurelyticsIA**:
- Technical proof-of-concept
- Scaling roadmap
- Budget recommendations

---

## 🎓 Learning Objectives

This project demonstrates:

1. **Transfer Learning**: Leveraging pretrained models for medical imaging
2. **Unsupervised Learning**: Clustering for weak label generation
3. **Semi-Supervised Learning**: Combining labeled and unlabeled data
4. **Model Evaluation**: Comprehensive metrics for medical AI
5. **Business Analysis**: Translating technical results to ROI

---

## 📚 References

### Papers
- He et al. (2016). "Deep Residual Learning for Image Recognition"
- Tarvainen & Valpola (2017). "Mean teachers are better role models"
- Chapelle et al. (2006). "Semi-Supervised Learning"

### Frameworks
- PyTorch: https://pytorch.org/
- scikit-learn: https://scikit-learn.org/
- torchvision: https://pytorch.org/vision/

---

## 👤 Author

**Junior Data Scientist**
CurelyticsIA - AI-Powered Medical Imaging Division
Project: BrainScanAI
Date: 2025

---

## 📝 License

This project is for academic and research purposes at CurelyticsIA.

---

## 🙏 Acknowledgments

- Brain MRI dataset providers
- PyTorch and scikit-learn communities
- CurelyticsIA research team

---

## 🚦 Next Steps

After completing this project:

1. **Deploy model** as REST API for clinical testing
2. **Collect real-world feedback** from radiologists
3. **Implement active learning** to optimize labeling
4. **Expand to other tumor types** (glioma, meningioma, etc.)
5. **Scale to 4M images** using hybrid strategy

---

**Status**: ✅ All notebooks ready to run
**Estimated Total Runtime**: 1-2 hours (GPU) | 4-6 hours (CPU)

**Good luck with your analysis! 🧠🔬**
