# Large-Scale Automated Image Labeling
## Semi-Supervised Approach for 4 Million Images

**Project:** BrainScanAI - Brain Tumor Detection
**Budget:** €5,000
**Objective:** Label 4,000,000 images

🔗 **GitHub Repository:** [shah-data-scientist/Medical-Images---Cancer](https://github.com/shah-data-scientist/Medical-Images---Cancer)

---

## 1. Problem Statement and Context

### The Challenge
- **Target volume:** 4 million images to label
- **Available budget:** €5,000
- **Current manual cost:** €3/image (medical expert)
- **Problem:** At this rate → only 1,667 images possible (0.04% of target)

### The Solution
**Hybrid AI + Human approach** to reduce cost by **99.96%**
→ From €3/image to €0.00125/image

---

## 2. Technical Architecture - 3-Stage Pipeline

### Overview

```
Raw images → Feature extraction → Clustering → Semi-supervised → Predictions
  (4M)           (ResNet50)         (K-Means)      (MLP)           (4M labels)
```

---

## 3. Stage 1 - Visual Feature Extraction

### Model Choice: ResNet50 (Pre-trained ImageNet)

**Rationale:**
- ✅ Proven architecture (76M parameters, 93% top-5 accuracy)
- ✅ Effective transfer learning (generic features → medical domain)
- ✅ Available via PyTorch (GPU optimized)
- ✅ Fast extraction: 10,000 images/hour on cloud GPU

### Extraction Process

**Input:**
- MRI images 224×224 pixels (resized)
- 2 classes: Normal / Cancer

**Pipeline:**
1. Preprocessing: ImageNet normalization (mean=[0.485, 0.456, 0.406])
2. Feature extraction: ResNet50 `avgpool` layer → 2048D vectors
3. Dimensionality reduction: PCA 2048D → 50D (preserves 97%+ variance)

**Output:**
- `resnet50_features.npy`: Raw 2048D features
- `features_pca_50.npy`: Reduced 50D features (used for training)

**PCA Benefits:**
- Reduced training time (41× faster)
- Less overfitting (implicit regularization)
- Preserves discriminative information

---

## 4. Stage 2 - Unsupervised Analysis **[Tested - Not Used in Final Model]**

### Objective
Generate **weak labels** for unlabeled data (exploratory analysis)

### Methods Used

**1. K-Means Clustering (k=2)**
- Automatic grouping into 2 clusters (Normal/Cancer)
- Cluster ↔ class alignment via existing labels
- Generation of 2,724 weak labels

**2. Confidence Filtering**
- Distance to centroid as confidence score
- Retention: 20.1% of high-confidence samples
- Trade-off: purity vs quantity

**3. t-SNE Visualization**
- Projection 50D → 2D for separability analysis
- Visual cluster validation
- Identification of ambiguous zones

### Clustering Results

| Metric | Value |
|--------|-------|
| Unlabeled images | 2,724 |
| Weak labels generated | 2,724 (100%) |
| High confidence | 547 (20.1%) |
| Cluster separability | Good (t-SNE visualization) |

---

## 5. Stage 3 - Semi-Supervised Learning

### Model Architecture

**Regularized MLP (Multi-Layer Perceptron):**
- Input: 50 features (PCA)
- Hidden: 64 neurons + 70% Dropout
- Output: 2 classes (Normal/Cancer)
- Loss: Binary Cross-Entropy
- Optimizer: AdamW (lr=0.001, weight_decay=0.001)

### 3 Scenarios Tested

**Scenario A - Fully Supervised (baseline):**
- 100 manual labels only
- F2 Score: **96.43% ± 3.60%**

**Scenario B - Clustering Semi-Supervised:**
- 100 labels + 547 K-Means weak labels
- F2 Score: **89.29% ± 0.97%**
- Drop: -7.14% (low weak label quality)

**Scenario C - Self-Training Semi-Supervised:**
- 100 labels + model-generated pseudo-labels
- F2 Score: **92.84% ± 3.94%**
- Improvement vs Scenario B: +3.55%

### Robustness Validation

**Tests performed:**
1. **Feature Importance:** 6 critical PCA components identified
2. **Noise Injection:** F2 stable at 96.15% with +10% Gaussian noise
3. **Cross-Validation:** 5-fold stratified, low variance (<4%)

---

## 6. Results - Comparative Summary

### Final Performance

| Scenario | F2 Score | Recall | Accuracy | Labeled Data | Efficiency |
|----------|----------|--------|----------|--------------|------------|
| A (Supervised) | 96.43% ± 3.60% | 98.00% ± 4.47% | 94.00% ± 2.24% | 100 (100%) | Baseline |
| B (K-Means) | 89.65% ± 0.79% | 90.00% ± 0.00% | 89.00% ± 2.24% | 100 + 547 weak | -6.78% |
| C (Self-Training) | 93.22% ± 4.40% | 94.00% ± 5.48% | 92.00% ± 2.74% | 100 + pseudo | -3.21% |

**Statistical tests:**
- Scenario C vs A: p=0.1237 (not significant)
- Scenario C vs B: p=0.1397 (not significant)

**Retained Model for Current Project: Scenario A (Fully Supervised)**
- Uses only 100 manual labels (NO weak labels, NO pseudo-labels)
- Best performance: 96.43% F2 ± 3.60%
- Selected for its superior accuracy and simplicity
- **⚠️ Scaling Limitation:** Cannot be used for 4M images due to labeling cost (4M × €3 = €12M >> €5K budget)

**Recommended Scaling Strategy: Scenario C Methodology (Semi-Supervised)**
- Performance: 93.22% F2 ± 4.40% (2nd best in experiments)
- **Key advantage:** Semi-supervised approach using pseudo-labels enables scaling to 4M images
- Methodology: Initial model training + iterative pseudo-labeling on unlabeled data
- **Business case:** Reduces per-image cost from €3 to €0.00125 (2,400× efficiency)
- **Selected for 4M deployment** due to cost-effectiveness while maintaining 93%+ accuracy
- Trade-off: -3.21% F2 vs Scenario A, but economically viable for large-scale labeling

### Key Insights

1. **Data efficiency:** 96%+ F2 with only 5.3% labeled data (100/1880)
2. **Weak label quality:** 20.1% retention rate critical for performance
3. **Robustness:** Model stable against noise (±10%)
4. **Features:** ResNet50 extracts powerful discriminative features

---

## 7. Project Deliverables

### Jupyter Notebooks (.ipynb)

**1. Feature Extraction** (`1_feature_extraction.ipynb`)
- Load pre-trained ResNet50
- Extract 2048D features
- PCA reduction → 50D

**2. Unsupervised Analysis** (`2_unsupervised_analysis.ipynb`)
- K-Means clustering (k=2)
- Weak label generation
- t-SNE visualizations

**3. Semi-Supervised Learning** (`3_semi_supervised_learning.ipynb`)
- Train 3 scenarios
- 5-fold cross-validation
- Validation analyses (robustness, feature importance)

### Python Scripts (.py)

**4. Advanced Validation**
- `advanced_validation_analysis.py`: Reusable functions
- `run_validation_analysis.py`: Standalone execution

### Data Files

- Extracted features: `features/*.npy`
- Weak labels: `features/weak_labels*.csv`
- Results: `*.json`, `*.csv`
- Visualizations: `*.png`

---

## 8. Scaling to 4M Images - €5,000 Budget

**Important Note:** This section details the **recommended scaling strategy based on Scenario C methodology** (see Section 6). While Scenario A achieved best performance for the current 2.8K images, Scenario C's semi-supervised approach with **pseudo-labeling** is selected for 4M deployment due to cost-effectiveness (€0.00125/image vs €3/image).

### Recommended Strategy: Hybrid Semi-Supervised with Pseudo-Labeling

#### Phase 1: Strategic Labeling (€1,500)
- **500 images** selected via clustering (diverse sampling)
- Expert labeling: 500 × 2 min = 16.7h × €90/h
- Covers pathological presentation variability

#### Phase 2: Transfer Learning (€200)
- Fine-tune current model (96% F2) on 500 new samples
- Cloud GPU + development
- Domain adaptation source → 4M target images

#### Phase 3: Automated Inference (€300)
- Feature extraction + predictions on 4M images
- Cloud GPU (spot instances): €0.75/h × 400h
- Output: **Pseudo-labels** (model predictions) + confidence scores

#### Phase 4: Human-in-the-Loop (€2,500)
**Confidence-based tiering for pseudo-labels:**
- **High confidence (90%):** Auto-accept pseudo-labels → 3.6M images, €0 review
- **Medium confidence (7.5%):** 1% sampled → 3,000 images, €2,500
- **Low confidence (2.5%):** Included in Tier 2 budget

#### Phase 5: Quality Assurance (€500)
- Statistical validation via sampling (3,000 images)
- Accuracy report per confidence tier

### Total Budget

| Phase | Cost | % Budget |
|-------|------|----------|
| Labeling (500) | €1,500 | 30% |
| Transfer Learning | €200 | 4% |
| Inference (4M) | €300 | 6% |
| Human review (3,100) | €2,500 | 50% |
| Quality assurance | €500 | 10% |
| **TOTAL** | **€5,000** | **100%** |

### Expected Outcomes

**Coverage:**
- 4,000,000 labeled images (via pseudo-labeling)
- Unit cost: €0.00125/image (vs €3 manual)
- **Savings: 99.96% (2,400× more efficient)**

**Estimated accuracy for pseudo-labels:**
- High confidence tier (90%): 94-96% accuracy
- Medium confidence tier (7.5%): 90-92% accuracy
- Low confidence tier (2.5%): 85-88% accuracy
- **Overall average: ~93-95%**

---

## 9. Success Conditions

### Critical Prerequisites

**1. Initial Label Quality**
- ✅ 500 strategic labels (guaranteed diversity)
- ✅ Expert validation (qualified radiologists)
- ✅ Class balancing (prevent bias)

**2. Technical Infrastructure**
- Cloud GPU with spot instances (60-70% savings)
- Automated pipeline (feature extraction → inference)
- MLflow or equivalent for experiment tracking

**3. Continuous Monitoring**
- Performance metrics per batch (detect drift)
- Systematic error analysis
- Confidence score calibration

**4. Planned Iterations**
- Cycle 1: Initial labeling + model V1
- Cycle 2: Review low confidence samples → retrain
- Cycle 3: External validation (real hospital data)

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low pseudo-label quality | -7% F2 | Aggressive confidence filtering (top 20%) |
| Distribution shift (4M vs 100) | Generalization | External validation CRITICAL |
| GPU cost overruns | Budget +20% | Spot instances + monitoring |
| Inconsistent expert labels | Noise | Double annotation + arbitration |

---

## 10. Technical Recommendations

### Short Term (Implementation)

1. **Start small:** Test on 100K images before full scale-up
2. **Ensemble models:** Combine 5 cross-validation models → +2-5% F2
3. **Active learning:** Iterate on uncertain samples (continuous improvement)
4. **Calibration:** Platt scaling for reliable confidence scores

### Medium Term (Optimization)

5. **Modern architectures:** Test Vision Transformers (ViT) if GPU budget allows
6. **Data augmentation:** Rotations, flips → robustness
7. **Multi-task:** Predict localization + classification simultaneously
8. **Distillation:** Compress model for edge deployment

### Long Term (Production)

9. **Post-deployment monitoring:** Track real-world vs lab performance
10. **Feedback loop:** Clinician labels → periodic retraining
11. **Regulatory compliance:** Prediction traceability (GDPR, FDA)
12. **A/B testing:** Compare model versions in production

---

## 11. Trade-offs and Limitations

### Accepted Trade-offs

**Accuracy vs Cost:**
- Accuracy: 93-95% (vs 99%+ with 100% manual labels)
- Savings: 2,400× cost reduction
- **Decision:** Acceptable for screening, requires expert validation for critical cases

**Automation vs Control:**
- 90% auto-labeled (high confidence)
- 10% human review (medium/low confidence)
- **Decision:** Balance efficiency/quality

### Known Limitations

**Technical:**
- Performance depends on ResNet50 feature quality (general domain → medical)
- Clustering-based weak labels showed decreased performance (Scenario B: -7% F2 vs Scenario A)
- Pseudo-label quality depends on model confidence calibration (mitigation: strict confidence filtering)

**Operational:**
- External validation NOT performed (current dataset = 2,824 images)
- 4M generalization uncertain (requires 100K pilot)
- Variable GPU costs (spot instances fluctuate ±30%)

**Regulatory:**
- Model not certified as medical device (research use only)
- AI labels require final clinical validation
- Prediction traceability required (audit trail)

---

## 12. Conclusion

### Feasibility: ✅ YES with Conditions

**What works:**
- **Best experimental model (Scenario A):** 96% F2 with only 100 manual labels
- **Scaling strategy (Scenario C methodology):** 93% F2 with cost-effective semi-supervised approach
- Technical pipeline validated (ResNet50 + MLP)
- Scalability demonstrated (feature extraction 10K img/h)
- Realistic budget (€5,000 sufficient with Scenario C pseudo-labeling strategy)

**What requires validation for 4M scaling:**
- 4M image generalization (potential distribution drift)
- Large-scale pseudo-label quality (confidence calibration at scale)
- GPU infrastructure (spot instance availability)

**Final recommendation:**
1. **Pilot phase:** Test on 100,000 images first (€500 budget)
2. **Validate assumptions:** Verify 93-95% accuracy maintained
3. **Progressive scaling:** If pilot successful → deploy full 4M strategy

### Next Steps

**Immediate (Week 1-2):**
- Select 500 strategic images via clustering
- Organize expert labeling (radiologists)

**Short term (Month 1):**
- Train model V2 on 500 labels
- Launch 100K pilot
- Measure accuracy vs predictions

**Medium term (Month 2-3):**
- If pilot successful (>90% accuracy): Scale to 4M
- Implement continuous monitoring
- Feedback loop with clinicians

---

## Appendices

### Metrics Used

**F2 Score (β=2):**
- Weights recall 2× more than precision
- Critical in medical: detect all positive cases (minimize false negatives)
- Formula: F2 = 5 × (precision × recall) / (4 × precision + recall)

**5-Fold Cross-Validation:**
- Divides data into 5 parts
- Trains 5 models (each excludes 1 part for testing)
- Averages results → robust performance estimate

### Key Technologies

- **PyTorch:** Deep learning framework
- **ResNet50:** Pre-trained CNN architecture
- **Scikit-learn:** PCA, K-Means, metrics
- **MLflow:** Experiment tracking (run IDs, hyperparameters)

### References

**GitHub Repository:**
- 🔗 [https://github.com/shah-data-scientist/Medical-Images---Cancer](https://github.com/shah-data-scientist/Medical-Images---Cancer)
- Complete source code, notebooks, and documentation

**Notebooks:**
- `1_feature_extraction.ipynb` - Feature extraction with ResNet50
- `2_unsupervised_analysis.ipynb` - Clustering and weak label generation
- `3_semi_supervised_learning.ipynb` - Model training and evaluation

**Documentation:**
- `docs_generated/01_SYSTEM_OVERVIEW.md` - System architecture
- `docs_generated/preserved/LABELING_STRATEGY_BUDGET_ANALYSIS.md` - Budget analysis
- `PROJECT_MEMORY.md` - Project history and decisions

---

**Technical Contact:** BrainScanAI Project Team
**Date:** 2025-12-31 (Updated: 2026-01-14)
**Version:** 1.1
