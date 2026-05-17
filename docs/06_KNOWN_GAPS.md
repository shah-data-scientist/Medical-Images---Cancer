# Known Documentation Gaps

**Source:** Code inspection
**Last Verified:** 2025-12-28

---

## Purpose

This document **explicitly lists what is NOT documented** in the codebase, either in code or in generated documentation. These are areas where information must be inferred from code behavior or is simply unknown.

**Transparency Principle:** It's better to acknowledge gaps than to pretend complete documentation exists.

---

## 1. Missing In-Code Documentation

### 1.1 No Function Docstrings in Notebooks

**Location:** All 3 notebooks

**Impact:** High - Functions lack formal documentation

**Examples:**
```python
# 1_feature_extraction.ipynb
def extract_features(image_dir, model):
    # NO DOCSTRING
    # What does it return? What exceptions can it raise?
    pass

def apply_pca(features, n_components):
    # NO DOCSTRING
    # What's the return shape? What if n_components > n_features?
    pass
```

**Consequences:**
- Parameter types unclear (must infer from usage)
- Return value structure undocumented
- Exception behavior unknown
- Side effects not declared (saves files)

**Workaround:** See [03_MODULE_API.md](03_MODULE_API.md) for inferred API documentation

---

### 1.2 No Type Hints

**Location:** All Python files (notebooks, scripts)

**Impact:** Medium - Type safety not enforced

**Example:**
```python
# What are the types? Must guess from usage
def analyze_feature_importance(model, X_test, y_test, n_repeats=30, random_state=42):
    # model: object? nn.Module? SklearnWrapper?
    # X_test: ndarray? DataFrame? Tensor?
    # Returns: dict? tuple? object?
    pass
```

**Missing Information:**
- Input parameter types
- Return value types
- Union types for optional parameters
- Generic types for containers

**Workaround:** Inferred from code analysis in [03_MODULE_API.md](03_MODULE_API.md)

---

### 1.3 No Error Handling Documentation

**Location:** All files

**Impact:** Medium - Exception behavior unknown

**Examples:**
```python
# What exceptions can this raise?
features = np.load('features/features_pca_50.npy')
# FileNotFoundError? PermissionError? IOError?

# What if this fails?
model = RegularizedMLP(input_size=50, hidden_size=64)
# ValueError for negative sizes? TypeError for strings?

# What happens on GPU memory error?
loss.backward()
# RuntimeError? CudaOutOfMemoryError?
```

**Missing Information:**
- Exception types that can be raised
- Error messages and meanings
- Recovery strategies
- Validation of inputs

**Workaround:** Test empirically or inspect torch/sklearn source code

---

### 1.4 No Performance Documentation

**Location:** All functions

**Impact:** Low - Runtime unpredictable

**Examples:**
```python
# How long does this take?
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_coords = tsne.fit_transform(combined_features)
# 1 second? 1 minute? Scales with dataset size how?

# Memory usage?
features = extract_features(image_dir, model)
# 1 GB? 10 GB? Depends on number of images how?
```

**Missing Information:**
- Time complexity (O notation)
- Space complexity
- GPU memory requirements
- Scaling behavior

**Workaround:** Runtime observations in [02_HOW_TO_RUN.md](02_HOW_TO_RUN.md)

---

### 1.5 No Inline Comments for Complex Logic

**Location:** Notebooks, especially clustering and training loops

**Impact:** Medium - Intent unclear

**Example:**
```python
# What is this threshold for? Why 0.7?
high_confidence_mask = (
    (weak_labels_df['confidence'] > 0.7) &
    (weak_labels_df['weak_label'] != -1)
)

# Why these specific regularization values?
DROPOUT_RATE = 0.70
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
# Empirically determined? From literature? Random?
```

**Missing Information:**
- Rationale for magic numbers
- Algorithm choice justification
- Edge case handling logic

**Workaround:** Infer from context or see [LABELING_STRATEGY_BUDGET_ANALYSIS.md](preserved/LABELING_STRATEGY_BUDGET_ANALYSIS.md) for strategic rationale

---

## 2. Missing Configuration Documentation

### 2.1 No Config File Support

**Location:** Entire project

**Impact:** Medium - All configuration hardcoded

**Gap:**
```python
# These are HARDCODED in notebooks
RANDOM_STATE = 42
IMAGE_SIZE = (224, 224)
PCA_COMPONENTS = 50
N_CLUSTERS = 2
CONFIDENCE_THRESHOLD = 0.7

# No way to override without editing code
# No config.yaml, config.json, or .env support
```

**Consequences:**
- Must edit code to change parameters
- No environment-specific configs (dev/prod)
- Difficult to run parameter sweeps
- No configuration validation

**Documented In:** [04_CONFIGURATION.md](04_CONFIGURATION.md) - "Known Limitations" section

---

### 2.2 No Environment Variables

**Location:** All scripts

**Impact:** Low - Paths hardcoded

**Gap:**
```bash
# These don't work:
export BRAINSCAN_DATA_DIR=/custom/path
export BRAINSCAN_RANDOM_SEED=123
export BRAINSCAN_MODEL_PATH=/models/checkpoint.pth

# Code always uses hardcoded paths:
data_dir = "data/labelled/yes/"
features_path = "features/features_pca_50.npy"
```

**Consequences:**
- Cannot customize paths without code edits
- Difficult to run in different environments
- No support for containerization best practices

---

### 2.3 No CLI Argument Support

**Location:** Standalone scripts (run_validation_analysis.py, advanced_validation_analysis.py)

**Impact:** Low - Scripts have fixed behavior

**Gap:**
```bash
# This doesn't work:
python run_validation_analysis.py --noise-levels 0.1,0.2,0.3 --output results/

# Scripts always use hardcoded defaults
# No argparse or click support
```

**Consequences:**
- Cannot customize behavior from command line
- Must edit code to change parameters
- Difficult to integrate into pipelines

---

## 3. Missing Deployment Documentation

### 3.1 No Production Deployment Guide

**Location:** Documentation

**Impact:** Medium - Unclear how to deploy

**Missing Information:**
- How to serve model for inference
- REST API or batch processing setup
- Model checkpoint management
- Monitoring and logging setup
- Rollback procedures

**Note:** This is a research project, not production software. Deployment may be out of scope.

---

### 3.2 No Docker/Container Support

**Location:** Project root

**Impact:** Low - Must install dependencies manually

**Missing Files:**
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- Container registry info

**Consequences:**
- Manual environment setup required
- Difficult to ensure reproducibility across systems
- No easy cloud deployment

---

### 3.3 No CI/CD Pipeline

**Location:** Project root

**Impact:** Low - Manual testing required

**Missing:**
- `.github/workflows/` (GitHub Actions)
- `.gitlab-ci.yml` (GitLab CI)
- Pre-commit hooks for linting
- Automated testing on push

**Consequences:**
- Cannot verify changes automatically
- No automated model validation
- Manual regression testing needed

---

## 4. Missing Data Documentation

### 4.1 No Data Schema Documentation

**Location:** Data files

**Impact:** Medium - Format unclear

**Missing:**
```
features_pca_50.npy:
  - Shape: (?, 50) - What's the first dimension?
  - Dtype: float32? float64?
  - Range: Normalized? Standardized? Raw?
  - Missing values: Allowed? How encoded?

weak_labels.csv:
  - Columns: Which are required? Which optional?
  - Value ranges: What do -1, 0, 1 mean?
  - Confidence scale: 0-1? 0-100? Arbitrary?
```

**Workaround:** Inspect files manually or see [01_SYSTEM_OVERVIEW.md](01_SYSTEM_OVERVIEW.md) for inferred structure

---

### 4.2 No Data Versioning

**Location:** `features/` directory

**Impact:** Low - Cannot track data changes

**Missing:**
- Data version tracking (DVC, git-lfs)
- Data lineage (which script generated which file)
- Data validation (checksums, schema validation)
- Data changelog

**Consequences:**
- Cannot reproduce results if data changes
- No audit trail for feature engineering
- Difficult to debug data issues

---

## 5. Missing Testing Documentation

### 5.1 No Unit Tests

**Location:** Entire project

**Impact:** Medium - No automated verification

**Missing:**
```
tests/
├── test_feature_extraction.py
├── test_clustering.py
├── test_model.py
└── test_validation.py
```

**Consequences:**
- Cannot verify functions work correctly
- Regressions undetected until runtime
- Refactoring risky

**Note:** Validation analyses serve as integration tests, but no unit tests exist.

---

### 5.2 No Test Coverage Metrics

**Location:** Entire project

**Impact:** Low - Unknown code coverage

**Missing:**
- pytest-cov configuration
- Coverage reports
- Coverage badges

---

## 6. Missing Dependency Documentation

### 6.1 No Dependency Pinning Justification

**Location:** `pyproject.toml`

**Impact:** Low - Unclear why versions chosen

**Gap:**
```toml
# Why these version constraints?
torch = ">=2.9.1,<3.0.0"  # Why not 2.0.0? Why block 3.x?
scikit-learn = ">=1.8.0,<2.0.0"  # Why 1.8 minimum?

# No documentation of:
# - Compatibility issues between versions
# - Known bugs in excluded versions
# - Features required from minimum versions
```

---

### 6.2 No Security Audit

**Location:** Dependencies

**Impact:** Low - Unknown vulnerabilities

**Missing:**
- `pip-audit` or `safety` checks
- Vulnerability scanning
- Security update policy

---

## 7. Missing Operational Documentation

### 7.1 No Troubleshooting Guide

**Location:** Documentation

**Impact:** Medium - Users must debug alone

**Missing:**
```markdown
## Common Issues

### "FileNotFoundError: features/features_pca_50.npy"
Solution: Run notebook 1 first to generate features

### "CUDA out of memory"
Solution: Reduce batch size in config

### "Perfect scores (1.0000) in cross-validation"
Solution: Check for data leakage
```

**Workaround:** See [POST_COMMIT_AUDIT_2025-12-28.md](../POST_COMMIT_AUDIT_2025-12-28.md) for historical issues and fixes

---

### 7.2 No Performance Tuning Guide

**Location:** Documentation

**Impact:** Low - Optimization unclear

**Missing:**
- How to speed up t-SNE
- GPU vs CPU trade-offs
- Memory optimization tips
- Batch size tuning guidance

---

### 7.3 No Logging Configuration

**Location:** All code

**Impact:** Low - Debugging difficult

**Gap:**
```python
# No structured logging:
import logging
logging.basicConfig(...)

# Only print statements:
print(f"Epoch {epoch}: Loss = {loss.item()}")
```

**Consequences:**
- Cannot filter log levels
- No log file output
- Difficult to debug production issues

---

## 8. API Stability

### 8.1 No Versioning

**Location:** All modules

**Impact:** Low - Breaking changes untracked

**Missing:**
```python
# No version numbers:
__version__ = "0.1.0"

# No deprecation warnings:
@deprecated("Use new_function instead")
def old_function():
    pass
```

**Consequences:**
- Breaking changes undocumented
- No migration guides
- Users unaware of API changes

---

### 8.2 No Changelog

**Location:** Project root

**Impact:** Low - Changes untracked

**Missing:**
- `CHANGELOG.md`
- Release notes
- Migration guides

**Workaround:** Git commit history serves as informal changelog

---

## Summary

### Gap Categories by Severity

**High Impact (5 gaps):**
1. No function docstrings in notebooks
2. No type hints
3. No data schema documentation
4. No config file support
5. No troubleshooting guide

**Medium Impact (6 gaps):**
1. No error handling documentation
2. No inline comments for complex logic
3. No production deployment guide
4. No unit tests
5. No performance documentation
6. No CLI argument support

**Low Impact (12 gaps):**
1. No environment variables
2. No Docker support
3. No CI/CD
4. No data versioning
5. No test coverage metrics
6. No dependency pinning justification
7. No security audit
8. No performance tuning guide
9. No logging configuration
10. No API versioning
11. No changelog
12. No Docker/container support

### Total Documented Gaps: 23

---

## Remediation Priority

**If time permits, address in this order:**

1. **Add function docstrings to notebooks** - Highest value, moderate effort
2. **Create troubleshooting guide** - High value, low effort
3. **Document data schema** - High value, low effort
4. **Add type hints to key functions** - Medium value, medium effort
5. **Create config.yaml support** - Medium value, high effort

**Not recommended:**
- Docker/CI/CD (overkill for research project)
- Unit tests (validation analyses already provide coverage)
- API versioning (single-user project)

---

## Acceptance Criteria

**This project is complete without these gaps because:**
1. It's a **research project**, not production software
2. **Notebooks are self-documenting** through execution outputs
3. **Validation analyses** provide quality assurance
4. **Generated documentation** fills most gaps through inference
5. **Code is simple enough** to understand without extensive docs

**These gaps are acceptable trade-offs** for a time-constrained academic project.

---

**Last Updated:** 2025-12-28
**Verified By:** Code inspection and documentation audit
