# Preserved Non-Code Documentation

**Source:** Whitelisted strategic/policy documents (not derived from code)
**Last Verified:** 2025-12-28

---

## Purpose

This section preserves **3 non-code markdown files** that contain strategic analysis, policies, and planning documents that cannot be derived from running code alone.

**Preservation Criteria:**
- Documents strategic decisions and business logic
- Contains external context (budget, timeline, business constraints)
- Provides rationale for technical choices
- Not derivable from code inspection

---

## Preserved Documents

### 1. [LABELING_STRATEGY_BUDGET_ANALYSIS.md](preserved/LABELING_STRATEGY_BUDGET_ANALYSIS.md)

**Type:** Strategic Business Analysis
**Lines:** 810
**Created:** 2025-12-28

**Purpose:** Comprehensive budget analysis for scaling data labeling from 100 images (€300) to 4M images (€5,000)

**Key Content:**
- **Current Reality:** €3.00/image unit cost from manual labeling
- **Challenge:** Need 2,400x cost reduction to meet budget constraint
- **4 Strategic Approaches:**
  1. Semi-Supervised Learning (500 labels, 93-95% accuracy, €5,000, 8-12 weeks)
  2. Active Learning (1,300 labels, 95-97% accuracy, €5,200, 16 weeks) ⚠️ Over budget
  3. Weak Supervision (1,000 labels, 88-92% accuracy, €4,500, 4-6 weeks)
  4. **Hybrid (RECOMMENDED)** (800 labels, 94-96% accuracy, €5,000, 12-14 weeks)
- **Feasibility Assessment:** 90% confidence in success
- **Risk Analysis:** Budget overruns, accuracy targets, timeline delays

**Why Preserved:**
- Contains external business constraints (€5,000 budget)
- Strategic trade-off analysis not derivable from code
- Timeline estimates based on operational experience
- Budget calculations specific to this project's context

**Cross-References:**
- Validates approach used in `3_semi_supervised_learning.ipynb`
- Justifies 80/20 data split strategy
- Explains confidence threshold choices (0.7)

---

### 2. [DOCUMENTATION_POLICY.md](preserved/DOCUMENTATION_POLICY.md)

**Type:** Meta-Documentation Policy
**Lines:** 177
**Created:** 2025-12-28

**Purpose:** Defines documentation standards and code-first reconstruction methodology

**Key Content:**
- **Policy:** Regenerate all technical docs from code, preserve only strategic/policy docs
- **Rationale:** Prevent documentation drift, ensure accuracy
- **Scope:** Applies to all `.md` files in root directory
- **Whitelisted Files:** Lists 3 preserved documents with justification
- **Regeneration Files:** Lists 7 documents to rebuild from code
- **Verification:** All docs must cite source (code file, line numbers, or "derived from code")

**Why Preserved:**
- Meta-level policy document (self-referential)
- Defines documentation standards not derivable from code
- Authorizes this preservation process
- Contains decision rationale for documentation approach

**Cross-References:**
- Authorizes creation of `docs_generated/` folder structure
- Justifies preservation of these 3 files
- Defines standards used in 00-04 documentation files

---

### 3. [alternative_validation_plan.md](preserved/alternative_validation_plan.md)

**Type:** Strategic Technical Plan
**Lines:** 436
**Created:** 2025-12-28

**Purpose:** Alternative validation strategies when external validation dataset unavailable

**Key Content:**
- **7 Alternative Strategies:**
  1. ✅ **80/20 Internal Split (IMPLEMENTED)** - Use 20% as held-out test set
  2. ✅ **K-Fold Cross-Validation (IMPLEMENTED)** - 5-fold stratified CV
  3. ✅ **Feature Importance Analysis (IMPLEMENTED)** - Permutation importance
  4. ✅ **Visualization (IMPLEMENTED)** - t-SNE class separation
  5. ✅ **Noise Robustness (IMPLEMENTED)** - Gaussian noise injection
  6. ⏳ Bootstrap Confidence Intervals - Non-parametric uncertainty
  7. ⏳ Temporal Split - If images have acquisition dates
- **Implementation Status:** 5/7 strategies completed
- **Results Summary:** All implemented validations passed
- **Rationale:** Explains why each strategy provides validation evidence

**Why Preserved:**
- Strategic planning document with decision rationale
- Explains validation choices made in notebooks
- Contains implementation trade-offs not visible in code
- Provides context for why certain approaches were selected

**Cross-References:**
- Justifies validation functions in `advanced_validation_analysis.py`
- Explains analyses in `run_validation_analysis.py`
- Documents validation results in `3_semi_supervised_learning.ipynb`

---

## Relationship to Code-Derived Docs

**These preserved docs provide context for:**
- [01_SYSTEM_OVERVIEW.md](01_SYSTEM_OVERVIEW.md) - Why 80/20 split chosen
- [02_HOW_TO_RUN.md](02_HOW_TO_RUN.md) - What validation outputs to expect
- [03_MODULE_API.md](03_MODULE_API.md) - Why validation functions designed this way
- [04_CONFIGURATION.md](04_CONFIGURATION.md) - Why certain hyperparameters chosen

**Code-derived docs describe "what" and "how"**
**Preserved docs explain "why" and "why not"**

---

## Verification Status

| Document | Source | Lines | Last Modified | Status |
|----------|--------|-------|---------------|--------|
| LABELING_STRATEGY_BUDGET_ANALYSIS.md | Business analysis | 810 | 2025-12-28 | ✅ Preserved |
| DOCUMENTATION_POLICY.md | Meta-policy | 177 | 2025-12-28 | ✅ Preserved |
| alternative_validation_plan.md | Strategic plan | 436 | 2025-12-28 | ✅ Preserved |

**Preservation Complete:** 2025-12-28

---

## Update Policy

**When to Update Preserved Docs:**
1. **Budget changes:** If labeling budget or constraints change
2. **Strategy pivots:** If validation approach fundamentally changes
3. **Policy revisions:** If documentation standards evolve

**When to Add New Preserved Docs:**
- New strategic business decisions made
- External constraints or requirements added
- Policy or governance documents created

**Do NOT Preserve:**
- Technical documentation derivable from code
- API references, configuration guides, how-to guides
- System overviews, architecture diagrams from code structure
- Any content that becomes stale when code changes

---

**Note:** All preserved documents are copied to `docs_generated/preserved/` to maintain a complete documentation package. Original files remain in project root for easy access during development.
