# Documentation Policy - BrainScanAI

**Inherits from:** `C:\Users\shahu\Documents\GLOBAL_DOCUMENTATION_POLICY.md`

**Last Updated:** 2025-12-26

---

## 🎯 Local Requirements (Project-Specific)

### Jupyter Notebook Documentation (REQUIRED)

**Every notebook MUST include:**

1. **Findings Section** (at end of notebook)
   - Markdown cell titled "## Key Findings"
   - Bullet list of main results
   - Numerical results with ± uncertainty
   - Statistical significance (p-values if applicable)
   - Key visualizations/plots

2. **Summary Cell** (final cell)
   - What was done
   - What was learned
   - Next steps or implications

**Template for Findings Section:**
```markdown
## Key Findings

### Results
- Metric 1: XX.XX% ± YY.YY%
- Metric 2: XX.XX% ± YY.YY%
- Statistical test: p < 0.05 (significant/not significant)

### Insights
- [Key insight 1]
- [Key insight 2]
- [Limitation or caveat]

### Next Steps
- [Action item 1]
- [Action item 2]
```

**Location:** End of notebook, before any appendix/debugging sections

---

## 📓 Notebook-Specific Rules

### Structure Requirements

Each notebook must have:
1. **Title cell** - Notebook purpose (markdown)
2. **Table of Contents** - Section links (for long notebooks)
3. **Setup/Imports** - All imports in one cell at top
4. **Section headers** - Markdown cells with `##` or `###`
5. **Explanatory cells** - Before each major code block
6. **Key Findings** - Dedicated section at end (REQUIRED)
7. **Summary** - Final cell with conclusions

### Code Cell Documentation

- Add docstrings to any function defined in notebook
- Use inline comments for complex operations
- Explain "why" not "what" in markdown cells

### Output Management

- Keep outputs for key results/visualizations
- Clear outputs for debugging/intermediate steps
- Before commit: Clear debugging outputs, keep final results

---

## 📊 ML Experiment Documentation

**For this project (semi-supervised learning):**

### Required Documentation per Experiment

1. **Experiment metadata**
   - Date, scenario name
   - Hyperparameters used
   - Data split (train/test sizes)

2. **Results**
   - All metrics with uncertainty (mean ± std)
   - Confusion matrix or equivalent
   - Statistical comparison between scenarios

3. **Findings interpretation**
   - What worked well
   - What didn't work
   - Why (hypothesis)

**Location:** Findings section of relevant notebook + PROJECT_MEMORY.md

---

## 🔄 Update Workflow (Specific to This Project)

### After Running Experiments

1. Update notebook with results
2. Add findings to "Key Findings" section
3. Update PROJECT_MEMORY.md with summary
4. Update relevant .md files if needed (FINAL_REPORT.md, etc.)

### Before Commit

- Verify notebook has "Key Findings" section
- Verify findings match current results
- Clear unnecessary outputs
- Update PROJECT_MEMORY.md

---

## 📁 Documentation Files (This Repository)

**Core Files:**
- `README.md` - Project overview, setup
- `PROJECT_MEMORY.md` - Consolidated change log (PRIMARY)
- `DOCUMENTATION_POLICY.md` - This file (local policy)

**Reports (as needed):**
- `FINAL_REPORT.md` - Comprehensive results
- `EXECUTIVE_SUMMARY.md` - High-level overview
- `PRIORITY*_COMPLETE.md` - Milestone reports

**Rule:** Don't create new report files for every change. Update PROJECT_MEMORY.md instead.

---

## 🎓 AI Assistant Instructions (Local)

**When modifying notebooks in this repository:**

1. **ALWAYS add/update "Key Findings" section** if results changed
2. Update PROJECT_MEMORY.md with summary
3. Preserve cell outputs for final results
4. Use markdown cells to explain methodology changes

**Template response after notebook changes:**
```
✓ Updated [notebook]:
- [What changed in code]
- Key Findings section updated with [results]
- PROJECT_MEMORY updated
```

---

## 📋 Checklist: Before Marking Work Complete

- [ ] Notebook has "Key Findings" section
- [ ] Findings include numerical results with uncertainty
- [ ] PROJECT_MEMORY.md updated
- [ ] Statistical significance reported (if applicable)
- [ ] Outputs saved for key results
- [ ] Summary cell explains implications

---

## 💡 Local Customizations

**Differences from global policy:**
- Notebook findings are MANDATORY (not just recommended)
- Keep final outputs in notebooks (don't clear all)
- Statistical reporting required (mean ± std, p-values)

**Follows global policy for:**
- PROJECT_MEMORY.md workflow
- Code documentation (docstrings)
- Token optimization
- Commit workflow
