# Documentation Policy - Medical Images Cancer Project

## 📌 Global Policy Reference

This project follows the **[Global Documentation Policy](C:\Users\shahu\Documents\GLOBAL_DOCUMENTATION_POLICY.md)**.

All requirements from the global policy apply to this project.

---

## 🎯 Project-Specific Requirements

In addition to the global policy, this medical imaging project requires:

### Medical/Scientific Documentation
- All models must document their architecture and rationale
- Training parameters must be documented in notebooks
- Evaluation metrics must be explained
- Data preprocessing steps must be documented

### Notebook-Specific Rules
- Each analysis notebook must have:
  - Summary markdown cell at the top
  - Section headers for each major step
  - Interpretation of results
  - References to papers/methods used

---

## Solution: Documentation-First Workflow (from Global Policy)

### 🎯 Core Principle
**Every code change MUST include documentation updates in the same commit.**

---

## Claude's Required Behavior

When Claude (AI assistant) modifies code, it MUST:

### 1. **Before Making Changes**
- [ ] List all documentation files that may be affected
- [ ] Identify which markdown files describe the code being changed
- [ ] Check if README or other top-level docs reference this code

### 2. **During Changes**
- [ ] Update code AND documentation in the SAME response
- [ ] Update notebook markdown cells when modifying notebook code cells
- [ ] Update standalone .md files when modifying related .py scripts

### 3. **After Changes**
- [ ] Explicitly list which documentation files were updated
- [ ] Confirm consistency between code and docs
- [ ] If docs are too extensive, create TODO items with specific file references

### 4. **When Asked to "Check Documentation Consistency"**
Claude must:
- [ ] Read ALL relevant markdown files
- [ ] Compare with current codebase
- [ ] Provide a detailed inconsistency report with:
  - File name
  - Section with inconsistency
  - What docs say vs. what code does
  - Recommended fix
- [ ] Offer to fix ALL inconsistencies (not just mention them)

---

## Documentation Structure

### Required Documentation Files

```
project/
├── README.md                          # Project overview, quick start
├── ARCHITECTURE.md                    # System design, components
├── DOCUMENTATION_POLICY.md            # This file
├── docs/
│   ├── 00_INDEX.md                   # Master index of all documentation
│   ├── 01_setup.md                   # Installation and setup
│   ├── 02_usage.md                   # How to use the system
│   ├── 03_api_reference.md           # API documentation
│   ├── 04_notebooks_guide.md         # Notebook documentation index
│   └── 05_troubleshooting.md         # Common issues
├── notebooks/
│   ├── 1_exploration.ipynb           # Each notebook has markdown cells
│   ├── 2_training.ipynb              # explaining what it does
│   └── 3_evaluation.ipynb
└── scripts/
    ├── train.py                       # Inline docstrings + external docs
    └── evaluate.py
```

### Documentation Hierarchy

1. **README.md** - Entry point, links to everything
2. **docs/00_INDEX.md** - Master index linking all documentation
3. **Specialized docs** - Detailed documentation by topic
4. **Inline docs** - Docstrings, notebook markdown cells

---

## Enforcement Mechanisms

### 1. Git Pre-commit Hook
- Warns when code is modified without documentation updates
- Located at: `.git/hooks/pre-commit-docs-check`

### 2. Mandatory Checklist
Before any commit, verify:
- [ ] Code changes are documented
- [ ] Notebook markdown cells updated
- [ ] Related .md files updated
- [ ] docs/00_INDEX.md reflects changes
- [ ] README.md updated if needed

### 3. Regular Audits
Monthly documentation consistency check:
```bash
# Request from Claude:
"Perform a complete documentation audit:
1. List all .py, .ipynb files
2. List all .md files
3. For each code file, identify which docs describe it
4. Check if docs are current
5. Report all inconsistencies
6. Provide fix plan"
```

---

## Templates

### Notebook Markdown Cell Template
```markdown
# Section: [Name]

## Purpose
[What this section does]

## Inputs
- Variable/file 1: [description]
- Variable/file 2: [description]

## Process
1. Step 1
2. Step 2

## Outputs
- Result 1: [description]
- Result 2: [description]

## Related Documentation
- See docs/[relevant-doc].md for details
```

### Script Documentation Template
```python
"""
Module: [name]

Purpose:
    [What this module does]

Usage:
    python script.py --arg1 value1

Related Documentation:
    - docs/03_api_reference.md
    - notebooks/2_training.ipynb

Author: [name]
Last Updated: [date]
"""
```

---

## How to Request Documentation Updates from Claude

### ❌ Don't Say:
- "Update the code"
- "Fix this bug"
- "Add this feature"

### ✅ Do Say:
- "Update the code AND all related documentation files"
- "Fix this bug and update docs/troubleshooting.md to reflect the fix"
- "Add this feature and ensure README, relevant notebooks, and API docs are updated"

### 📋 Use This Template:
```
Task: [description]

Requirements:
1. Modify code as needed
2. Update ALL affected documentation files including:
   - Notebook markdown cells
   - Relevant .md files in docs/
   - README.md if applicable
3. Confirm which docs were updated
4. Verify consistency
```

---

## Periodic Maintenance Commands

### Weekly Check
```
"Claude, check if documentation in docs/ is consistent with current codebase.
List any drift and provide fix commands."
```

### After Major Changes
```
"Claude, I've made changes to [files]. Please:
1. Review all documentation for consistency
2. Update docs/00_INDEX.md
3. Update README.md if needed
4. Confirm all cross-references are valid"
```

### Documentation Reorganization
```
"Claude, audit all .md files and notebooks. Propose:
1. Better organization structure
2. Removal of redundant documentation
3. Consolidation of overlapping docs
4. Master index update"
```

---

## Success Criteria

Documentation is considered **good** when:

✅ Every code file has corresponding documentation
✅ All documentation files are listed in docs/00_INDEX.md
✅ Notebook markdown cells explain what code does
✅ No contradictions between docs and code
✅ Documentation is updated in same commit as code
✅ Cross-references between docs are valid
✅ README provides accurate overview

---

## Enforcement Level

**STRICT MODE** (Recommended):
- Block commits without documentation updates
- Set in `.git/hooks/pre-commit-docs-check`: change `exit 0` to `exit 1`

**WARNING MODE** (Current):
- Warn about missing docs but allow commit
- Relies on developer discipline

---

## This is a Living Document

Update this policy when:
- Documentation structure changes
- New documentation requirements emerge
- Better practices are discovered

Last Updated: 2025-12-26
