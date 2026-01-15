# Contributing to BrainScanAI

Welcome to the BrainScanAI project! We appreciate your interest in improving our brain tumor detection system.

This document guides you through the contribution process, ensuring consistency and quality across the codebase.

---

## 🚀 Quick Start

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Medical-Images-Cancer
    ```

2.  **Install dependencies (Poetry):**
    ```bash
    poetry install
    ```

3.  **Activate the environment:**
    ```bash
    poetry shell
    ```

---

## 🛠 Development Workflow

### 1. Code Style
We follow strict coding standards enforced by **Ruff**.
*   **Linter:** `ruff`
*   **Formatter:** `ruff format`

**Before committing, run:**
```bash
poetry run ruff check . --fix
```

### 2. Testing
We use **pytest** for testing, including automated notebook validation.

**Run all tests:**
```bash
poetry run pytest --nbval notebooks/
```
*   *Note:* Notebook tests may report failures due to output mismatches (timestamps/run IDs). Check that the execution itself passed (no errors).

### 3. Documentation
*   **Policy:** Follow the global policy at `C:\Users\shahu\Documents\coding_agent_policies\GLOBAL_POLICY.md`.
*   **Requirement:** Every code change **MUST** be accompanied by an update to `PROJECT_MEMORY.md`.
*   **Notebooks:** Must include a "Key Findings" section at the end.

---

## 🧠 Project Structure

*   `notebooks/`: Core analysis pipelines (`1_...`, `2_...`, `3_...`).
*   `scripts/`: Reusable Python modules for validation and analysis.
*   `data/`: Input MRI images (gitignored).
*   `features/`: Intermediate processed data (gitignored).
*   `docs_generated/`: Auto-generated system documentation.

---

## 🔒 Security

*   **Secrets:** Never commit API keys, passwords, or tokens.
*   **Pre-commit:** Our git hooks will block commits containing secrets or security vulnerabilities (OWASP Top 10).

---

## 📦 Data Management

See [DATA_DICTIONARY.md](docs_generated/DATA_DICTIONARY.md) for detailed definitions of our data schemas and labels.

---

**Questions?** Check `PROJECT_MEMORY.md` for the latest project context.
