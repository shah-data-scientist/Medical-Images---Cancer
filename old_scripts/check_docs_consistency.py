#!/usr/bin/env python3
"""
Documentation Consistency Checker

This script checks for inconsistencies between code and documentation.

Usage:
    python check_docs_consistency.py

Output:
    Report of documentation issues and inconsistencies
"""

import os
import re
from pathlib import Path
from typing import List, Dict


class DocumentationChecker:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

    def find_files(self, extensions: List[str]) -> List[Path]:
        """Find all files with given extensions."""
        files = []
        for ext in extensions:
            files.extend(self.project_root.glob(f"**/*{ext}"))
        return [f for f in files if '.git' not in str(f)]

    def check_notebook_markdown(self, notebook_path: Path) -> List[str]:
        """Check if notebook has adequate markdown cells."""
        issues = []
        try:
            import json
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            markdown_cells = [c for c in nb.get('cells', []) if c['cell_type'] == 'markdown']
            code_cells = [c for c in nb.get('cells', []) if c['cell_type'] == 'code']
            if not markdown_cells:
                issues.append(f"No markdown in {notebook_path.name}")
            elif code_cells and len(markdown_cells) / len(code_cells) < 0.2:
                issues.append(f"Low docs ratio in {notebook_path.name}")
        except Exception as e:
            issues.append(f"Error reading {notebook_path.name}: {e}")
        return issues

    def run_checks(self):
        """Run all documentation checks."""
        print("🔍 Documentation Consistency Check\n")
        notebooks = self.find_files(['.ipynb'])
        all_issues = []
        for nb in notebooks:
            issues = self.check_notebook_markdown(nb)
            all_issues.extend(issues)

        if not all_issues:
            print("✅ No issues found!\n")
        else:
            print(f"⚠️  Found {len(all_issues)} issues:\n")
            for issue in all_issues:
                print(f"  - {issue}")
        return len(all_issues)


if __name__ == '__main__':
    checker = DocumentationChecker()
    checker.run_checks()
