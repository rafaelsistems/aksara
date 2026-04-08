# CHANGELOG

## Unreleased
- Introduced the term **Cognitive Language Model (CLM)** as the public label for AKSARA.
- Clarified that AKSARA is a CLM focused on understanding, reasoning, and generative capability.
- Added public release guardrails via `.gitignore` and release packaging script.
- Added end-to-end readiness tests for generation, reasoning, and evaluator smoke checks.
- Added minimal public release packaging workflow to prevent internal data leakage.
- Updated README with public usage and verification notes.

## Baseline Public Release
- Public repository published from a sanitized allowlist-only build.
- Internal datasets, checkpoints, debug artifacts, and training outputs excluded from public release.
