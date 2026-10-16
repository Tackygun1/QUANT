# Contributing

This repository is in early-stage setup. Contributions should prioritize clarity, reversible changes, and enough documentation that future work does not rely on tribal knowledge.

## Workflow

1. Create a focused branch from `main`.
2. Make the smallest change that moves the project forward.
3. Update docs when behavior, interfaces, or architecture change.
4. Install dev tooling with `python3 -m pip install -e '.[dev,ml]'` and run `make check` before opening a pull request. Use `.[dev,stack]` plus CPU PyTorch for the full optional ML test surface.
5. Describe verification in the pull request, even if the change is documentation-only.

## Quality Bar

- New code should include focused tests.
- Configuration changes should document local development impact.
- Behavior changes should explain user impact and rollback expectations.
- Ambiguous naming, dead code, and TODOs without context should not be merged.

## Pull Requests

Each pull request should include:

- A concise problem statement.
- The chosen approach and any tradeoffs.
- Verification steps and outcomes.
- Follow-up work, if the change intentionally leaves gaps.

## Scope Discipline

Do not combine unrelated cleanup with feature work unless the cleanup is required to make the feature correct. Keep refactors mechanically safe and well-explained.
