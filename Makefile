SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf .venv/bin/python; else printf python3; fi)

.PHONY: help bootstrap check lint test tree

help: ## Show available commands.
	@awk 'BEGIN {FS = ": ## "}; /^[a-zA-Z_-]+: ## / {printf "%-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## Prepare the local repository shell.
	@./scripts/bootstrap.sh

check: ## Validate repository shell, linting, and tests.
	@./scripts/check-repo.sh
	@$(MAKE) lint
	@$(MAKE) test

lint: ## Run static checks.
	@$(PYTHON) -m ruff check .

test: ## Run unit tests.
	@$(PYTHON) -m pytest -q

tree: ## Print the current top-level repository layout.
	@find . -maxdepth 2 \
		-not -path './.git' \
		-not -path './.git/*' \
		-not -path './node_modules' \
		-not -path './node_modules/*' \
		| sort
