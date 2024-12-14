.PHONY: help

install:
	uv sync --all-extras
	uv tool install pre-commit --with pre-commit-uv --force-reinstall

.PHONY: default
default: install

test:
	uv run pytest

check:
	uv run pre-commit run --all-files

build-kfp:
	uv run python -m main
	uv run pre-commit run --all-files

build:
	uv build

coverage:
	uv run pytest --cov=ml_orchestrator --cov-report=xml

clear:
	uv venv --python 3.11
