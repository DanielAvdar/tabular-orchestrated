.PHONY: help

install:
	uv sync --all-extras --all-groups --frozen
	uv tool install pre-commit --with pre-commit-uv --force-reinstall

.PHONY: default
default: install

test:
	uv run pytest

check:
	uvx pre-commit run --all-files

comps:
	uv run python -m main
	uv tool run pre-commit run --all-files

build:
	uv build

coverage:
	uv run pytest --cov=tabular_orchestrated --cov-report=xml

clear:
	uv venv --python 3.10

update:
	uv lock

	uvx pre-commit autoupdate
	$(MAKE) install


mypy:
	uv tool run mypy . --config-file pyproject.toml
