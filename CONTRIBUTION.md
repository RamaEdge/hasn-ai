# Contributing to HASN-AI

Thank you for considering a contribution. This guide helps you set up your environment, follow coding standards, and submit high-quality changes.

## Quick start

- Fork and clone the repository
- Create a branch: `git checkout -b feat/<short-description>`
- Set up environment and tools:
  - Create venv and install deps: `make venv`
  - Lint: `make lint`
  - Format: `make format`
- Run locally:
  - API: `python src/api/main.py`
  - Training CLI: `python src/training/train_cli.py start`
  - Docker image: `make docker-build` then `docker run -p 8000:8000 hasn-ai:local`
- Push your branch and open a PR

## Development standards

- Python 3.11
- Black for formatting (line length 100)
- Ruff for linting (rules E, F, I; E203 and E501 ignored)
- Prefer descriptive names and add type hints for public APIs
- Avoid bare `except` where practical; handle specific exceptions
- Keep edits focused; update docs in `docs/` when behavior changes

## Linting, formatting, and security

- Lint: `make lint`
- Auto-format and quick fixes: `make format`
- Build image: `make docker-build`
- Security scans (requires Docker; local Trivy preferred if available):
  - Filesystem: `make trivy-fs`
  - Image: `make trivy-image`
  - Both: `make trivy-all`

## Tests

- Run tests (if present): `.venv/bin/pytest -q`
- Add tests for new behavior where feasible

## Commit and PR guidelines

- Commit messages: short imperative summary, optional body
- Branch naming: `feat/...`, `fix/...`, `chore/...`, `docs/...`, `refactor/...`
- PR checklist:
  - [ ] Code compiles and runs
  - [ ] `make lint` passes
  - [ ] `make format` applied
  - [ ] Security scan clean or findings documented
  - [ ] Docs updated (if user-facing change)

## Reporting issues

- Include reproduction steps, expected vs actual behavior, and logs/tracebacks
- For security-related issues, see `Security.md`

## License

By contributing, you agree your contributions will be licensed under the repository's license.
