# Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies: `pip install -r requirements.txt`
4. Make changes with tests
5. Run checks: `ruff check . --fix && ruff format . && pytest`
6. Commit with conventional format: `feat(scope): description`
7. Push and open a PR

## Code Standards

- Type hints on all public functions
- `ruff` for linting and formatting
- `pytest` for testing
- No `print()` — use `logging.getLogger(__name__)`
