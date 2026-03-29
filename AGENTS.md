# AGENTS.md

Agent instructions for this repository:

- Follow the Cursor rule in `.cursor/rules/quality-gate.mdc`.
- Do not mark a task complete until `uv run ruff check .` passes and `uv run mypy .` passes.
- If you change Python code, run both checks before finishing.
- Do not bypass a failing check by weakening rules unless the user explicitly asks for that change.
