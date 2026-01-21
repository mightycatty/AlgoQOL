"""Entry point for the algo-qol Typer CLI."""
from __future__ import annotations

from ._cli_wrappers import app as app


def main() -> None:
    """Invoke the Typer CLI application."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
