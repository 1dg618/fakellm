"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
import uvicorn

DEFAULT_CONFIG = """\
# fakellm.yaml — see https://github.com/1dg618/fakellm for docs
version: 1

defaults:
  fallback: deterministic_echo

rules:
  - name: greeting
    when:
      messages_contain: "hello"
    respond:
      content: "Hi there! This is a mock response from fakellm."

  - name: classifier_demo
    when:
      messages_contain: "classify the sentiment"
    respond:
      content: "positive"

  - name: rate_limit_scenario
    when:
      header.x-test-scenario: rate_limit
    respond:
      status: 429
      error: "Rate limit exceeded (mock)"
"""


@click.group()
@click.version_option()
def main() -> None:
    """fakellm — a mock LLM server for testing."""


@main.command()
def init() -> None:
    """Create a starter fakellm.yaml in the current directory."""
    p = Path("fakellm.yaml")
    if p.exists():
        click.echo("fakellm.yaml already exists. Not overwriting.")
        return
    p.write_text(DEFAULT_CONFIG)
    click.echo("Created fakellm.yaml")
    click.echo("Run `fakellm serve` to start the server.")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=9999, show_default=True, type=int)
@click.option(
    "--config",
    default="fakellm.yaml",
    show_default=True,
    help="Path to config file.",
)
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev only).")
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of uvicorn workers. fakellm's state is per-process; using "
    "more than 1 will partition conversations and stats across workers and "
    "is not supported. Pass --workers N anyway only if you understand this.",
)
def serve(host: str, port: int, config: str, reload: bool, workers: int) -> None:
    """Start the mock server."""
    import os

    config_path = Path(config)
    if not config_path.exists():
        click.echo(
            f"Error: config file not found at '{config}'.\n"
            f"Run `fakellm init` to create a starter config, "
            f"or pass --config <path> to point at an existing one.",
            err=True,
        )
        raise click.exceptions.Exit(1)

    # Validate the config up front so YAML errors surface here, not on the
    # first request. We import lazily to keep `fakellm --help` fast.
    from .config import load_config

    try:
        loaded = load_config(config_path)
    except (ValueError, Exception) as e:
        click.echo(f"Error loading config '{config}': {e}", err=True)
        raise click.exceptions.Exit(1)

    if workers > 1:
        click.echo(
            f"Warning: --workers={workers} requested. fakellm stores "
            f"conversations and stats per-process; with multiple workers "
            f"these will be partitioned and the dashboard will show only "
            f"one worker's view. Continue only if you understand this.",
            err=True,
        )

    os.environ["FAKELLM_CONFIG"] = config
    click.echo(f"fakellm serving on http://{host}:{port}")
    click.echo(f"  Loaded {len(loaded.rules)} rule(s) from {config}")
    click.echo(f"  OpenAI:    {host}:{port}/v1/chat/completions")
    click.echo(f"  Anthropic: {host}:{port}/v1/messages")
    click.echo(f"  Dashboard: http://{host}:{port}/_fakellm")

    # uvicorn.run doesn't accept workers=1 with reload=True; reload only
    # makes sense in single-process dev mode.
    if reload:
        uvicorn.run("fakellm.server:app", host=host, port=port, reload=True)
    else:
        uvicorn.run("fakellm.server:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
