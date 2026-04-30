"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
import uvicorn

DEFAULT_CONFIG = """\
# fakellm.yaml — see https://github.com/yourname/fakellm for docs
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
def serve(host: str, port: int, config: str, reload: bool) -> None:
    """Start the mock server."""
    import os

    os.environ["LLMOCK_CONFIG"] = config
    click.echo(f"fakellm serving on http://{host}:{port}")
    click.echo(f"  OpenAI:    {host}:{port}/v1/chat/completions")
    click.echo(f"  Anthropic: {host}:{port}/v1/messages")
    click.echo(f"  Dashboard: http://{host}:{port}/_fakellm")
    uvicorn.run("fakellm.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
