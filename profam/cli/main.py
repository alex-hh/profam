"""Unified ``profam`` CLI built with Typer."""

import sys

import typer

app = typer.Typer(
    name="profam",
    help="ProFam — protein family language model toolkit for fitness prediction and design.",
    add_completion=False,
)

_forward_ctx = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


@app.command(
    context_settings=_forward_ctx,
    help="Generate sequences from family prompts. Use `profam generate -- --help` for full options.",
)
def generate(ctx: typer.Context):
    from profam.cli.generate_sequences import main as _gen_main

    raise SystemExit(_gen_main(ctx.args or ["--help"]))


@app.command(
    context_settings=_forward_ctx,
    help="Score candidate sequences with family context. Use `profam score -- --help` for full options.",
)
def score(ctx: typer.Context):
    from profam.cli.score_sequences import main as _score_main

    raise SystemExit(_score_main(ctx.args or ["--help"]))


@app.command(
    context_settings=_forward_ctx,
    help="Download the pretrained ProFam-1 model weights.",
)
def download(ctx: typer.Context):
    from profam.download_checkpoint import main as _dl_main

    _dl_main()


if __name__ == "__main__":
    app()
