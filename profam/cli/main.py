"""Unified ``profam`` CLI built with Typer."""

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


@app.command(context_settings=_forward_ctx)
def generate(
    ctx: typer.Context,
):
    """Generate sequences from family prompts.

    All flags are forwarded to the underlying generate-sequences entrypoint.
    Run ``profam generate --help`` for the full list.
    """
    from profam.cli.generate_sequences import main as _gen_main

    raise SystemExit(_gen_main(ctx.args))


@app.command(context_settings=_forward_ctx)
def score(
    ctx: typer.Context,
):
    """Score candidate sequences with family context.

    All flags are forwarded to the underlying score-sequences entrypoint.
    Run ``profam score --help`` for the full list.
    """
    from profam.cli.score_sequences import main as _score_main

    raise SystemExit(_score_main(ctx.args))


@app.command(context_settings=_forward_ctx)
def download(
    ctx: typer.Context,
):
    """Download the pretrained ProFam-1 model weights.

    All flags are forwarded to the underlying download-checkpoint entrypoint.
    Run ``profam download --help`` for the full list.
    """
    from profam.download_checkpoint import main as _dl_main

    _dl_main()


if __name__ == "__main__":
    app()
