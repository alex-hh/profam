"""Unified ``profam`` CLI — thin dispatcher to subcommands."""

import sys


def app():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(
            "usage: profam <command> [options]\n"
            "\n"
            "ProFam — protein family language model toolkit for fitness prediction and design.\n"
            "\n"
            "commands:\n"
            "  generate   Generate sequences from family prompts\n"
            "  score      Score candidate sequences with family context\n"
            "  download   Download the pretrained ProFam-1 model weights\n"
            "\n"
            "Run `profam <command> --help` for full options."
        )
        raise SystemExit(0)

    command, rest = args[0], args[1:]

    if command == "generate":
        from profam.cli.generate_sequences import main as _main

        raise SystemExit(_main(rest or ["--help"]))
    elif command == "score":
        from profam.cli.score_sequences import main as _main

        raise SystemExit(_main(rest or ["--help"]))
    elif command == "download":
        from profam.download_checkpoint import main as _main

        _main()
    else:
        print(f"Unknown command: {command}")
        print("Run `profam --help` for available commands.")
        raise SystemExit(1)


if __name__ == "__main__":
    app()
