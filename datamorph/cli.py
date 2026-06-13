"""Command-line interface for datamorph.

    datamorph convert input.csv output.json
    datamorph convert log.txt --output-format csv > out.csv
    datamorph --version

Exit codes: 0 = converted and validated, 1 = ran but output failed validation,
2 = usage / input error.
"""

from __future__ import annotations

import argparse
import sys

from datamorph import __version__, convert_file

FORMATS = ("csv", "json", "txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datamorph",
        description="Convert files between CSV, JSON, and TXT with the distilled student model.",
    )
    parser.add_argument("--version", action="version", version=f"datamorph {__version__}")
    sub = parser.add_subparsers(dest="command")

    conv = sub.add_parser("convert", help="convert an input file to another format")
    conv.add_argument("input", help="path to the source file")
    conv.add_argument(
        "output",
        nargs="?",
        help="path to write (its extension sets the target format); "
        "if omitted, the result is printed to stdout and --output-format is required",
    )
    conv.add_argument("--input-format", choices=FORMATS, help="override input format detection")
    conv.add_argument("--output-format", choices=FORMATS, help="target format (required if no output path)")
    conv.add_argument("--instruction", help="extra natural-language guidance for the conversion")
    conv.add_argument("--max-retries", type=int, default=3, help="retries with error feedback (default 3)")
    conv.add_argument("--model", help="local model path or HF repo id (default: the published model)")
    conv.add_argument("-q", "--quiet", action="store_true", help="suppress the status line on stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "convert":
        parser.print_help(sys.stderr)
        return 2

    try:
        result = convert_file(
            args.input,
            args.output,
            input_format=args.input_format,
            output_format=args.output_format,
            instruction=args.instruction,
            max_retries=args.max_retries,
            model=args.model,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output is None:
        sys.stdout.write(result.output_text)

    if not args.quiet:
        where = str(result.output_path) if result.output_path else "stdout"
        status = "ok" if result.accepted else f"NOT VALIDATED ({result.error or 'low score'})"
        print(
            f"datamorph: {result.input_format} -> {result.output_format} {status} "
            f"(retries={result.retries}, scores={result.scores}) -> {where}",
            file=sys.stderr,
        )

    return 0 if result.accepted else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
