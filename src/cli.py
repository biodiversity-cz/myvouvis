"""CLI smoke test: python -m cli path/to/sheet.jpg"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m cli` from repo with src on path
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.sheet import process_sheet  # noqa: E402
from pipeline.types import OutputMode  # noqa: E402
from settings import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Process one herbarium sheet → DwC JSON")
    parser.add_argument("image", type=Path, help="Path to sheet image")
    parser.add_argument(
        "--output",
        choices=[m.value for m in OutputMode],
        default=get_settings().output_mode.value,
        help="Response shape: full (default), dwc, or bbox",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        print(f"File not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    mode = OutputMode(args.output)
    result = process_sheet(args.image, output_mode=mode)
    print(json.dumps(result.as_score(mode), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
