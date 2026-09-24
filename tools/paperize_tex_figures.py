#!/usr/bin/env python3
"""
Switch LaTeX figure references from PNG to vector PDF.

Only paths inside \\includegraphics and \\figmaybe commands are changed.
A .bak copy is written before modification.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys


PATTERN = re.compile(
    r"(\\(?:includegraphics|figmaybe)(?:\[[^\]]*\])?\{[^{}]*?)\.png(\})"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python tools/paperize_tex_figures.py path/to/file.tex"
        )

    path = Path(sys.argv[1]).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)

    updated, n = PATTERN.subn(r"\1.pdf\2", text)

    path.write_text(updated)

    print(f"Updated {n} figure reference(s)")
    print(f"Backup: {backup}")
    print(f"Written: {path}")


if __name__ == "__main__":
    main()
