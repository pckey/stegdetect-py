"""
Stegdetect command-line interface.

Usage:
    stegdetect [options] <jpeg_file> [<jpeg_file> ...]

Options:
    -t <type>     Detection type: j=JSteg, o=Outguess, p=JPHide, i=Invisible, f=F5, F=F5(slow), h=Steghide
                  Can combine: e.g., -tjopfiFh for all
    -s <scale>    Sensitivity scale factor (default: 1.0)
    -n            Suppress summary output
    -v            Verbose output
    -q <file>     Use results database
    -d <file>     Write results to database
    -r            Use rules for additional checks
    -c            Check for class discrimination
"""

import argparse
import sys
from pathlib import Path

from .detectors import (
    detect, FLAG_JSTEG, FLAG_OUTGUESS, FLAG_JPHIDE, FLAG_INVIS, FLAG_F5, FLAG_F5_SLOW,
    FLAG_STEGHIDE, FLAG_ALL, DetectionResult
)


def _format_result(filename: str, result: DetectionResult, verbose: bool = False) -> str:
    """Format a detection result for output."""
    type_names = {
        "jsteg": "jsteg",
        "outguess": "outguess",
        "jphide": "jphide",
        "invis": "invisible",
        "f5": "f5",
        "steghide": "steghide",
    }

    name = type_names.get(result.stego_type, result.stego_type)
    conf = result.confidence

    if conf >= 1.0:
        certainty = "definitely"
    elif conf >= 0.5:
        certainty = "probably"
    elif conf >= 0.1:
        certainty = "possibly"
    else:
        certainty = ""

    if certainty:
        line = f"{filename}: \"{name}\" ({certainty}"
        if verbose and result.estimated_bits:
            line += f", {result.estimated_bits} bits"
        if verbose and result.details:
            line += f", {result.details}"
        line += ")"
        return line

    return ""


def _parse_type_flag(type_str: str) -> int:
    """Parse -t flag string to detection flags."""
    flags = 0
    type_map = {
        'j': FLAG_JSTEG,
        'o': FLAG_OUTGUESS,
        'p': FLAG_JPHIDE,
        'i': FLAG_INVIS,
        'f': FLAG_F5,
        'F': FLAG_F5_SLOW,
        'h': FLAG_STEGHIDE,
    }
    for ch in type_str:
        if ch in type_map:
            flags |= type_map[ch]
    return flags if flags else FLAG_ALL


def main():
    parser = argparse.ArgumentParser(
        prog='stegdetect',
        description='Automated tool for detecting steganographic content in JPEG images',
        epilog='Detection types: j=JSteg, o=Outguess, p=JPHide, i=Invisible, f=F5, F=F5(slow), h=Steghide'
    )
    parser.add_argument('files', nargs='*', help='JPEG file(s) to analyze')
    parser.add_argument('-t', type=str, default='', help='Detection type(s) to use')
    parser.add_argument('-s', type=float, default=1.0, help='Sensitivity scale factor (default: 1.0)')
    parser.add_argument('-n', action='store_true', help='Suppress summary output')
    parser.add_argument('-v', action='store_true', help='Verbose output')
    parser.add_argument('-V', action='store_true', help='Version')

    args = parser.parse_args()

    if args.V:
        from . import __version__
        print(f"stegdetect (Python rewrite) {__version__}")
        sys.exit(0)

    if not args.files:
        parser.error('the following arguments are required: files')

    flags = _parse_type_flag(args.t)

    total_files = 0
    detected_files = 0

    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"stegdetect: {filepath}: No such file or directory", file=sys.stderr)
            continue

        total_files += 1

        try:
            results = detect(filepath, flags=flags, scale=args.s)
        except Exception as e:
            if args.v:
                print(f"stegdetect: {filepath}: Error: {e}", file=sys.stderr)
            continue

        if results:
            detected_files += 1
            for r in results:
                line = _format_result(filepath, r, verbose=args.v)
                if line:
                    print(line)

    if not args.n and total_files > 1:
        print(f"\n{detected_files} out of {total_files} images tested positive")


if __name__ == '__main__':
    main()
