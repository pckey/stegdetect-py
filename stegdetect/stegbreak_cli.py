"""
Stegbreak command-line interface.

Usage:
    stegbreak [options] <jpeg_file> [<jpeg_file> ...]

Options:
    -f <wordlist>  Password wordlist file (required)
    -t <type>      Stego type: jphide, jsteg, outguess, auto (default: auto)
    -v             Verbose output
"""

import argparse
import sys
from pathlib import Path

from .stegbreak import crack_file


def main():
    parser = argparse.ArgumentParser(
        prog='stegbreak',
        description='Dictionary attack for cracking steganography passwords'
    )
    parser.add_argument('files', nargs='+', help='JPEG file(s) to crack')
    parser.add_argument('-f', type=str, required=True, help='Password wordlist file')
    parser.add_argument('-t', type=str, default='auto',
                       choices=['auto', 'jphide', 'jsteg', 'outguess'],
                       help='Steganography type (default: auto)')
    parser.add_argument('-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if not Path(args.f).exists():
        print(f"stegbreak: {args.f}: No such file or directory", file=sys.stderr)
        sys.exit(1)

    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"stegbreak: {filepath}: No such file or directory", file=sys.stderr)
            continue

        if args.v:
            print(f"Cracking {filepath}...", file=sys.stderr)

        result = crack_file(filepath, args.f, stego_type=args.t)

        if result and result.success:
            print(f"{filepath}: password=\"{result.password}\" ({result.stego_type}, {result.details})")
        elif args.v:
            print(f"{filepath}: no password found", file=sys.stderr)


if __name__ == '__main__':
    main()
