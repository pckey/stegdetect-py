"""
Stegcompare - Compare original and steganographic JPEG images.

Usage:
    stegcompare <original.jpg> <stego.jpg>

Shows DCT coefficient differences between the two images.
"""

import argparse
import sys
from pathlib import Path

from .jpeg_dct import read_jpeg


def compare_images(orig_path: str, stego_path: str, verbose: bool = False):
    """Compare DCT coefficients of two JPEG images."""
    orig = read_jpeg(orig_path)
    stego = read_jpeg(stego_path)

    if len(orig.components) != len(stego.components):
        print("Error: images have different number of components", file=sys.stderr)
        return

    total_diff = 0
    total_coeffs = 0
    diff_hist = {}  # value -> count

    for ci in range(len(orig.components)):
        oc = orig.components[ci]
        sc = stego.components[ci]

        if oc.blocks_x != sc.blocks_x or oc.blocks_y != sc.blocks_y:
            print(f"Error: component {ci} has different dimensions", file=sys.stderr)
            continue

        for by in range(oc.blocks_y):
            for bx in range(oc.blocks_x):
                for idx in range(64):
                    ov = oc.coeffs[by, bx, idx]
                    sv = sc.coeffs[by, bx, idx]
                    if ov != sv:
                        diff = sv - ov
                        total_diff += 1
                        diff_hist[diff] = diff_hist.get(diff, 0) + 1
                    total_coeffs += 1

    print(f"Total coefficients: {total_coeffs}")
    print(f"Changed coefficients: {total_diff} ({100.0*total_diff/total_coeffs:.2f}%)")

    if verbose and diff_hist:
        print("\nDifference histogram:")
        for diff in sorted(diff_hist.keys()):
            print(f"  {diff:+4d}: {diff_hist[diff]}")


def main():
    parser = argparse.ArgumentParser(
        prog='stegcompare',
        description='Compare original and steganographic JPEG images'
    )
    parser.add_argument('original', help='Original JPEG file')
    parser.add_argument('stego', help='Steganographic JPEG file')
    parser.add_argument('-v', action='store_true', help='Verbose output (show histogram)')

    args = parser.parse_args()

    if not Path(args.original).exists():
        print(f"stegcompare: {args.original}: No such file", file=sys.stderr)
        sys.exit(1)
    if not Path(args.stego).exists():
        print(f"stegcompare: {args.stego}: No such file", file=sys.stderr)
        sys.exit(1)

    compare_images(args.original, args.stego, verbose=args.v)


if __name__ == '__main__':
    main()
