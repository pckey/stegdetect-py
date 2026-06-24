#!/usr/bin/env python3
"""Stegcompare entry point.

Run directly without installation:

    uv run stegcompare.py original.jpg stego.jpg
"""

from stegdetect.stegcompare_cli import main

if __name__ == "__main__":
    main()
