#!/usr/bin/env python3
"""Stegbreak entry point.

Run directly without installation:

    uv run stegbreak.py -f wordlist.txt image.jpg
"""

from stegdetect.stegbreak_cli import main

if __name__ == "__main__":
    main()
