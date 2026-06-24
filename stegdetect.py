#!/usr/bin/env python3
"""Stegdetect entry point.

Run directly without installation:

    uv run stegdetect.py image.jpg
"""

from stegdetect.cli import main

if __name__ == "__main__":
    main()
