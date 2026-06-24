#!/usr/bin/env python3
"""Stegdeimage entry point.

Run directly without installation:

    uv run stegdeimage.py stego.jpg output.jpg
"""

from stegdetect.stegdeimage_cli import main

if __name__ == "__main__":
    main()
