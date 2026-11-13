# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""Run with `python inference.py` after installing the project environment.

The installed `relayvsr` command is the same entry point.
"""

from refvsr_inference.cli import main

if __name__ == "__main__":
    main()
