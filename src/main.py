"""
Development entrypoint for the modular code under `src/`.

OpenWebUI requires a single-file pipe; generate it with `python build_pipe.py`.
"""

from src.pipe_impl import Pipe, name

__all__ = ["Pipe", "name"]

