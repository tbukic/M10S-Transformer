"""Project-wide path constants."""

from __future__ import annotations

import os
from pathlib import Path

from attrs import frozen

_root: Path = Path(__file__).resolve().parent.parent.parent


@frozen
class Data:
    root: Path = Path(os.environ.get("APP_DATA_DIR", str(_root / "data")))


@frozen
class ProjectPaths:
    root: Path = _root
    data: Data = Data()


project_paths = ProjectPaths()
