"""The settings file: model/config.yaml, or one given with --config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parent / "config.yaml"


def load(path: str | Path | None = None) -> dict:
    """Read a settings file. It must have the sections and keys of config.yaml."""
    path = Path(path or DEFAULT)
    cfg = yaml.safe_load(path.read_text())
    if DEFAULT.exists() and path.resolve() != DEFAULT:
        ref = yaml.safe_load(DEFAULT.read_text())
        for section, keys in ref.items():
            have = set(cfg.get(section) or {})
            if have != set(keys):
                raise ValueError(
                    f"{path} [{section}]: missing {sorted(set(keys) - have)}, "
                    f"unknown {sorted(have - set(keys))}"
                )
    return cfg


def add_argument(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--config", type=Path, default=DEFAULT, help="settings file (default: %(default)s)"
    )
