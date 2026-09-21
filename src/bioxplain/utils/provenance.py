"""Provenance helpers: file hashes, git state, software versions, config hashing."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import platform
import subprocess
from importlib import metadata
from typing import Any

_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "shap", "xgboost", "gseapy")


class DataIntegrityError(RuntimeError):
    """Raised when a data file does not match its recorded SHA-256."""


def sha256_file(path: str | pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_file(path: str | pathlib.Path, expected_sha256: str) -> None:
    """Raise DataIntegrityError unless the file's SHA-256 equals ``expected_sha256``."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise DataIntegrityError(f"{path}: sha256 {actual} != expected {expected_sha256}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def git_state(repo_root: str | pathlib.Path) -> dict[str, Any]:
    """Return commit hash, branch and whether tracked files are modified. Never raises."""

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=30).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD") or "unknown",
            "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except Exception:  # noqa: BLE001 - provenance must not break a run
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def software_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out
