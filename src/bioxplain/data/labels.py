"""Label derivation with cross-field consistency checks."""
from __future__ import annotations

import pandas as pd


class LabelError(ValueError):
    """Raised when class labels cannot be determined reliably."""


def derive_cancer_normal_labels(samples: pd.DataFrame) -> pd.Series:
    """GSE42568-style labels: 1 = cancer, 0 = normal.

    The ``tissue`` field defines the label; ``source_name``, ``title`` and ``description`` must all agree
    with it, otherwise a LabelError names the disagreeing samples (labels are never guessed).
    """
    required = ["tissue", "source_name", "title", "description"]
    missing = [c for c in required if c not in samples.columns]
    if missing:
        raise LabelError(f"missing metadata columns: {missing}")
    tissue = samples["tissue"].str.strip().str.lower()
    mapping = {"breast cancer": 1, "normal breast": 0}
    unknown = sorted(set(tissue) - set(mapping))
    if unknown:
        raise LabelError(f"unrecognised tissue values: {unknown}")
    y = tissue.map(mapping).astype(int)
    is_cancer = y.eq(1)
    checks = {
        "source_name": samples["source_name"].str.lower().str.contains("cancer"),
        "title": samples["title"].str.lower().str.startswith("breast cancer"),
        "description": samples["description"].str.lower().str.contains("cancer"),
    }
    for name, flag in checks.items():
        bad = samples.index[flag != is_cancer]
        if len(bad):
            raise LabelError(f"'{name}' disagrees with 'tissue' for {len(bad)} samples, e.g. {list(bad[:3])}")
    y.name = "cancer"
    return y
