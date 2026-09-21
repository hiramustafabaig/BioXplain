"""Model factory. Configurations are fixed and pre-specified (decision D10/D11); nothing is tuned here."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

IMPLEMENTED = ("logreg",)


def make_model(spec: dict[str, Any], seed: int):
    """Build an unfitted estimator from a config ``spec`` (``{"name": ..., **params}``)."""
    spec = dict(spec)
    name = spec.pop("name")
    if name == "logreg":
        # sklearn's default penalty is L2; we do not pass `penalty` (deprecated in recent scikit-learn).
        return LogisticRegression(
            C=spec.pop("C", 1.0),
            class_weight=spec.pop("class_weight", "balanced"),
            solver="lbfgs",
            max_iter=spec.pop("max_iter", 5000),
            random_state=seed,
        )
    raise NotImplementedError(f"model {name!r} is not implemented yet (implemented: {IMPLEMENTED})")
