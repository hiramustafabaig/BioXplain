"""Feature-set stability measures.

Definitions were checked against the primary sources (see docs/methodology_decisions.md D14):

* Nogueira, Sechidis & Brown (2018), JMLR 18(174):1-54, Definition 4 (Eq. 2):
      Phi = 1 - [ (1/d) * sum_f s_f^2 ] / [ (kbar/d) * (1 - kbar/d) ],   s_f^2 = M/(M-1) * p_f * (1 - p_f)
  M = number of feature sets, d = total number of features, p_f = fraction of sets containing f,
  kbar = mean set size. Maximum 1 (iff all sets identical); minimum -1/(M-1); expectation 0 under random
  selection; undefined when kbar is 0 or d.
* Kuncheva (2007) consistency index for two sets of equal size k with intersection r:
      (r - k^2/d) / (k - k^2/d),   averaged over all pairs.  Range [-1, 1].  (Equals Phi for constant k,
  Nogueira et al. Theorem 5.)
* Jaccard (Kalousis et al. 2005): |A n B| / |A u B| averaged over all pairs. Not chance-corrected.

The asymptotic confidence interval of Nogueira et al. (Theorem 7) assumes independently drawn feature sets
and is NOT valid for overlapping cross-validation training sets; none is provided here on purpose.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Collection, Hashable, Sequence

FeatureSet = Collection[Hashable]


def _as_sets(sets: Sequence[FeatureSet]) -> list[frozenset]:
    out = [frozenset(s) for s in sets]
    if len(out) < 2:
        raise ValueError("stability needs at least two feature sets")
    return out


def _check_universe(sets: list[frozenset], n_features: int) -> None:
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    seen = set().union(*sets)
    if len(seen) > n_features:
        raise ValueError(f"{len(seen)} distinct features selected but the universe has only {n_features}")


def nogueira_stability(sets: Sequence[FeatureSet], n_features: int) -> float:
    """Nogueira et al. (2018) stability estimator Phi (chance-corrected; ~0 for random selection, 1 = identical)."""
    fsets = _as_sets(sets)
    _check_universe(fsets, n_features)
    m, d = len(fsets), n_features
    kbar = sum(len(s) for s in fsets) / m
    if kbar <= 0 or kbar >= d:
        raise ValueError("Phi is undefined when the mean set size is 0 or equals the number of features")
    counts = Counter(itertools.chain.from_iterable(fsets))
    # features never selected have p_f = 0 and contribute 0 to the sum of variances.
    # math.fsum is exactly rounded, so the result does not depend on set/dict iteration order
    # (which varies with string-hash randomisation); a plain sum() differed in the last bits between runs.
    sum_var = math.fsum((m / (m - 1)) * (c / m) * (1 - c / m) for c in counts.values())
    return 1.0 - (sum_var / d) / ((kbar / d) * (1 - kbar / d))


def kuncheva_index(sets: Sequence[FeatureSet], n_features: int) -> float:
    """Mean pairwise Kuncheva consistency index; all sets must have the same size 0 < k < n_features."""
    fsets = _as_sets(sets)
    _check_universe(fsets, n_features)
    sizes = {len(s) for s in fsets}
    if len(sizes) != 1:
        raise ValueError(f"Kuncheva's index requires equal-size sets, got sizes {sorted(sizes)}")
    k, d = sizes.pop(), n_features
    if k == 0 or k == d:
        raise ValueError("Kuncheva's index is undefined for k = 0 or k = n_features")
    vals = [(len(a & b) * d - k * k) / (k * (d - k)) for a, b in itertools.combinations(fsets, 2)]
    return sum(vals) / len(vals)


def jaccard_stability(sets: Sequence[FeatureSet]) -> float:
    """Mean pairwise Jaccard similarity; empty sets are rejected (top-k sets are never empty)."""
    fsets = _as_sets(sets)
    if any(len(s) == 0 for s in fsets):
        raise ValueError("Jaccard is undefined for empty feature sets")
    vals = [len(a & b) / len(a | b) for a, b in itertools.combinations(fsets, 2)]
    return sum(vals) / len(vals)
