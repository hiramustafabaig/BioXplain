"""Stability metrics validated against toy cases, hand calculation, published reference values and theory."""
import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

from bioxplain.stability.metrics import jaccard_stability, kuncheva_index, nogueira_stability


def test_identical_sets_give_one():
    s = [{1, 2, 3}] * 4
    assert nogueira_stability(s, 10) == pytest.approx(1.0)
    assert kuncheva_index(s, 10) == pytest.approx(1.0)
    assert jaccard_stability(s) == 1.0


def test_stabm_documentation_reference_values():
    """stabm docs: features 1:3, 1:4, 1:5 with p=10 -> Nogueira 0.7222222, Jaccard 0.7166667 (R docs, not re-run here)."""
    feats = [set(range(1, 4)), set(range(1, 5)), set(range(1, 6))]
    assert nogueira_stability(feats, 10) == pytest.approx(0.7222222, abs=1e-7)
    assert jaccard_stability(feats) == pytest.approx(0.7166667, abs=1e-7)


def test_nogueira_hand_calculation():
    # M=3, d=10; f4: p=2/3, f5: p=1/3 -> s^2 = 1/3 each; sum = 2/3; kbar = 4
    # Phi = 1 - (2/3 / 10) / (0.4 * 0.6) = 1 - 0.0666667/0.24
    assert nogueira_stability([{1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5}], 10) == pytest.approx(1 - (2 / 3 / 10) / 0.24)


def test_complementary_sets_reach_the_theoretical_minimum():
    # M=2, d=2k, disjoint sets: Phi = -1/(M-1) = -1 (Nogueira et al. 2018, Appendix D)
    assert nogueira_stability([{1, 2, 3}, {4, 5, 6}], 6) == pytest.approx(-1.0)
    assert kuncheva_index([{1, 2, 3}, {4, 5, 6}], 6) == pytest.approx(-1.0)
    assert jaccard_stability([{1, 2, 3}, {4, 5, 6}]) == 0.0


def test_random_selection_is_near_zero_but_jaccard_is_not_chance_corrected():
    rng = np.random.default_rng(0)
    d, k, m = 1000, 25, 10
    phis, jacs = [], []
    for _ in range(200):
        sets = [set(rng.choice(d, k, replace=False).tolist()) for _ in range(m)]
        phis.append(nogueira_stability(sets, d))
        jacs.append(jaccard_stability(sets))
    assert abs(np.mean(phis)) < 0.01                       # E[Phi | H0] = 0
    assert np.mean(jacs) > 0.005                           # raw overlap is above 0 by chance and grows with k/d


def test_theorem5_nogueira_equals_mean_pairwise_kuncheva_for_constant_size():
    rng = np.random.default_rng(1)
    for _ in range(25):
        d, k, m = int(rng.integers(30, 200)), int(rng.integers(3, 20)), int(rng.integers(2, 9))
        sets = [set(rng.choice(d, k, replace=False).tolist()) for _ in range(m)]
        assert nogueira_stability(sets, d) == pytest.approx(kuncheva_index(sets, d), abs=1e-12)


def test_more_overlap_is_more_stable():
    d = 500
    base = set(range(25))
    lo = [set(range(i * 25, i * 25 + 25)) for i in range(5)]                                   # disjoint
    hi = [(base - set(range(j))) | set(range(100 + j, 100 + 2 * j)) for j in range(1, 6)]       # high overlap
    assert nogueira_stability(hi, d) > nogueira_stability(lo, d)


def test_dependence_on_k_for_fixed_relative_overlap():
    d = 1000
    # sets range(k) and range(k//2, k//2+k): intersection r = k - k//2 (5, 13, 25); hand-computed (r*d - k^2)/(k*(d-k))
    hand = {10: (5 * 1000 - 100) / (10 * 990), 25: (13 * 1000 - 625) / (25 * 975), 50: (25 * 1000 - 2500) / (50 * 950)}
    assert hand[10] == pytest.approx(4900 / 9900) and hand[25] == pytest.approx(12375 / 24375)
    for k, expected in hand.items():
        a, b = set(range(k)), set(range(k // 2, k // 2 + k))
        assert len(a & b) == k - k // 2
        assert kuncheva_index([a, b], d) == pytest.approx(expected)


@pytest.mark.parametrize("call", [
    lambda: nogueira_stability([{1}], 10),                      # need >= 2 sets
    lambda: nogueira_stability([set(), set()], 10),             # kbar = 0
    lambda: nogueira_stability([set(range(3)), set(range(3))], 3),   # kbar = d
    lambda: nogueira_stability([{1, 2}, {3, 4}], 3),            # more distinct features than the universe
    lambda: nogueira_stability([{1, 2}, {1, 3}], 0),
    lambda: kuncheva_index([{1, 2}, {1, 2, 3}], 10),            # unequal sizes
    lambda: kuncheva_index([set(), set()], 10),
    lambda: jaccard_stability([set(), {1}]),
])
def test_edge_cases_raise(call):
    with pytest.raises(ValueError):
        call()


SCRIPT = """
import random
from bioxplain.stability.metrics import nogueira_stability
rnd = random.Random(7)
pool = rnd.sample([f'GENE{i}' for i in range(20848)], 60)
sets = [set(rnd.sample(pool, 25)) for _ in range(10)]
print(repr(nogueira_stability(sets, 20848)))
"""


def test_result_is_bit_identical_across_python_hash_seeds():
    """Regression for a real bug found on GSE42568 results: a plain sum() over a Counter of a frozenset union
    of string features changed in the last bits between runs, because string-hash randomisation changes the
    iteration order and float addition is not associative. With the old implementation this scenario gave 4
    different values over 8 PYTHONHASHSEEDs; math.fsum makes it exactly reproducible."""
    outputs = set()
    for seed in range(8):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        res = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=True)
        outputs.add(res.stdout.strip())
    assert len(outputs) == 1, outputs


def test_order_of_sets_does_not_matter():
    sets = [{1, 2, 3}, {2, 3, 4}, {1, 3, 5}, {7, 8, 9}]
    ref = nogueira_stability(sets, 20)
    for perm in itertools.islice(itertools.permutations(sets), 6):
        assert nogueira_stability(list(perm), 20) == pytest.approx(ref)
