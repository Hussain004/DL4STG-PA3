"""
Question 1
NumPy verification of the constructions in parts (a), (b) and (c)

Prints a summary and writes the numbers quoted in the report to
q1/results/verify_encodings.json
"""
import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results" / "verify_encodings.json"

# all trend-count triples (a, b, c) with 1 <= a + b + c <= K
# (a = slowing, b = steady, c = speeding up)
def all_triples(K):
    for a in range(K + 1):
        for b in range(K + 1 - a):
            for c in range(K + 1 - a - b):
                if a + b + c >= 1:
                    yield (a, b, c)

TRIPLES_8 = list(all_triples(8))
print(f"K = 8: {len(TRIPLES_8)} non-empty triples (formula: C(11,3) - 1 = {math.comb(11, 3) - 1})")

# part (a): an affine message collapses the neighbourhood to (n_v, S_v)
N1, N2 = np.array([-1, 0, 1]), np.array([0, 0, 0])
assert N1.shape == N2.shape
assert N1.sum() == N2.sum() == 0          # equal total trend S_v
assert len(N1) == len(N2) == 3            # equal neighbourhood size n_v
print("Part (a): N1 = (-1, 0, +1) and N2 = (0, 0, 0)  =>  n_v = 3, S_v = 0 for both; "
      "steady counts differ (1 vs 3)")
print("          for every (w, c): "
      f"sum_phi(N1) = {2 * N1.sum() + 1 * len(N1)}, sum_phi(N2) = {2 * N2.sum() + 1 * len(N2)} "
      "(with w = 2, c = 1)")
assert 2 * N1.sum() + 1 * len(N1) == 2 * N2.sum() + 1 * len(N2)

# part (b): base-(K + 1) place-value encoding
def decode(m, B):
    return m % B, (m // B) % B, m // (B * B)

def check_encoding(K):
    B = K + 1
    phi = (1, B, B * B)           # phi(-1), phi(0), phi(+1)
    bad = []
    for (a, b, c) in all_triples(K):
        m = a * phi[0] + b * phi[1] + c * phi[2]
        if decode(m, B) != (a, b, c):
            bad.append((a, b, c, m))
    return bad

for K in range(1, 17):
    bad = check_encoding(K)
    status = "ok" if not bad else f"FAILS on {len(bad)} triples, e.g. {bad[0]}"
    print(f"Part (b): K = {K:2d}, B = {K + 1:2d}, triples = {math.comb(K + 3, 3) - 1:6d}  -> {status}")

# K = 8 in detail
K, B = 8, 9
print(f"Part (b): K = 8 example, m = a + 9 b + 81 c, max m = {8 * 1 + 8 * 9 + 8 * 81} (< 9^3 = {9**3})")
assert all(decode(a * 1 + b * 9 + c * 81, 9) == (a, b, c) for (a, b, c) in TRIPLES_8)

# what goes wrong with B = K (a count of K carries into the next digit)?
# concrete carry collision: (8, 0, 0) and (0, 1, 0) both give m = 8 with B = 8
m1, m2 = 8 * 1 + 0 * 8 + 0 * 64, 0 * 1 + 1 * 8 + 0 * 64
print(f"Part (b): with B = K = 8, (a,b,c) = (8,0,0) and (0,1,0) both give m = {m1} == {m2}: carries break decoding")
assert m1 == m2 == 8

# part (b), simpler target: count the non-steady neighbours (-1 or +1)
for (a, b, c) in TRIPLES_8:
    assert a * 1 + b * 0 + c * 1 == a + c
print("Part (b, simpler target): phi(x) = x^2 gives sum = a + c for every triple")

# part (c): one-hot messages, three aggregators
ONE_HOT = {-1: np.array([1, 0, 0]), 0: np.array([0, 1, 0]), 1: np.array([0, 0, 1])}

def aggregate(triple, method):
    a, b, c = triple
    msgs = [ONE_HOT[-1]] * a + [ONE_HOT[0]] * b + [ONE_HOT[1]] * c
    M = np.array(msgs)
    if method == "sum":
        return M.sum(axis=0)
    if method == "mean":
        return M.mean(axis=0)
    return M.max(axis=0)

aggs = {"sum": {}, "mean": {}, "max": {}}
for t in TRIPLES_8:
    for method in aggs:
        key = tuple(aggregate(t, method))
        aggs[method].setdefault(key, []).append(t)

print(f"Part (c): distinct aggregated vectors over the 164 triples — "
      f"sum: {len(aggs['sum'])}, mean: {len(aggs['mean'])}, max: {len(aggs['max'])}")
assert len(aggs["sum"]) == len(TRIPLES_8)      # sum is injective on triples
assert len(aggs["max"]) == 7          # all non-empty presence patterns in {0,1}^3

# task (i) counterexample, reused: {{0}} vs {{0, 0}}
t_i_1, t_i_2 = (0, 1, 0), (0, 2, 0)
print(f"Part (c) (i):  N1 = {{0}} -> mean {aggregate(t_i_1, 'mean')}, max {aggregate(t_i_1, 'max')}; "
      f"N2 = {{0,0}} -> mean {aggregate(t_i_2, 'mean')}, max {aggregate(t_i_2, 'max')}")
assert tuple(aggregate(t_i_1, 'mean')) == tuple(aggregate(t_i_2, 'mean'))
assert tuple(aggregate(t_i_1, 'max')) == tuple(aggregate(t_i_2, 'max'))

# task (ii) counterexample for max: {-1, -1, 0} (majority slowing) vs {-1, 0, 0}
t_ii_1, t_ii_2 = (2, 1, 0), (1, 2, 0)
print(f"Part (c) (ii): N1 = {{-1,-1,0}} -> max {aggregate(t_ii_1, 'max')}, majority slowing: yes; "
      f"N2 = {{-1,0,0}} -> max {aggregate(t_ii_2, 'max')}, majority slowing: no")
assert tuple(aggregate(t_ii_1, 'max')) == tuple(aggregate(t_ii_2, 'max'))
print("          sum test 2a > n: a - b - c > 0 ->",
      f"N1: {t_ii_1[0] - t_ii_1[1] - t_ii_1[2]} > 0 (yes), N2: {t_ii_2[0] - t_ii_2[1] - t_ii_2[2]} > 0 (no)")
print("          mean test a/n > 1/2 ->",
      f"N1: {t_ii_1[0] / 3:.3f} > 0.5 (yes), N2: {t_ii_2[0] / 3:.3f} > 0.5 (no)")

# task (iii): all three methods decide it (a > 0)
for method in ("sum", "mean", "max"):
    assert all((aggregate(t, method)[0] > 0) == (t[0] > 0) for t in TRIPLES_8)
print("Part (c) (iii): sum/mean/max first coordinate > 0 exactly when a > 0 (checked on all 164 triples)")

# task (ii): sum test a > n/2, mean test a/n > 1/2 (integer arithmetic as reference)
for t in TRIPLES_8:
    assert (aggregate(t, "sum")[0] > sum(t) / 2) == (2 * t[0] > sum(t))
    assert (aggregate(t, "mean")[0] > 0.5) == (2 * t[0] > sum(t))
print("Part (c) (ii):  sum test a > n/2, mean test a/n > 1/2 — both agree with 2a > n (all 164 triples)")

out = {
    "K8_triples": len(TRIPLES_8),
    "distinct_sum_vectors": len(aggs["sum"]),
    "distinct_mean_vectors": len(aggs["mean"]),
    "distinct_max_vectors": len(aggs["max"]),   # 7: every pattern except (0,0,0), impossible for a non-empty neighbourhood
    "part_b_all_K_1_to_16": "all triples decode correctly",
}
RESULTS.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS, "w") as f:
    json.dump(out, f, indent=2)
print("Wrote q1/results/verify_encodings.json")
