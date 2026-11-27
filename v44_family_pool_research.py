"""Offline V44 research: chronological comparison of family-weighted forecasts.

Inputs must be independently authenticated by the caller. No provider calls,
live forecasts, archive writes, or model promotion. Each family has one vote,
regardless of its number of correlated variants.
"""
from datetime import datetime
import math
import numpy as np

MIN_TRAIN_PAIRS = 30
PRIOR_STRENGTH = 20.0


def _time(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.utcoffset() is None:
        raise ValueError("timezone required")
    return t


def _grid(raw, shape=None):
    p = np.asarray(raw, dtype=float)
    if (p.ndim != 2 or min(p.shape) < 5 or not np.isfinite(p).all()
            or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0)
            or (shape is not None and p.shape != shape)):
        raise ValueError("invalid score distribution")
    return p / p.sum()


def family_pool(families):
    """Average variant grids within each declared family, separately per leg."""
    if not families:
        raise ValueError("no model families")
    out = {}
    shape = None
    for name in sorted(families):
        variants = families[name]
        if not variants:
            raise ValueError("empty model family")
        matrices = []
        for pair in variants.values():
            if len(pair) != 2:
                raise ValueError("two legs required")
            legs = []
            for raw in pair:
                p = _grid(raw, shape)
                shape = p.shape
                legs.append(p)
            matrices.append(legs)
        out[name] = np.mean(matrices, axis=0)
    return out


def _validate(row):
    decision, kickoff, known = map(_time, (row["decision_utc"],
                                          row["earliest_kickoff_utc"],
                                          row["finals_known_utc"]))
    if not decision < kickoff <= known or row.get("verified") is not True:
        raise ValueError("unverified or invalid time order")
    ids = row["fixture_ids"]
    if (len(ids) != 2 or len(set(ids)) != 2
            or any(not isinstance(i, str) or not i for i in ids)):
        raise ValueError("distinct canonical fixture IDs required")
    pooled = family_pool(row["families"])
    names = tuple(pooled)
    scores = row["actual"]
    if len(scores) != 2:
        raise ValueError("two final scores required")
    for score in scores:
        if (len(score) != 2 or any(type(v) is not int or v < 0 for v in score)):
            raise ValueError("invalid final score")
    grids = np.stack(list(pooled.values()))
    return decision, known, tuple(ids), names, grids, tuple(tuple(s) for s in scores)


def _likelihood(grids, scores):
    # Out-of-grid actuals are not dropped; the floor penalizes every family.
    return np.array([[p[leg, h, a] if h < p.shape[1] and a < p.shape[2] else 0.
                      for p in grids] for leg, (h, a) in enumerate(scores)])


def _weights(training, n):
    prior = np.full(n, 1 / n)
    if len(training) < MIN_TRAIN_PAIRS:
        return prior
    likelihood = np.maximum(np.concatenate(training, axis=0), 1e-12)
    w = prior.copy()
    # Regularized mixture likelihood, with a fixed symmetric prior.
    for _ in range(100):
        resp = likelihood * w
        updated = (np.sum(resp / resp.sum(axis=1, keepdims=True), axis=0)
                   + PRIOR_STRENGTH * prior) / (len(likelihood) + PRIOR_STRENGTH)
        if np.max(np.abs(updated - w)) < 1e-10:
            return updated
        w = updated
    return w


def _metrics(grids, scores, weights):
    mixed = np.tensordot(weights, grids, axes=1)
    joint = np.outer(mixed[0].ravel(), mixed[1].ravel())
    top = np.argsort(-joint.ravel(), kind="stable")[:5]
    selected = set(map(int, top))
    shape = mixed[0].shape
    observed = []
    for h, a in scores:
        observed.append(h * shape[1] + a if h < shape[0] and a < shape[1] else None)
    hit = (None not in observed and observed[0] * joint.shape[1] + observed[1] in selected)
    likelihood = _likelihood(grids, scores) @ weights
    return {"hit": bool(hit), "coverage": float(joint.ravel()[top].sum()),
            "score_log_loss": float(-np.log(np.maximum(likelihood, 1e-15)).mean())}


def walk_forward(rows):
    """Compare learned and equal family weights on the same verified pairs.

    Require a consistent declared roster; separate data-availability cohorts
    instead of silently changing weights. Only finals known strictly before
    a forecast decision can train its weights. No repeated canonical fixture.
    """
    records = sorted((_validate(row) for row in rows), key=lambda r: (r[0], r[2]))
    seen, history, results = set(), [], []
    roster = records[0][3] if records else ()
    for decision, known, ids, names, grids, scores in records:
        if names != roster or seen.intersection(ids):
            raise ValueError("inconsistent family roster or repeated fixture")
        seen.update(ids)
        prior = [lik for available, lik in history if available < decision]
        weights = _weights(prior, len(names))
        baseline = np.full(len(names), 1 / len(names))
        results.append({"fixture_ids": ids, "training_pairs": len(prior),
                        "weights": dict(zip(names, map(float, weights))),
                        "learned": _metrics(grids, scores, weights),
                        "equal": _metrics(grids, scores, baseline)})
        history.append((known, _likelihood(grids, scores)))
    # Warm-up pairs are reported but excluded from the learned-weight comparison.
    evaluated = [r for r in results if r["training_pairs"] >= MIN_TRAIN_PAIRS]
    summary = {}
    for name in ("learned", "equal"):
        summary[name] = {"pairs": len(evaluated),
                         "hits": sum(r[name]["hit"] for r in evaluated),
                         "hit_rate": (sum(r[name]["hit"] for r in evaluated) / len(evaluated)
                                      if evaluated else None),
                         "mean_log_loss": (math.fsum(r[name]["score_log_loss"] for r in evaluated)
                                           / len(evaluated) if evaluated else None)}
    return {"status": "research_only", "automatic_promotion": False,
            "summary": summary, "rows": results}
