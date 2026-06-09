# -*- coding: utf-8 -*-
"""
concat_for_training.py

Build a DPO-format training file by concatenating a strict file with a budget
of ties drawn from a tie pool, mixed at a target ratio alpha:

    alpha = n_strict / (n_strict + n_ties)

so (1 - alpha) = n_ties / (n_strict + n_ties). alpha = 1.0 is strict-only;
lowering alpha injects more ties. 

USAGE
-----
Alpha-mixed augmented set (pool input, auto-converted):
    python concat_for_training.py \
        --strict hotel_data/seed0/strict.jsonl \
        --ties   hotel_data/seed0/ties_near-tie_decorrelated_mixture.jsonl \
        --alpha  0.5 \
        --output hotel_data/seed0/aug_a0.5.jsonl \
        --seed 0

Strict-only baseline (alpha = 1.0; any --ties are ignored):
    python concat_for_training.py \
        --strict hotel_data/seed0/strict.jsonl \
        --alpha  1.0 \
        --output hotel_data/seed0/aug_a1.0.jsonl
"""

import argparse
import json
import os
import random

from typing import Dict, List

REQUIRED_FIELDS = ("prompt", "chosen", "rejected")
POOL_FIELDS = ("prompt", "response_a", "response_b")

# ===================
# function: load json
# ===================
def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]

# =======================
# function: detect format
# =======================
def detect_format(row: Dict) -> str:
    """Return 'dpo', 'pool', or raise if neither."""
    if "chosen" in row and "rejected" in row and "prompt" in row:
        return "dpo"
    if "response_a" in row and "response_b" in row and "prompt" in row:
        return "pool"
    raise ValueError(
        f"unrecognized row schema. Found fields: {sorted(row.keys())}. "
        f"Expected DPO {REQUIRED_FIELDS} or pool {POOL_FIELDS}."
    )

# ======================
# function: check strict
# ======================
def check_strict(rows: List[Dict], path: str) -> None:
    if not rows:
        raise ValueError(f"{path} is empty")
    if detect_format(rows[0]) != "dpo":
        raise ValueError(
            f"{path} must be a strict file in DPO format "
            f"(prompt/chosen/rejected)."
        )
    for i, r in enumerate(rows[:50]):
        for f in REQUIRED_FIELDS:
            if r.get(f) is None or r.get(f) == "":
                raise ValueError(f"{path} row {i}: field '{f}' is null/empty")

# ===================
# function: get score
# ===================
def _get_score(row: Dict, field: str) -> float:
    """Look up a scalar score on the row, falling back to row['metadata']."""
    if field in row and row[field] is not None:
        return float(row[field])
    meta = row.get("metadata") or {}
    if field in meta and meta[field] is not None:
        return float(meta[field])
    raise ValueError(
        f"--select topk needs score field '{field}', not found on row or in "
        f"its metadata. Available top-level fields: {sorted(row.keys())}"
    )

# =======================
# function: normalize tie
# =======================
def normalize_tie(row: Dict, keep_extra: bool) -> Dict:
    fmt = detect_format(row)
    cand = {"prompt": row["prompt"], "_fmt": fmt, "_raw": row}
    if fmt == "dpo":
        cand["_chosen"], cand["_rejected"] = row["chosen"], row["rejected"]
    else:
        cand["_a"], cand["_b"] = row["response_a"], row["response_b"]
    if keep_extra:
        cand["_extra"] = {k: v for k, v in row.items()
                          if k not in ("prompt", "chosen", "rejected",
                                       "response_a", "response_b")}
    return cand


def realize(cand: Dict, rng: random.Random, keep_extra: bool) -> Dict:
    """Collapse a candidate into a {prompt, chosen, rejected} record."""
    if cand["_fmt"] == "dpo":
        chosen, rejected = cand["_chosen"], cand["_rejected"]
    else:
        # symmetric 50/50 tie label -- realizes the tie measure
        if rng.random() < 0.5:
            chosen, rejected = cand["_a"], cand["_b"]
        else:
            chosen, rejected = cand["_b"], cand["_a"]
    out = {"prompt": cand["prompt"], "chosen": chosen, "rejected": rejected}
    if keep_extra and "_extra" in cand:
        out.update(cand["_extra"])
    return out


def k_from_alpha(alpha: float, n_strict: int) -> int:
    if alpha >= 1.0:
        return 0
    if alpha <= 0.0:
        raise ValueError("alpha must be in (0, 1].")
    return int(round(n_strict * (1.0 - alpha) / alpha))

# ==============
# function: main
# ==============
def main():
    ap = argparse.ArgumentParser(
        description="Concatenate strict + ties (pool or DPO) at a target alpha."
    )
    ap.add_argument("--strict", required=True,
                    help="Path to strict.jsonl (DPO format).")
    ap.add_argument("--ties", nargs="*", default=[],
                    help="Zero or more tie files, pool (response_a/response_b) "
                         "or DPO (prompt/chosen/rejected) format.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")

    ap.add_argument("--alpha", type=float, default=None,
                    help="Strict fraction n_strict/(n_strict+n_ties). "
                         "Sets k = S*(1-alpha)/alpha. 1.0 => strict-only.")
    ap.add_argument("--k", type=int, default=None,
                    help="Explicit tie budget. Overrides --alpha if both given.")
    ap.add_argument("--select", choices=["random", "topk", "first"],
                    default="random",
                    help="How to pick k ties from the combined pool.")
    ap.add_argument("--score-field", default="score",
                    help="Scalar field for --select topk (row or metadata).")

    ap.add_argument("--keep-extra", action="store_true",
                    help="Keep extra fields beyond prompt/chosen/rejected.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the final concatenated rows before writing.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"[load] strict: {args.strict}")
    strict_rows = load_jsonl(args.strict)
    check_strict(strict_rows, args.strict)
    S = len(strict_rows)
    print(f"  -> {S} rows")

    candidates: List[Dict] = []
    for tie_path in args.ties:
        print(f"[load] ties:   {tie_path}")
        rows = load_jsonl(tie_path)
        if not rows:
            print("  -> 0 rows (skipped)")
            continue
        fmt = detect_format(rows[0])
        print(f"  -> {len(rows)} rows ({fmt} format)")
        candidates.extend(normalize_tie(r, args.keep_extra) for r in rows)

    pool_n = len(candidates)

    if args.k is not None:
        k = args.k
    elif args.alpha is not None:
        k = k_from_alpha(args.alpha, S)
    else:
        k = pool_n  # use everything (original behavior)

    if k > pool_n:
        print(f"  WARNING: requested k={k} ties but pool has only {pool_n}. "
              f"Capping at {pool_n}. Realized alpha will be higher than asked "
              f"(one-tie-per-anchor pools can't reach alpha < 0.5).")
        k = pool_n

    if k <= 0:
        selected = []
    elif args.select == "first":
        selected = candidates[:k]
    elif args.select == "topk":
        scored = sorted(candidates,
                        key=lambda c: _get_score(c["_raw"], args.score_field),
                        reverse=True)
        selected = scored[:k]
    else:  # random
        idx = list(range(pool_n))
        rng.shuffle(idx)
        selected = [candidates[i] for i in idx[:k]]

    denom = S + len(selected)
    realized_alpha = S / denom if denom else 1.0
    print(f"[budget] strict={S}  tie_pool={pool_n}  ties_used={len(selected)} "
          f"(select={args.select})")
    print(f"[budget] realized alpha = {realized_alpha:.4f} "
          f"(asked {args.alpha if args.alpha is not None else 'n/a'})")

    combined = ([r if args.keep_extra else {f: r[f] for f in REQUIRED_FIELDS}
                 for r in strict_rows]
                + [realize(c, rng, args.keep_extra) for c in selected])

    if args.shuffle:
        rng.shuffle(combined)
        print(f"[shuffle] applied with seed={args.seed}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(combined)} rows -> {args.output}")

    one = sum(1 for r in combined if "Option ONE" in (r.get("chosen") or ""))
    two = len(combined) - one
    print(f"[balance] chose ONE: {one} ({one/len(combined)*100:.1f}%); "
          f"chose TWO: {two} ({two/len(combined)*100:.1f}%)")
    if abs(one - two) > len(combined) * 0.1:
        print("  WARNING: imbalance > 10%. Verify upstream label assignment.")


if __name__ == "__main__":
    main()
