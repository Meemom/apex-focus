"""Group splits that keep detected duplicates and pilot sources out of test."""

import hashlib
import math


def assign_splits(records, seed=42, validation_fraction=0.2, test_fraction=0.0,
                  excluded_test_groups=(), near_distance=4):
    if not (0 < validation_fraction < 1 and 0 <= test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Invalid validation/test fractions")
    if not isinstance(near_distance, int) or not 0 <= near_distance <= 63:
        raise ValueError("Near-duplicate distance must be an integer from 0 to 63")
    groups = sorted({r["group"] for r in records})
    parents = {group: group for group in groups}

    def find(group):
        while parents[group] != group:
            group = parents[group]
        return group

    duplicates = []
    # shortcut: pairwise pHash comparison is intended for the <=3,000-image pilot/study.
    if len(records) > 3000:
        raise ValueError("More than 3,000 images requires an indexed duplicate audit")
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            distance = (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count()
            exact = left["image_sha256"] == right["image_sha256"]
            pixels = left["pixel_sha256"] == right["pixel_sha256"]
            if exact or pixels or distance <= near_distance:
                kind = "file" if exact else "pixels" if pixels else "perceptual_candidate"
                duplicates.append({"left": left["id"], "right": right["id"],
                                   "kind": kind, "distance": distance})
                a, b = find(left["group"]), find(right["group"])
                parents[max(a, b)] = min(a, b)
    components = {}
    for group in groups:
        components.setdefault(find(group), []).append(group)
    ordered = sorted(components, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    n = len(ordered)
    n_val = max(1, math.floor(n * validation_fraction))
    n_test = max(1, math.floor(n * test_fraction)) if test_fraction else 0
    if n - n_val - n_test < 1:
        raise ValueError("Too few independent source groups after duplicate clustering")
    excluded = set(excluded_test_groups)
    eligible = [g for g in ordered if not (set(components[g]) & excluded)]
    if len(eligible) < n_test:
        raise ValueError("Not enough non-pilot source groups for a held-out test split")
    test = set(eligible[:n_test])
    val = set([g for g in ordered if g not in test][:n_val])
    assignment = {g: "test" if find(g) in test else "val" if find(g) in val else "train"
                  for g in groups}
    return assignment, duplicates
