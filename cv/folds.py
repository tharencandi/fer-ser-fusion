"""Actor-wise fold assignment.

Six folds of four test actors, each 2 male + 2 female (RAVDESS: odd actor ids
are male, even are female). Males and females are shuffled separately with a
fixed seed and cut into pairs; fold k tests male pair k + female pair k.
Fold k validates on the test actors of fold (k+1) mod 6 (also 2M/2F) and trains
on the remaining 16. Every actor is tested exactly once and used for
validation exactly once.
"""
import json

import numpy as np

N_ACTORS = 24


def make_folds(n_folds=6, seed=42, actors=range(1, N_ACTORS + 1)):
    actors = sorted(int(a) for a in actors)
    males = [a for a in actors if a % 2 == 1]
    females = [a for a in actors if a % 2 == 0]
    if len(males) != len(females) or len(males) % n_folds:
        raise ValueError("Need equal male/female counts divisible by n_folds")
    per = len(males) // n_folds
    rng = np.random.default_rng(seed)
    males = [int(a) for a in rng.permutation(males)]
    females = [int(a) for a in rng.permutation(females)]
    tests = [sorted(males[k * per:(k + 1) * per] + females[k * per:(k + 1) * per]) for k in range(n_folds)]

    folds = []
    for k in range(n_folds):
        test, val = tests[k], tests[(k + 1) % n_folds]
        train = sorted(a for a in actors if a not in test and a not in val)
        folds.append({"fold": f"fold{k}", "train": train, "val": val, "test": test})
    return {
        "scheme": (f"{n_folds} actor-wise folds; test = 2M+2F per fold (males and females shuffled "
                   f"separately with numpy default_rng(seed), cut into pairs); val = test actors of "
                   f"the next fold (mod {n_folds}); train = remaining actors"),
        "seed": int(seed),
        "male_actors": "odd ids",
        "female_actors": "even ids",
        "folds": folds,
    }


def validate_folds(spec):
    folds = [f for f in spec["folds"]]
    all_test = []
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & va) and not (tr & te) and not (va & te), f"{f['fold']}: overlapping actors"
        assert len(tr | va | te) == N_ACTORS, f"{f['fold']}: actors missing"
        for name, s in (("val", va), ("test", te)):
            n_m = sum(a % 2 == 1 for a in s)
            assert n_m == len(s) - n_m, f"{f['fold']}: {name} is not gender-balanced"
        all_test += f["test"]
    assert sorted(all_test) == list(range(1, N_ACTORS + 1)), "every actor must be tested exactly once"


def save_folds(spec, path):
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=2)


def load_folds(path):
    with open(path) as fh:
        return json.load(fh)


def split_indices(clips, fold):
    """HF row indices of the train/val/test clips of a fold."""
    out = {}
    for split in ("train", "val", "test"):
        out[split] = np.sort(clips.loc[clips["actor"].isin(fold[split]), "idx"].values)
    return out


def check_fold(clips, fold, aligned_index, require_full_alignment=True):
    """Per-fold leakage and alignment checks. Returns a small report dict."""
    idx = split_indices(clips, fold)
    sets = {k: set(v.tolist()) for k, v in idx.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"]), f"{fold['fold']}: a clip appears in two splits"
    by_idx = clips.set_index("idx")
    for split, ids in idx.items():
        actors = set(by_idx.loc[ids, "actor"].tolist())
        assert actors <= set(fold[split]), f"{fold['fold']}: {split} contains a foreign actor"
    clip_ids = by_idx.loc[np.concatenate(list(idx.values())), "clip_id"]
    assert not clip_ids.duplicated().any(), f"{fold['fold']}: duplicate clip ids across splits"
    report = {"fold": fold["fold"]}
    for split, ids in idx.items():
        n_face = int(np.isin(ids, aligned_index).sum())
        if require_full_alignment and n_face != len(ids):
            raise ValueError(f"{fold['fold']}/{split}: {len(ids) - n_face} clips have no blendshapes")
        report[f"n_{split}"] = int(len(ids))
        report[f"n_{split}_with_face"] = n_face
    return report
