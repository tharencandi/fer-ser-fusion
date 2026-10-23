"""Per-(fold, seed, system) runs with resumable on-disk outputs.

Layout under RESULTS_DIR:
    folds.json, clips.csv, alignment.json, params.json, config_<timestamp>.json
    runs/<system>/<fold>/seed<seed>/<cfg>.npz + .json      one training run
    runs/<system>/<fold>/seed<seed>/selected.json          validation-selected cfg
    runs/fer/<fold>/seed<seed>/xgbrf.npz + .json
    fusion/<system>+fer/<fold>/seed<seed>.npz + .json      (seed = speech seed; FER seed in the .json)

Seeds are per system (cfg["SYSTEM_SEEDS"], cfg["FER_SEEDS"]). The primary seed
(cfg["PRIMARY_SEED"], 42) runs for every system and fold first, so one complete
CV exists as early as possible; the extra seeds follow.

A run is complete when its .json exists (written last, after the .npz), so an
interrupted run is simply redone on restart.
"""
import json
import os
import platform
import time

import numpy as np
import torch

from . import models as M
from .fer import train_xgbrf, xgb_margins
from .folds import check_fold, split_indices
from .fusion import METHOD as FUSION_METHOD
from .fusion import calibrated_sum_fusion
from .train import (build_training_components, class_weights_for, make_loader,
                    predict_logits, seed_everything, train_model)

# ── Systems ─────────────────────────────────────────────────────────────────
SYSTEMS = {
    "w2v_wlp":            dict(label="W2V2-WLP",              backbone="w2v", adapt="frozen", protocol="pipeline"),
    "w2v_lora":           dict(label="W2V2-LoRA",             backbone="w2v", adapt="lora",   protocol="pipeline"),
    "e2v_lp":             dict(label="E2V-LP",                backbone="e2v", adapt="frozen", protocol="pipeline"),
    "w2v_frozen_matched": dict(label="W2V2 frozen (matched)", backbone="w2v", adapt="frozen", protocol="matched"),
    "e2v_frozen_matched": dict(label="E2V frozen (matched)",  backbone="e2v", adapt="frozen", protocol="matched"),
}

# Fixed recipes of the three pipeline systems (not tuned). use_class_weights=True for all.
PIPELINE_RECIPES = {
    "w2v_wlp":  dict(learning_rate=1e-3, weight_decay=1e-2, dropout=0.25, epochs=25, patience=5),
    "w2v_lora": dict(learning_rate=1e-3, weight_decay=1e-2, dropout=0.25, epochs=25, patience=5),
    "e2v_lp":   dict(learning_rate=1e-3, weight_decay=1e-4, dropout=0.20, epochs=50, patience=10),
}


def cfg_id(hp):
    return f"lr{hp['learning_rate']:.0e}_do{hp['dropout']:.2f}".replace("+", "")


def system_grid(system, cfg):
    """List of (cfg_id, recipe) for a system. Pipeline systems have one fixed recipe."""
    if SYSTEMS[system]["protocol"] == "pipeline":
        r = dict(PIPELINE_RECIPES[system], use_class_weights=True)
        if cfg.get("EPOCHS_OVERRIDE"):
            r["epochs"] = cfg["EPOCHS_OVERRIDE"]
        return [("fixed", r)]
    base = dict(cfg["MATCHED_RECIPE"])
    if cfg.get("EPOCHS_OVERRIDE"):
        base["epochs"] = cfg["EPOCHS_OVERRIDE"]
    return [(cfg_id(hp), dict(base, **hp)) for hp in cfg["MATCHED_GRID_FROZEN"]]


# ── Storage ─────────────────────────────────────────────────────────────────
def _atomic_write_bytes(path, write_fn):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        write_fn(fh)
    os.replace(tmp, path)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write_bytes(path, lambda fh: fh.write(json.dumps(obj, indent=2, default=_json_default).encode()))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class RunStore:
    def __init__(self, root):
        self.root = root

    def run_base(self, system, fold, seed, cfg):
        return os.path.join(self.root, "runs", system, fold, f"seed{seed}", cfg)

    def fusion_base(self, fusion_name, fold, seed):
        return os.path.join(self.root, "fusion", fusion_name, fold, f"seed{seed}")

    @staticmethod
    def done(base):
        return os.path.exists(base + ".json") and os.path.exists(base + ".npz")

    @staticmethod
    def save(base, arrays, meta):
        os.makedirs(os.path.dirname(base), exist_ok=True)
        # no object arrays (pandas 3 strings would need pickle to load)
        arrays = {k: (np.asarray(v, dtype=str) if np.asarray(v).dtype == object else v) for k, v in arrays.items()}
        _atomic_write_bytes(base + ".npz", lambda fh: np.savez(fh, **arrays))
        save_json(base + ".json", meta)  # written last: marks the run complete

    @staticmethod
    def load(base):
        with np.load(base + ".npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, load_json(base + ".json")


# ── Context ─────────────────────────────────────────────────────────────────
class CVContext:
    """Everything a run needs. Built once in the notebook.

    Attributes set by the notebook:
        cfg, device, clips (DataFrame), labels and clip_ids (arrays by HF idx), num_classes,
        padded_inputs (N, T) audio padded to 5.27 s, then normalised (pipeline systems),
        caches {"w2v_states", "e2v_states", "e2v_last"} -> arrays by HF idx,
        fer_X (N, 376) blendshape features by HF idx, backbone_frozen_params {"w2v": n, "e2v": n},
        store (RunStore)
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _env_info():
    info = {"torch": torch.__version__, "python": platform.python_version()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def build_system(ctx, system, recipe):
    """Returns (model, xs, frozen_param_count)."""
    C = ctx.num_classes
    spec = SYSTEMS[system]
    w2v_id = ctx.cfg["W2V_MODEL_ID"]
    if system == "w2v_wlp":
        return M.Wav2Vec2LinearProbe(w2v_id, C, dropout_prob=recipe["dropout"]), ctx.padded_inputs, None
    if system == "w2v_lora":
        return M.Wav2Vec2LoRAProbe(w2v_id, C), ctx.padded_inputs, None
    if system == "e2v_lp":
        feats = ctx.caches["e2v_last"]
        m = M.Emotion2VecLastLayerHead(feats.shape[-1], C, dropout_prob=recipe["dropout"])
        return m, feats, ctx.backbone_frozen_params["e2v"]
    feats = ctx.caches[f"{spec['backbone']}_states"]
    m = M.SharedProbe(feats.shape[1], feats.shape[2], C, recipe["dropout"])
    return m, feats, ctx.backbone_frozen_params[spec["backbone"]]


def run_speech(ctx, system, fold, seed, cid, recipe, verbose=True):
    base = ctx.store.run_base(system, fold["fold"], seed, cid)
    if RunStore.done(base):
        return False
    idx = split_indices(ctx.clips, fold)
    cw = class_weights_for(ctx.labels[idx["train"]], ctx.num_classes)
    seed_everything(seed)
    t0 = time.time()
    model, xs, frozen_extra = build_system(ctx, system, recipe)
    model = model.to(ctx.device)
    params = M.param_breakdown(model)
    if frozen_extra is not None:
        params["frozen"] += int(frozen_extra)

    bs, ebs = ctx.cfg["BATCH_SIZE"], ctx.cfg["EVAL_BATCH_SIZE"]
    tr = make_loader(xs, ctx.labels, idx["train"], bs, shuffle=True, seed=seed)
    va = make_loader(xs, ctx.labels, idx["val"], ebs)
    te = make_loader(xs, ctx.labels, idx["test"], ebs)
    opt, sch, crit = build_training_components(model, recipe, cw, ctx.device)
    info = train_model(model, tr, va, opt, sch, crit, recipe["epochs"], recipe["patience"],
                       ctx.device, use_amp=ctx.cfg["USE_AMP"], name=f"{system}/{fold['fold']}/s{seed}/{cid}",
                       verbose=verbose)
    val_logits, val_labels = predict_logits(model, va, ctx.device)
    test_logits, test_labels = predict_logits(model, te, ctx.device)
    wall = time.time() - t0

    arrays = dict(val_logits=val_logits, val_labels=val_labels, val_idx=idx["val"],
                  test_logits=test_logits, test_labels=test_labels, test_idx=idx["test"],
                  test_clip_ids=ctx.clip_ids[idx["test"]], test_preds=test_logits.argmax(1))
    meta = dict(system=system, fold=fold["fold"], seed=int(seed), cfg=cid, recipe=recipe,
                params=params, wall_time_s=wall, env=_env_info(),
                val_acc=float((val_logits.argmax(1) == val_labels).mean()),
                test_acc=float((test_logits.argmax(1) == test_labels).mean()), **info)
    RunStore.save(base, arrays, meta)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[done] {system:20s} {fold['fold']:6s} seed {seed} {cid:16s} "
          f"val_loss {info['best_val_loss']:.4f} epochs {info['epochs_trained']:2d} "
          f"test_acc {meta['test_acc']:.3f} params {params['trainable_total']:,} {wall:.0f}s")
    return True


def select_cfg(ctx, system, fold_name, seed):
    """Validation-only selection: lowest best-validation loss over the grid
    (ties: first in grid order). Returns cfg id, or None if the grid is incomplete."""
    grid = system_grid(system, ctx.cfg)
    metas = []
    for cid, _ in grid:
        base = ctx.store.run_base(system, fold_name, seed, cid)
        if not RunStore.done(base):
            return None
        metas.append(load_json(base + ".json"))
    best = min(range(len(metas)), key=lambda i: (metas[i]["best_val_loss"], i))
    sel = {"selected": grid[best][0], "criterion": "min best_val_loss on validation actors",
           "candidates": {m["cfg"]: m["best_val_loss"] for m in metas}}
    save_json(os.path.join(ctx.store.root, "runs", system, fold_name, f"seed{seed}", "selected.json"), sel)
    return grid[best][0]


def fer_seed_for(cfg, seed):
    """FER model fused with a speech run of `seed`: the same seed if FER has it,
    otherwise the primary seed (single-seed systems use the seed-42 FER model)."""
    return seed if seed in cfg["FER_SEEDS"] else cfg["PRIMARY_SEED"]


def run_fer(ctx, fold, seed):
    name = "fer"
    base = ctx.store.run_base(name, fold["fold"], seed, "xgbrf")
    if RunStore.done(base):
        return False
    X = ctx.fer_X
    idx = split_indices(ctx.clips, fold)
    seed_everything(seed)
    t0 = time.time()
    clf = train_xgbrf(X[idx["train"]], ctx.labels[idx["train"]], seed)
    val_logits, test_logits = xgb_margins(clf, X[idx["val"]]), xgb_margins(clf, X[idx["test"]])
    arrays = dict(val_logits=val_logits, val_labels=ctx.labels[idx["val"]], val_idx=idx["val"],
                  test_logits=test_logits, test_labels=ctx.labels[idx["test"]], test_idx=idx["test"],
                  test_clip_ids=ctx.clip_ids[idx["test"]], test_preds=test_logits.argmax(1))
    meta = dict(system=name, fold=fold["fold"], seed=int(seed), cfg="xgbrf", wall_time_s=time.time() - t0,
                test_acc=float((test_logits.argmax(1) == arrays["test_labels"]).mean()),
                params={"n_estimators": clf.n_estimators, "max_depth": clf.max_depth})
    RunStore.save(base, arrays, meta)
    print(f"[done] {name:20s} {fold['fold']:6s} seed {seed} test_acc {meta['test_acc']:.3f}")
    return True


def run_fusion(ctx, system, fold, seed, speech_cfg):
    fname = f"{system}+fer"
    fseed = fer_seed_for(ctx.cfg, seed)
    base = ctx.store.fusion_base(fname, fold["fold"], seed)
    if RunStore.done(base):
        old = load_json(base + ".json")
        if (old.get("speech_cfg") == speech_cfg and old.get("fer_seed") == fseed
                and old.get("temperature_method") == FUSION_METHOD):
            return False
    fer_base = ctx.store.run_base("fer", fold["fold"], fseed, "xgbrf")
    if not RunStore.done(fer_base):
        return False
    a, _ = RunStore.load(ctx.store.run_base(system, fold["fold"], seed, speech_cfg))
    v, _ = RunStore.load(fer_base)
    for split in ("val", "test"):
        if not (np.array_equal(a[f"{split}_idx"], v[f"{split}_idx"])
                and np.array_equal(a[f"{split}_labels"], v[f"{split}_labels"])):
            raise ValueError(f"{fname}/{fold['fold']}/seed{seed}: {split} clips or labels differ between modalities")
    probs, t_a, t_v = calibrated_sum_fusion(a["val_logits"], v["val_logits"], a["val_labels"],
                                           a["test_logits"], v["test_logits"])
    arrays = dict(test_probs=probs, test_preds=probs.argmax(1), test_labels=a["test_labels"], test_idx=a["test_idx"],
                  test_clip_ids=ctx.clip_ids[a["test_idx"]])
    meta = dict(system=fname, speech_system=system, speech_cfg=speech_cfg, fold=fold["fold"],
                seed=int(seed), fer_seed=int(fseed), temperature_method=FUSION_METHOD,
                temperature_audio=t_a, temperature_visual=t_v,
                test_acc=float((arrays["test_preds"] == arrays["test_labels"]).mean()))
    RunStore.save(base, arrays, meta)
    return True


def seed_order(cfg, systems):
    """Primary seed first, then the remaining seeds in ascending order."""
    seeds = set(cfg["FER_SEEDS"]) | {sd for s in systems for sd in cfg["SYSTEM_SEEDS"][s]}
    return [cfg["PRIMARY_SEED"]] + sorted(seeds - {cfg["PRIMARY_SEED"]})


def run_all(ctx, folds, systems, with_fer=True, verbose_epochs=False):
    """Main loop. Seed 42 for every system and fold first (one complete CV as
    early as possible), then the extra seeds of the systems that have them."""
    cfg = ctx.cfg
    seeds = seed_order(cfg, systems)
    total = sum(len(system_grid(s, cfg)) * len(cfg["SYSTEM_SEEDS"][s]) for s in systems) * len(folds)
    n, t_start = 0, time.time()
    for seed in seeds:
        print(f"══ seed {seed}: " + ", ".join(s for s in systems if seed in cfg["SYSTEM_SEEDS"][s])
              + (" + FER" if with_fer and seed in cfg["FER_SEEDS"] else ""))
        for fold in folds:
            check_fold(ctx.clips, fold, ctx.aligned_index, cfg["REQUIRE_FULL_ALIGNMENT"])
            if with_fer and seed in cfg["FER_SEEDS"]:
                run_fer(ctx, fold, seed)
            for system in systems:
                if seed not in cfg["SYSTEM_SEEDS"][system]:
                    continue
                for cid, recipe in system_grid(system, cfg):
                    ran = run_speech(ctx, system, fold, seed, cid, recipe, verbose=verbose_epochs)
                    n += 1
                    if ran:
                        el = time.time() - t_start
                        print(f"        progress {n}/{total} runs checked, {el / 60:.1f} min elapsed this session")
                sel = select_cfg(ctx, system, fold["fold"], seed)
                if with_fer and sel is not None:
                    run_fusion(ctx, system, fold, seed, sel)
    print(f"All requested runs present ({total} speech runs).")
