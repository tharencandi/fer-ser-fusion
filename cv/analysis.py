"""Analysis of saved CV outputs. Reads only RESULTS_DIR (no training, no models).

    python -m cv.analysis --results <RESULTS_DIR> --out results_cv

Writes CSV/JSON tables, PNG figures and RESULTS_CV.md into --out. RESULTS_CV.md
is organised around three research questions:

    RQ1  does calibrated late fusion with FER improve SER across actors?
    RQ2  frozen wav2vec 2.0 vs frozen emotion2vec under the matched probe/recipe
    RQ3  wav2vec 2.0: LoRA pipeline vs frozen weighted-layer-probe pipeline

Only REPORTED_SPEECH systems, FER and their fusions are analysed; anything else
under runs/ is ignored.

Statistics
----------
* seeds differ per system (3 for cached-feature systems, FER and their fusions,
  1 for backbone-in-the-loop systems); each system is summarised on its own seeds.
* per run (fold x seed): accuracy and macro-F1 on the fold's 4 test actors;
  reported as mean +- std over the system's CV runs.
* pooled: for each seed, predictions of the 6 folds are concatenated (every
  clip once, all 24 actors) and scored; reported as the mean over seeds.
* 95% CI: cluster bootstrap over actors (resample 12 male and 12 female actors
  with replacement; a resampled actor brings all its clips, for every seed).
* paired comparisons use the seeds both systems have (all 3 if both are
  multi-seed, seed 42 if either is single-seed) and the same resampled actors
  for both systems. Bootstrap p is two-sided, Holm-adjusted within each table.
  Per-fold: Wilcoxon signed-rank on the per-fold differences (mean over seeds),
  rank-biserial r and Cohen's d_z; with 6 folds the smallest possible
  two-sided Wilcoxon p is 0.031, so verdicts rest on the actor bootstrap.
  McNemar (exact) per seed ignores clustering by actor and is optimistic.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .runner import SYSTEMS, RunStore, load_json

REPORTED_SPEECH = ["w2v_frozen_matched", "e2v_frozen_matched", "w2v_wlp", "w2v_lora", "e2v_lp"]

COMPARISONS = (
    # (A, B, family, description)
    [(f"{s}+fer", s, "rq1", "fusion gain") for s in REPORTED_SPEECH]
    + [(f"{s}+fer", "fer", "rq1_face", "fusion vs face only") for s in REPORTED_SPEECH]
    + [("w2v_frozen_matched", "e2v_frozen_matched", "rq2", "speech only"),
       ("w2v_frozen_matched+fer", "e2v_frozen_matched+fer", "rq2", "fused with FER")]
    + [("w2v_lora", "w2v_wlp", "rq3", "speech only"),
       ("w2v_lora+fer", "w2v_wlp+fer", "rq3", "fused with FER")]
    + [("w2v_wlp", "e2v_lp", "other", "frozen pipelines: W2V2-WLP vs E2V-LP"),
       ("w2v_lora", "e2v_lp", "other", "W2V2 LoRA vs E2V-LP"),
       ("w2v_frozen_matched", "w2v_wlp", "other", "W2V2 frozen: matched probe vs WLP"),
       ("e2v_frozen_matched", "e2v_lp", "other", "E2V frozen: matched probe vs final-layer probe")]
)

FAMILY_COLORS = {"Speech": "#2a78d6", "FER": "#eb6834", "Speech+FER": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def label_of(system):
    if "+" in system:
        speech, face = system.split("+", 1)
        return f"{label_of(speech)} + {label_of(face)}".replace(" + FER (XGBRF)", " + FER")
    if system == "fer":
        return "FER (XGBRF)"
    return SYSTEMS[system]["label"]


def family_of(system):
    return "FER" if system.startswith("fer") else ("Speech+FER" if "+fer" in system else "Speech")


def reported_systems():
    return REPORTED_SPEECH + ["fer"] + [f"{s}+fer" for s in REPORTED_SPEECH]


def system_order(present):
    return [s for s in reported_systems() if s in present]


# ── Loading ─────────────────────────────────────────────────────────────────
def _selected_cfg(run_dir):
    sel = os.path.join(run_dir, "selected.json")
    if os.path.exists(sel):
        return load_json(sel)["selected"]
    metas = [load_json(os.path.join(run_dir, f)) for f in sorted(os.listdir(run_dir))
             if f.endswith(".json") and f != "selected.json"]
    if not metas:
        return None
    return min(metas, key=lambda m: m["best_val_loss"])["cfg"] if "best_val_loss" in metas[0] else metas[0]["cfg"]


def _wanted(system, fold):
    return system in reported_systems()


def load_predictions(results_dir):
    """Long table: system, fold, seed, idx, label, pred. Plus run metadata."""
    rows, metas = [], []

    def add(arrays, meta, system, fold):
        rows.append(pd.DataFrame({"system": system, "fold": fold, "seed": meta["seed"], "idx": arrays["test_idx"],
                                  "label": arrays["test_labels"], "pred": arrays["test_preds"]}))
        metas.append(meta)

    runs_root = os.path.join(results_dir, "runs")
    for system in sorted(os.listdir(runs_root)) if os.path.isdir(runs_root) else []:
        for fold in sorted(os.listdir(os.path.join(runs_root, system))):
            if not _wanted(system, fold):
                continue
            for sd in sorted(os.listdir(os.path.join(runs_root, system, fold))):
                run_dir = os.path.join(runs_root, system, fold, sd)
                cid = _selected_cfg(run_dir)
                if cid is None or not RunStore.done(os.path.join(run_dir, cid)):
                    continue
                arrays, meta = RunStore.load(os.path.join(run_dir, cid))
                add(arrays, meta, system, fold)
    fus_root = os.path.join(results_dir, "fusion")
    for system in sorted(os.listdir(fus_root)) if os.path.isdir(fus_root) else []:
        for fold in sorted(os.listdir(os.path.join(fus_root, system))):
            if not _wanted(system, fold):
                continue
            for f in sorted(os.listdir(os.path.join(fus_root, system, fold))):
                base = os.path.join(fus_root, system, fold, f[:-5])
                if f.endswith(".json") and RunStore.done(base):
                    add(*RunStore.load(base), system, fold)
    if not rows:
        raise FileNotFoundError(f"No completed runs under {results_dir}")
    preds = pd.concat(rows, ignore_index=True)
    preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    return preds, metas


# ── Metrics ─────────────────────────────────────────────────────────────────
def acc_f1_from_cm(cm):
    """cm[..., true, pred] -> (accuracy, macro-F1 over classes with support)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    total = support.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = tp.sum(-1) / total
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        has = support > 0
        macro = (f1 * has).sum(-1) / has.sum(-1)
    return acc, macro


def confusion(labels, preds, C):
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels), np.asarray(preds)), 1)
    return cm


def per_run_metrics(preds, C):
    out = []
    for (s, f, sd), g in preds.groupby(["system", "fold", "seed"]):
        acc, f1 = acc_f1_from_cm(confusion(g["label"], g["pred"], C))
        out.append({"system": s, "fold": f, "seed": sd, "n": len(g), "acc": float(acc), "macro_f1": float(f1)})
    return pd.DataFrame(out)


class ActorBootstrap:
    """Cluster bootstrap over actors, stratified by gender, shared across systems."""

    def __init__(self, actors, B=10000, seed=0, stratify=True):
        self.actors = np.asarray(sorted(actors))
        rng = np.random.default_rng(seed)
        W = np.zeros((B, len(self.actors)))
        groups = [np.where(self.actors % 2 == g)[0] for g in (1, 0)] if stratify else [np.arange(len(self.actors))]
        for grp in groups:
            draws = rng.choice(grp, size=(B, len(grp)), replace=True)
            for j in range(len(grp)):
                np.add.at(W, (np.arange(B), draws[:, j]), 1)
        self.W = W

    def cm_by_actor(self, sub, seeds, C, actor_of):
        """[S, A, C, C] confusion matrices for one system."""
        a_pos = {a: i for i, a in enumerate(self.actors)}
        cms = np.zeros((len(seeds), len(self.actors), C, C))
        sub = sub.assign(a=actor_of.loc[sub["idx"]].values)
        for si, sd in enumerate(seeds):
            g = sub[sub["seed"] == sd]
            np.add.at(cms[si], (g["a"].map(a_pos).values, g["label"].values, g["pred"].values), 1)
        return cms

    def stats(self, cms):
        """Point estimate and bootstrap draws of seed-averaged pooled acc and macro-F1."""
        point = acc_f1_from_cm(cms.sum(1))
        boot_cm = np.einsum("ba,sacd->bscd", self.W, cms)
        boot = acc_f1_from_cm(boot_cm)
        return (point[0].mean(), point[1].mean()), (boot[0].mean(1), boot[1].mean(1))


def _ci(draws):
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _boot_p(diff_draws):
    return float(min(1.0, 2 * min((diff_draws <= 0).mean(), (diff_draws >= 0).mean())))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


# ── Main analysis ───────────────────────────────────────────────────────────
def analyse(results_dir, out_dir, B=10000, boot_seed=0, make_figures=True):
    from scipy.stats import binomtest, wilcoxon

    os.makedirs(out_dir, exist_ok=True)
    clips = pd.read_csv(os.path.join(results_dir, "clips.csv"))
    folds_spec = load_json(os.path.join(results_dir, "folds.json"))
    with open(os.path.join(results_dir, "label_space.json")) as fh:
        id2label = {int(k): v for k, v in json.load(fh)["id2label"].items()}
    C = len(id2label)
    actor_of = clips.set_index("idx")["actor"]
    intensity_of = clips.set_index("idx")["intensity"]

    preds_all, metas = load_predictions(results_dir)
    all_cv_folds = [f["fold"] for f in folds_spec["folds"]]
    preds = preds_all[preds_all["fold"].isin(all_cv_folds)]
    cv_folds = [f for f in all_cv_folds if f in set(preds["fold"])]
    partial = len(cv_folds) < len(all_cv_folds)
    if partial:
        print(f"PARTIAL RESULTS: {len(cv_folds)}/{len(all_cv_folds)} CV folds present; "
              "pooled metrics cover only their test actors.")

    systems = system_order(set(preds["system"]))
    cover = preds.groupby(["system", "seed"])["fold"].nunique()
    sys_seeds = {s: sorted(int(sd) for (ss, sd), n in cover.items() if ss == s and n == len(cv_folds))
                 for s in systems}
    for s in systems:
        if not sys_seeds[s]:
            print(f"WARNING: {s} has no seed covering all {len(cv_folds)} CV folds; it is left out of pooled metrics.")
    runs_all = per_run_metrics(preds, C)
    runs_all.to_csv(os.path.join(out_dir, "per_run_metrics.csv"), index=False)
    runs = runs_all[[sd in sys_seeds.get(s, []) for s, sd in zip(runs_all["system"], runs_all["seed"])]]

    actors = sorted(actor_of.loc[preds["idx"].unique()].unique())
    boot = ActorBootstrap(actors, B=B, seed=boot_seed)
    _cm_cache = {}

    def cms_for(s, seeds):
        key = (s, tuple(seeds))
        if key not in _cm_cache:
            sub = preds[(preds["system"] == s) & preds["seed"].isin(seeds)]
            _cm_cache[key] = boot.cm_by_actor(sub, list(seeds), C, actor_of)
        return _cm_cache[key]

    # ── Summary per system (each system on its own seeds)
    cms, summary = {}, []
    for s in systems:
        seeds_s = sys_seeds[s]
        r = runs[runs["system"] == s]
        row = {"system": s, "label": label_of(s), "family": family_of(s), "n_seeds": len(seeds_s),
               "seeds": ",".join(map(str, seeds_s)), "n_runs": len(r),
               "acc_mean": r["acc"].mean(), "acc_std": r["acc"].std(ddof=1),
               "f1_mean": r["macro_f1"].mean(), "f1_std": r["macro_f1"].std(ddof=1)}
        if seeds_s:
            cms[s] = cms_for(s, seeds_s)
            (pa, pf), (ba, bf) = boot.stats(cms[s])
            per_seed = acc_f1_from_cm(cms[s].sum(1))
            row.update(pooled_acc=pa, pooled_acc_lo=_ci(ba)[0], pooled_acc_hi=_ci(ba)[1],
                       pooled_acc_seed_sd=float(np.std(per_seed[0], ddof=1)) if len(seeds_s) > 1 else np.nan,
                       pooled_f1=pf, pooled_f1_lo=_ci(bf)[0], pooled_f1_hi=_ci(bf)[1])
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ── Paired comparisons on the seeds both systems have
    mcnemar_rows = []

    def paired(a, b, fam, desc):
        if not sys_seeds.get(a) or not sys_seeds.get(b):
            return None
        common = sorted(set(sys_seeds[a]) & set(sys_seeds[b]))
        if not common:
            return None
        (pa_acc, pa_f1), (ba, bfa) = boot.stats(cms_for(a, common))
        (pb_acc, pb_f1), (bb, bfb) = boot.stats(cms_for(b, common))
        da, db = ba - bb, bfa - bfb
        ra = runs[(runs["system"] == a) & runs["seed"].isin(common)].set_index(["fold", "seed"])["acc"]
        rb = runs[(runs["system"] == b) & runs["seed"].isin(common)].set_index(["fold", "seed"])["acc"]
        both = ra.index.intersection(rb.index)
        diffs = ra.loc[both] - rb.loc[both]
        fold_diff = diffs.groupby(level="fold").mean().reindex([f for f in cv_folds if f in set(both.get_level_values(0))])
        w_p = rb_r = np.nan
        if len(fold_diff) >= 2 and np.any(fold_diff.values != 0):
            w_p = float(wilcoxon(fold_diff.values).pvalue)
            nz = fold_diff.values[fold_diff.values != 0]
            ranks = pd.Series(np.abs(nz)).rank().values
            rb_r = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        sd = fold_diff.std(ddof=1)
        mc_p = []
        for sd_ in common:
            ga = preds[(preds["system"] == a) & (preds["seed"] == sd_)].set_index("idx")["correct"]
            gb = preds[(preds["system"] == b) & (preds["seed"] == sd_)].set_index("idx")["correct"]
            j = ga.index.intersection(gb.index)
            n01 = int(((ga.loc[j] == 1) & (gb.loc[j] == 0)).sum())
            n10 = int(((ga.loc[j] == 0) & (gb.loc[j] == 1)).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            mc_p.append(p)
            mcnemar_rows.append({"A": a, "B": b, "seed": sd_, "A_right_B_wrong": n01, "A_wrong_B_right": n10,
                                 "odds_ratio": (n01 + 0.5) / (n10 + 0.5), "p_exact": p})
        return {
            "A": a, "B": b, "family": fam, "description": desc,
            "seeds": ",".join(map(str, common)), "n_seeds": len(common),
            "acc_A": pa_acc, "acc_B": pb_acc,
            "d_acc": pa_acc - pb_acc, "d_acc_lo": _ci(da)[0], "d_acc_hi": _ci(da)[1], "p_boot_acc": _boot_p(da),
            "d_f1": pa_f1 - pb_f1, "d_f1_lo": _ci(db)[0], "d_f1_hi": _ci(db)[1], "p_boot_f1": _boot_p(db),
            "folds_A_better": int((fold_diff > 0).sum()), "n_folds": int(len(fold_diff)),
            "fold_diff_min": float(fold_diff.min()), "fold_diff_max": float(fold_diff.max()),
            "runs_A_better": int((diffs > 0).sum()), "n_runs": int(len(diffs)),
            "fold_diffs": json.dumps({k: round(float(v), 6) for k, v in fold_diff.items()}),
            "fold_diff_mean": float(fold_diff.mean()), "fold_diff_sd": float(sd),
            "cohen_dz": float(fold_diff.mean() / sd) if sd > 0 else np.nan,
            "wilcoxon_p": w_p, "rank_biserial": rb_r,
            "mcnemar_p_per_seed": ";".join(f"{p:.3g}" for p in mc_p),
            "mcnemar_p_max": max(mc_p) if mc_p else np.nan,
        }

    comps = pd.DataFrame([r for r in (paired(*c) for c in COMPARISONS) if r is not None])
    if len(comps):
        comps["p_boot_acc_holm"] = np.nan
        for fam, g in comps.groupby("family"):
            comps.loc[g.index, "p_boot_acc_holm"] = holm(g["p_boot_acc"].values)
    comps.to_csv(os.path.join(out_dir, "paired_comparisons.csv"), index=False)

    # best fused system vs runner-up
    fused = summary[summary["family"] == "Speech+FER"].dropna(subset=["pooled_acc"]).sort_values("pooled_acc", ascending=False)
    best_vs_next = None
    if len(fused) >= 2:
        best_vs_next = paired(fused.iloc[0]["system"], fused.iloc[1]["system"], "best", "best fused vs runner-up")
        if best_vs_next is not None:
            best_vs_next["p_boot_acc_holm"] = best_vs_next["p_boot_acc"]
    pd.DataFrame(mcnemar_rows).to_csv(os.path.join(out_dir, "mcnemar_per_seed.csv"), index=False)

    # ── Per-class recall and confusion matrices (pooled over folds and each system's seeds)
    pooled_cm = {s: cms[s].sum((0, 1)).astype(int) for s in cms}
    recall = pd.DataFrame({s: np.diag(cm) / np.maximum(cm.sum(1), 1) for s, cm in pooled_cm.items()},
                          index=[id2label[i] for i in range(C)])
    recall.to_csv(os.path.join(out_dir, "per_class_recall.csv"))
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as fh:
        json.dump({"classes": [id2label[i] for i in range(C)], "rows_true_cols_pred": True,
                   "matrices": {s: cm.tolist() for s, cm in pooled_cm.items()}}, fh, indent=1)

    # ── Intensity-stratified accuracy
    p2 = preds.assign(intensity=intensity_of.loc[preds["idx"]].values)
    inten = []
    for s in systems:
        g = p2[(p2["system"] == s) & p2["seed"].isin(sys_seeds[s])]
        row = {"system": s, "n_seeds": len(sys_seeds[s])}
        for lvl in ("normal", "strong"):
            gl = g[g["intensity"] == lvl]
            row[f"{lvl}_acc"] = gl.groupby("seed")["correct"].mean().mean()
            row[f"{lvl}_sd_runs"] = gl.groupby(["fold", "seed"])["correct"].mean().std(ddof=1)
            row[f"{lvl}_n_clips_per_seed"] = int(len(gl) / max(gl["seed"].nunique(), 1))
        inten.append(row)
    inten = pd.DataFrame(inten)
    inten.to_csv(os.path.join(out_dir, "intensity_accuracy.csv"), index=False)

    # ── Parameters, training statistics, selected hyperparameters
    meta_df = pd.DataFrame([m for m in metas if "params" in m and m.get("system") in REPORTED_SPEECH
                            and m.get("fold") in cv_folds and m.get("seed") in sys_seeds.get(m["system"], [])])
    params, train_stats, selected = _meta_tables(meta_df)
    params.to_csv(os.path.join(out_dir, "trainable_params.csv"), index=False)
    train_stats.to_csv(os.path.join(out_dir, "training_stats.csv"), index=False)
    selected.to_csv(os.path.join(out_dir, "selected_hyperparameters.csv"), index=False)
    temps = pd.DataFrame([{"system": m["system"], "fold": m["fold"], "seed": m["seed"], "fer_seed": m.get("fer_seed"),
                           "T_audio": m["temperature_audio"], "T_visual": m["temperature_visual"]}
                          for m in metas if "temperature_audio" in m])
    temps.to_csv(os.path.join(out_dir, "fusion_temperatures.csv"), index=False)
    bad_t = temps[(temps["T_audio"] <= 0) | (temps["T_visual"] <= 0)] if len(temps) else temps
    if len(bad_t):
        print(f"WARNING: {len(bad_t)} fusions have a non-positive temperature")

    # seed design from the run's latest config (falls back to the seeds present)
    cfgs = sorted(f for f in os.listdir(results_dir) if f.startswith("config_") and f.endswith(".json"))
    run_cfg = load_json(os.path.join(results_dir, cfgs[-1])) if cfgs else {}
    design = {}
    for s in systems:
        speech = s.split("+")[0]
        if speech == "fer":
            design[s] = run_cfg.get("FER_SEEDS", sys_seeds[s])
        else:
            design[s] = run_cfg.get("SYSTEM_SEEDS", {}).get(speech, sys_seeds[s])
    info = {"n_clips": int(preds["idx"].nunique()), "n_actors": int(len(actors)),
            "n_clips_total": int(len(clips)), "n_actors_total": int(clips["actor"].nunique()),
            "seed_design": design, "cv_folds": cv_folds,
            "partial": partial, "n_cv_folds_total": len(all_cv_folds),
            "seeds_per_system": sys_seeds, "bootstrap_B": int(B), "systems": systems,
            "fold_seed": folds_spec.get("seed"),
            "n_nonpositive_temperatures": int(len(bad_t)),
            "temperature_range": [float(temps[["T_audio", "T_visual"]].min().min()),
                                  float(temps[["T_audio", "T_visual"]].max().max())] if len(temps) else None}
    with open(os.path.join(out_dir, "analysis_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    figs = []
    if make_figures:
        figs = _figures(out_dir, summary, runs, comps, pooled_cm, recall, id2label, systems, len(actors))
    md = render_markdown(info, summary, comps, recall, inten, params, train_stats, selected, best_vs_next, figs)
    with open(os.path.join(out_dir, "RESULTS_CV.md"), "w") as fh:
        fh.write(md)
    print(f"Wrote analysis to {out_dir}")
    return dict(summary=summary, comparisons=comps, recall=recall, intensity=inten, params=params,
                info=info)


def _meta_tables(meta_df):
    if meta_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prow = []
    for s, g in meta_df.groupby("system"):
        prow.append({"system": s, "label": label_of(s), **g.iloc[0]["params"]})
    params = pd.DataFrame(prow)
    params["order"] = params["system"].map({s: i for i, s in enumerate(REPORTED_SPEECH)})
    params = params.sort_values("order").drop(columns="order")
    stats = meta_df.groupby("system").agg(
        runs=("seed", "size"), n_seeds=("seed", "nunique"), epochs_mean=("epochs_trained", "mean"),
        best_epoch_mean=("best_epoch", "mean"), wall_min_mean=("wall_time_s", lambda x: x.mean() / 60)).reset_index()
    sel = meta_df.groupby(["system", "cfg"]).size().rename("times_selected").reset_index()
    return params, stats, sel


# ── Figures ─────────────────────────────────────────────────────────────────
def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figures(out_dir, summary, runs, comps, pooled_cm, recall, id2label, systems, n_actors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    blues = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    files = []

    def save(fig, name):
        fig.savefig(os.path.join(out_dir, name), dpi=200)
        plt.close(fig)
        files.append(name)

    # 1. Pooled accuracy per system with actor-bootstrap CI, single runs behind.
    sm = summary.set_index("system")
    sm = sm.loc[[s for s in systems if s in sm.index and np.isfinite(sm.loc[s].get("pooled_acc", np.nan))]]
    if len(sm):
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(sm) + 1.4))
        rng = np.random.default_rng(0)
        for i, (s, r) in enumerate(sm.iterrows()):
            col = FAMILY_COLORS[r["family"]]
            pts = runs[runs["system"] == s]["acc"].values
            ax.scatter(pts * 100, i + rng.uniform(-0.18, 0.18, len(pts)), s=10, color=col, alpha=0.3, linewidths=0)
            ax.plot([r["pooled_acc_lo"] * 100, r["pooled_acc_hi"] * 100], [i, i], color=col, linewidth=2,
                    solid_capstyle="round")
            ax.scatter([r["pooled_acc"] * 100], [i], s=60, color=col, edgecolor="white", linewidth=2, zorder=3)
            ax.text(r["pooled_acc_hi"] * 100 + 0.6, i, f"{r['pooled_acc'] * 100:.1f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(sm)))
        ax.set_yticklabels([f"{label_of(s)} ({int(sm.loc[s, 'n_seeds'])} seed{'s' if sm.loc[s, 'n_seeds'] > 1 else ''})"
                            for s in sm.index], fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy (%)", color=INK2)
        ax.set_title(f"Pooled accuracy over {n_actors} actors with 95% actor-bootstrap CI\n"
                     "(faint dots: single fold × seed runs)", fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, label=f) for f, c in FAMILY_COLORS.items()],
                  loc="upper center", bbox_to_anchor=(0.5, -0.12 * 8 / (len(sm) + 2)), ncol=3, fontsize=8, frameon=False)
        fig.tight_layout()
        save(fig, "fig_accuracy.png")

    # 2. RQ1: per-fold fusion gain for every speech system (small multiples)
    g = comps[comps["family"] == "rq1"] if len(comps) else comps
    if len(g):
        fig, axes = plt.subplots(1, len(g), figsize=(2.6 * len(g), 2.8), sharey=True, squeeze=False)
        allv = [v * 100 for r in g.itertuples() for v in json.loads(r.fold_diffs).values()]
        ymin, ymax = min(0.0, min(allv)) - 3, max(allv) * 1.15 + 1
        for ax, r in zip(axes.flat, g.itertuples()):
            fd = json.loads(r.fold_diffs)
            ax.bar(range(len(fd)), [v * 100 for v in fd.values()], color=FAMILY_COLORS["Speech+FER"], width=0.6)
            ax.axhline(0, color=INK2, linewidth=1)
            ax.set_xticks(range(len(fd)))
            ax.set_xticklabels([k.replace("fold", "") for k in fd], fontsize=8)
            ax.set_title(label_of(r.B), fontsize=8, color=INK)
            ax.set_xlabel("fold", fontsize=8, color=INK2)
            ax.set_ylim(ymin, ymax)
            for x, v in enumerate(fd.values()):
                if v < 0:
                    ax.text(x, v * 100 - 0.4, f"{v * 100:+.1f}", ha="center", va="top", fontsize=7, color=INK)
            _style(ax, "y")
        axes.flat[0].set_ylabel("Fusion gain (pp)", color=INK2)
        fig.suptitle("RQ1: accuracy gain from fusing with FER, per test fold", fontsize=10, color=INK)
        fig.tight_layout()
        save(fig, "fig_rq1_fold_gains.png")

    # 3. Paired differences for RQ1-RQ3 (forest plot)
    c = comps[comps["family"].isin(["rq1", "rq2", "rq3"])].reset_index(drop=True) if len(comps) else comps
    if len(c):
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(c) + 1.2))
        for i, r in c.iterrows():
            ax.plot([r["d_acc_lo"] * 100, r["d_acc_hi"] * 100], [i, i], color="#2a78d6", linewidth=2,
                    solid_capstyle="round")
            ax.scatter([r["d_acc"] * 100], [i], s=50, color="#2a78d6", edgecolor="white", linewidth=2, zorder=3)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(range(len(c)))
        ax.set_yticklabels([f"{r.family.upper()}: {label_of(r.A)} − {label_of(r.B)}" for r in c.itertuples()],
                           fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlabel("Difference in pooled accuracy (percentage points)", color=INK2)
        ax.set_title("Paired differences, 95% actor-bootstrap CI", fontsize=10, color=INK, loc="left")
        _style(ax)
        fig.tight_layout()
        save(fig, "fig_paired_differences.png")

    # 4. Pooled confusion matrices: speech only, FER only, fused
    names = [id2label[i] for i in range(len(id2label))]
    for fam, fname in (("Speech", "fig_confusion_speech.png"), ("FER", "fig_confusion_fer.png"),
                       ("Speech+FER", "fig_confusion_fused.png")):
        ss = [s for s in systems if family_of(s) == fam and s in pooled_cm]
        if not ss:
            continue
        ncol = min(3, len(ss))
        nrow = int(np.ceil(len(ss) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow), squeeze=False)
        for ax, s in zip(axes.flat, ss):
            cm = pooled_cm[s]
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(norm, cmap=blues, vmin=0, vmax=1)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.55 else INK)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_title(label_of(s), fontsize=8, color=INK)
            ax.set_xlabel("predicted", fontsize=7, color=INK2)
            ax.set_ylabel("true", fontsize=7, color=INK2)
        for ax in list(axes.flat)[len(ss):]:
            ax.axis("off")
        fig.suptitle("Pooled confusion matrices, % of true class (all folds and seeds)", fontsize=10, color=INK)
        fig.tight_layout()
        save(fig, fname)

    # 5. Per-class recall heatmap
    cols = [s for s in systems if s in recall.columns]
    if cols:
        fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 3.8))
        ax.imshow(recall[cols].values, cmap=blues, vmin=0, vmax=1, aspect="auto")
        for i in range(len(recall)):
            for j, s in enumerate(cols):
                v = recall[s].values[i]
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([label_of(s) for s in cols], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(recall)))
        ax.set_yticklabels(recall.index, fontsize=8)
        ax.set_title("Per-class recall (%), pooled over folds and seeds", fontsize=10, color=INK, loc="left")
        fig.tight_layout()
        save(fig, "fig_per_class_recall.png")

    # 6. Per-fold accuracy for RQ2 and RQ3 (two series per panel)
    pairs = [("RQ2", "w2v_frozen_matched", "e2v_frozen_matched"), ("RQ3", "w2v_lora", "w2v_wlp")]
    pairs = [p for p in pairs if all(s in set(runs["system"]) for s in p[1:])]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 3.2), squeeze=False, sharey=True)
        for ax, (rq, a, b) in zip(axes.flat, pairs):
            for s, col in ((a, "#2a78d6"), (b, "#eb6834")):
                gr = runs[runs["system"] == s].groupby("fold")["acc"]
                m, sd = gr.mean(), gr.std(ddof=1).fillna(0)
                ax.errorbar(np.arange(len(m)), m.values * 100, yerr=sd.values * 100, color=col, marker="o",
                            markersize=5, linewidth=2, capsize=0, label=label_of(s))
            ax.set_xticks(np.arange(len(m)))
            ax.set_xticklabels(m.index, fontsize=8)
            ax.set_ylabel("Accuracy (%)", color=INK2)
            ax.set_title(rq, fontsize=9, color=INK, loc="left")
            _style(ax, "y")
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle("Per-fold test accuracy (mean ± sd over seeds where available)", fontsize=10, color=INK)
        fig.tight_layout()
        save(fig, "fig_rq2_rq3_per_fold.png")
    return files


# ── Markdown ────────────────────────────────────────────────────────────────
def _pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}"


def _num(x, fmt=".2f"):
    return "–" if x is None or not np.isfinite(x) else format(x, fmt)


def _p(x):
    if x is None or not np.isfinite(x):
        return "–"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _d(r, key="acc"):
    return (f"{r[f'd_{key}'] * 100:+.1f} [{r[f'd_{key}_lo'] * 100:+.1f}, {r[f'd_{key}_hi'] * 100:+.1f}]")


def _verdict(r, alpha=0.05):
    """Plain-language verdict for 'A beats B' from the paired actor bootstrap."""
    if r["d_acc_lo"] > 0 and r["p_boot_acc_holm"] < alpha:
        return "A better (CI excludes 0)"
    if r["d_acc_hi"] < 0 and r["p_boot_acc_holm"] < alpha:
        return "B better (CI excludes 0)"
    if r["d_acc_lo"] > 0 or r["d_acc_hi"] < 0:
        return "CI excludes 0, not significant after Holm"
    return "no reliable difference"


def _comp_table(L, g):
    head = "| A | B | Seeds | Δ acc [95% CI] | p (boot, Holm) | Δ macro-F1 [95% CI] | Folds A>B | d_z | Wilcoxon p (r) | McNemar p per seed | Verdict |"
    L.append(head)
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in g.to_dict("records"):
        L.append(f"| {label_of(r['A'])} | {label_of(r['B'])} | {r['seeds']} | {_d(r)} | {_p(r['p_boot_acc_holm'])} | "
                 f"{_d(r, 'f1')} | {r['folds_A_better']}/{r['n_folds']} | {_num(r['cohen_dz'])} | "
                 f"{_p(r['wilcoxon_p'])} ({_num(r['rank_biserial'], '+.2f')}) | {r['mcnemar_p_per_seed']} | {_verdict(r)} |")
    L.append("")


def render_markdown(info, summary, comps, recall, inten, params, train_stats, selected, best_vs_next, figs):
    seeds = info["seeds_per_system"]
    design = info.get("seed_design") or seeds
    multi = [s for s in info["systems"] if len(design.get(s, [])) > 1]
    single = [s for s in info["systems"] if len(design.get(s, [])) == 1]

    def present(group):
        short = [f"{label_of(s)}: {len(seeds.get(s, []))}" for s in group if len(seeds.get(s, [])) < len(design[s])]
        return f" Seeds complete so far — {'; '.join(short)}." if short else ""
    L = ["# Actor-wise cross-validation results\n",
         "_Generated by `cv/analysis.py` from the saved run outputs. Do not edit by hand; re-run the analysis._\n"]
    if info.get("partial"):
        L.append(f"> **Partial results:** only {len(info['cv_folds'])} of {info['n_cv_folds_total']} CV folds are "
                 "present. 'Pooled' numbers cover only those folds' test actors. Do not report these.\n")

    # ── Protocol
    L.append("## Protocol\n")
    L.append(f"- **Cross-validation:** {info['n_cv_folds_total']}-fold actor-wise CV over the {info['n_actors_total']} RAVDESS "
             f"actors ({info['n_clips_total']} speech clips, 7 classes, calm merged into neutral). Each fold tests 4 actors "
             "(2 male, 2 female), validates on 4 different actors (2M/2F) and trains on the remaining 16. "
             "Every actor is tested exactly once and used for "
             f"validation exactly once (fold assignment seed {info['fold_seed']}, `results_cv/folds.json`).")
    L.append("- **No leakage:** checkpoint selection, hyperparameter selection and fusion temperatures use the "
             "validation actors only.")
    if multi:
        sm = sorted({sd for s in multi for sd in design[s]})
        L.append(f"- **{len(sm)} seeds ({', '.join(map(str, sm))}):** systems trained on cached frozen features, "
                 f"the FER model and their fusions — {', '.join(label_of(s) for s in multi)}.{present(multi)}")
    if single:
        L.append(f"- **Single seed (42):** systems with the backbone in the training loop and their fusions — "
                 f"{', '.join(label_of(s) for s in single)}. Their seed variance is not measured; their ± values "
                 "reflect variation across folds only. Fusions of single-seed systems use the seed-42 FER model."
                 f"{present(single)}")
    L.append("- **LoRA under the matched protocol was not evaluated.** Matching it would remove SpecAugment, which "
             "the LoRA recipe relies on against overfitting. The backbone comparison (RQ2) uses frozen encoders only, "
             "and RQ3 compares the LoRA and weighted-layer-probe pipelines, each with its own recipe.")
    L.append(f"- **Statistics:** pooled accuracy is computed over all {info['n_actors']} actors per seed and "
             f"averaged over the system's seeds; 95% CIs come from a cluster bootstrap over actors "
             f"({info['bootstrap_B']} resamples, stratified by gender). Paired comparisons re-score both systems "
             "on the seeds they share (all three when both are multi-seed, otherwise seed 42) with the same "
             "resampled actors; p-values are Holm-adjusted within each table. With 6 folds the smallest possible "
             "two-sided Wilcoxon p is 0.031, so verdicts rest on the actor bootstrap (95% CI excluding 0 and "
             "Holm-adjusted p < 0.05).\n")

    # ── Main table
    L.append("## 1. Main results\n")
    L.append("Per-run = one fold x seed (4 test actors, 240 clips); for multi-seed systems the ± mixes fold and "
             f"seed variation. Pooled = all {info['n_actors']} actors, mean over the system's seeds. All values in %.\n")
    L.append("| System | Seeds | Acc. mean ± sd | Macro-F1 mean ± sd | Pooled acc. [95% CI] | Pooled macro-F1 [95% CI] |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in summary.to_dict("records"):
        L.append(f"| {r['label']} | {r['n_seeds']} | {_pct(r['acc_mean'])} ± {_pct(r['acc_std'])} | "
                 f"{_pct(r['f1_mean'])} ± {_pct(r['f1_std'])} | {_pct(r.get('pooled_acc'))} "
                 f"[{_pct(r.get('pooled_acc_lo'))}, {_pct(r.get('pooled_acc_hi'))}] | {_pct(r.get('pooled_f1'))} "
                 f"[{_pct(r.get('pooled_f1_lo'))}, {_pct(r.get('pooled_f1_hi'))}] |")
    L.append("")

    # ── RQ1
    g1 = comps[comps["family"] == "rq1"] if len(comps) else comps
    if len(g1):
        L.append("## 2. RQ1 — Does calibrated late fusion with FER improve SER across actors?\n")
        L.append("Gain = fused − speech only, on the same clips. Per-fold columns are the gain in each test fold "
                 "(mean over seeds for multi-seed systems), in percentage points.\n")
        fold_names = info["cv_folds"]
        L.append("| Speech system | Seeds | Speech acc. | Fused acc. | Gain [95% CI] | p (boot, Holm) | "
                 + " | ".join(f.replace("fold", "F") for f in fold_names) + " | Folds > 0 | Min fold gain | Verdict |")
        L.append("|---|---|---:|---:|---:|---:|" + "---:|" * len(fold_names) + "---:|---:|---|")
        n_all_pos = 0
        for r in g1.to_dict("records"):
            fd = json.loads(r["fold_diffs"])
            all_pos = all(v > 0 for v in fd.values()) and len(fd) == len(fold_names)
            n_all_pos += all_pos
            L.append(f"| {label_of(r['B'])} | {r['seeds']} | {_pct(r['acc_B'])} | {_pct(r['acc_A'])} | {_d(r)} | "
                     f"{_p(r['p_boot_acc_holm'])} | " + " | ".join(f"{fd[f] * 100:+.1f}" if f in fd else "–"
                                                                    for f in fold_names)
                     + f" | {r['folds_A_better']}/{r['n_folds']} | {r['fold_diff_min'] * 100:+.1f} | "
                     + ("fusion better" if _verdict(r).startswith("A better") else _verdict(r)) + " |")
        L.append("")
        multi_runs = [f"{label_of(r['B'])} {r['runs_A_better']}/{r['n_runs']}"
                      for r in g1.to_dict("records") if r["n_seeds"] > 1]
        L.append(f"**Fusion gain is positive in every test fold for {n_all_pos} of {len(g1)} speech systems.**"
                 + (" Positive fold × seed runs for multi-seed systems: " + "; ".join(multi_runs) + "." if multi_runs else "")
                 + "\n")
        gf = comps[comps["family"] == "rq1_face"]
        if len(gf):
            L.append("Fused system vs FER alone:\n")
            _comp_table(L, gf)

    # ── RQ2
    g2 = comps[comps["family"] == "rq2"] if len(comps) else comps
    L.append("## 3. RQ2 — Frozen wav2vec 2.0 vs frozen emotion2vec under a matched probe and recipe\n")
    L.append("**Matched protocol** (identical for both backbones):\n")
    L.append("- *Probe:* one shared probe class — learned softmax weights over all 13 hidden states each backbone "
             "exposes (wav2vec2-base: CNN/positional output + 12 transformer layers; emotion2vec_base: input to the "
             "first block + 4 context-encoder blocks + 8 blocks, read with forward hooks), then "
             "Linear(768→128) → ReLU → Dropout → Linear(128→7).")
    L.append("- *Masked pooling:* each clip is fed at its true length (no padding enters either backbone), so the "
             "time average covers valid frames only.")
    L.append("- *Backbones frozen and in eval mode* (no internal dropout, LayerDrop or SpecAugment); no augmentation "
             "for either backbone.")
    L.append("- *Recipe:* AdamW, cosine schedule, class-weighted cross-entropy, batch 8, AMP, weight decay 1e-2, up "
             "to 30 epochs, patience 6; the same 4-setting grid (lr ∈ {1e-3, 3e-4} × dropout ∈ {0.1, 0.3}) for both, "
             "selected per fold and seed by the lowest validation loss.\n")
    if len(params):
        L.append("Trainable parameters (all systems in this document):\n")
        L.append("| System | Layer weights | Gate | LoRA | Head | **Trainable total** | Frozen backbone |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in params.to_dict("records"):
            L.append(f"| {r['label']} | {r['layer_weights']:,} | {r['weight_gate']:,} | {r['lora']:,} | {r['head']:,} | "
                     f"**{r['trainable_total']:,}** | {r['frozen']:,} |")
        L.append("")
    if len(g2):
        L.append("Paired comparison (A − B):\n")
        _comp_table(L, g2)
        for r in g2.to_dict("records"):
            fd = json.loads(r["fold_diffs"])
            L.append(f"- {r['description'][0].upper() + r['description'][1:]}: per-fold Δ " + ", ".join(f"{k} {v * 100:+.1f}" for k, v in fd.items())
                     + f"; wav2vec 2.0 better in {r['folds_A_better']}/{r['n_folds']} folds.")
        if len(selected):
            sel = selected[selected["system"].isin(["w2v_frozen_matched", "e2v_frozen_matched"])]
            if len(sel):
                L.append("- Validation-selected setting (count over fold × seed): " + "; ".join(
                    f"{label_of(s)}: " + ", ".join(f"{c} ×{n}" for c, n in zip(g["cfg"], g["times_selected"]))
                    for s, g in sel.groupby("system")) + ".")
        L.append("")

    # ── RQ3
    g3 = comps[comps["family"] == "rq3"] if len(comps) else comps
    if len(g3):
        L.append("## 4. RQ3 — wav2vec 2.0: does LoRA improve over a frozen weighted-layer probe?\n")
        L.append("LoRA pipeline (r = 4, alpha = 4, dropout 0.25 on q/v projections, following "
                 "Lashkarashvili et al.; fixed a priori, not tuned) vs the frozen weighted-layer probe, both "
                 "with their own recipes and SpecAugment, seed 42, paired over the CV folds. The two pipelines "
                 "have different heads (frame-level MLP vs gate + LayerNorm + linear), so this compares pipelines "
                 "rather than isolating the effect of adaptation.\n")
        _comp_table(L, g3)
        for r in g3.to_dict("records"):
            fd = json.loads(r["fold_diffs"])
            L.append(f"- {r['description'][0].upper() + r['description'][1:]}: per-fold Δ (LoRA − frozen) "
                     + ", ".join(f"{k} {v * 100:+.1f}" for k, v in fd.items()) + ".")
        L.append("")

    # ── Per-class recall, intensity
    L.append("## 5. Per-class recall and confusion matrices\n")
    L.append("Recall pooled over all folds and the system's seeds (%). Confusion matrices: `fig_confusion_speech.png`, "
             "`fig_confusion_fer.png`, `fig_confusion_fused.png` and `confusion_matrices.json`.\n")
    cols = list(recall.columns)
    L.append("| Class | " + " | ".join(label_of(s) for s in cols) + " |")
    L.append("|---|" + "---:|" * len(cols))
    L.append("| *seeds* | " + " | ".join(str(len(seeds.get(s, []))) for s in cols) + " |")
    for cls, row in recall.iterrows():
        L.append(f"| {cls} | " + " | ".join(_pct(v, 0) for v in row.values) + " |")
    L.append("")
    L.append("## 6. Accuracy by emotional intensity (%)\n")
    L.append("Pooled per seed, mean over seeds; ± is the sd over fold × seed runs.\n")
    L.append("| System | Seeds | Normal | Strong |")
    L.append("|---|---:|---:|---:|")
    for r in inten.to_dict("records"):
        L.append(f"| {label_of(r['system'])} | {r['n_seeds']} | {_pct(r['normal_acc'])} ± {_pct(r['normal_sd_runs'])} | "
                 f"{_pct(r['strong_acc'])} ± {_pct(r['strong_sd_runs'])} |")
    L.append("")

    # ── Best system
    if best_vs_next is not None:
        L.append("## 7. Best fused system vs runner-up\n")
        _comp_table(L, pd.DataFrame([best_vs_next]))
        L.append("")
    tr = info.get("temperature_range")
    if tr:
        L.append(f"Fusion temperatures (T = exp(log T), fitted on the validation actors): {tr[0]:.2f}–{tr[1]:.2f}.\n")
    L.append("## 8. Relation to prior protocols\n")
    L.append("Luna-Jiménez et al. (2022) used subject-wise 5-fold cross-validation on RAVDESS with no separate test "
             "set: the held-out actors of each fold served both for early stopping and for reporting, and no seeds "
             "were repeated. This protocol keeps validation and test actors separate in every fold, so model "
             "selection never sees the reported actors, and repeats the cheap systems over three seeds. Numbers "
             "from the two protocols are not directly comparable.\n")

    # ── Limitations
    L.append("## 9. Limitations\n")
    L.append("- Backbone-in-the-loop systems (W2V2-WLP, W2V2-LoRA) were run with one seed, "
             "so their seed-to-seed variance is unknown; RQ3 rests on a single seed.")
    L.append("- LoRA was not evaluated under the matched protocol, so the backbone comparison (RQ2) covers frozen "
             "encoders only, and RQ3 compares whole pipelines with different heads.")
    L.append("- The 6 folds share training actors, so per-fold results are not independent, and 24 actors leave "
             "substantial uncertainty about new speakers. McNemar treats clips as independent and ignores grouping "
             "by actor, so its p-values are optimistic; verdicts use the actor bootstrap.")
    L.append("- RAVDESS is acted, studio-recorded North-American English speech with fixed sentences; results may "
             "not carry over to spontaneous or conversational emotion.\n")

    # ── Appendix
    go = comps[comps["family"] == "other"] if len(comps) else comps
    if len(go):
        L.append("## Appendix A. Other paired comparisons\n")
        _comp_table(L, go)
    if len(train_stats):
        L.append("## Appendix B. Training statistics\n")
        L.append("| System | Seeds | Runs | Epochs trained (mean) | Best epoch (mean) | Wall time per run (min) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for r in train_stats.to_dict("records"):
            L.append(f"| {label_of(r['system'])} | {r['n_seeds']} | {r['runs']} | {r['epochs_mean']:.1f} | "
                     f"{r['best_epoch_mean']:.1f} | {r['wall_min_mean']:.1f} |")
        L.append("")
    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"![{f}]({f})\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    analyse(a.results, a.out, B=a.bootstrap, make_figures=not a.no_figures)


if __name__ == "__main__":
    main()
