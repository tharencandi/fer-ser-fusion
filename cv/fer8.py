"""FER-only evaluation with 8 classes (calm kept apart from neutral) and with 7 classes,
for comparison with published RAVDESS face-only results.

    python -m cv.fer8 --blendshapes "<dir with the blendshape CSVs>" --out results_cv/fer8

Same feature pipeline (376 blendshape statistics) and XGBRF settings as the main
study; every clip, calm included, is paired with the blendshapes of its own
recording. Two actor-wise protocols, both test-only (XGBRF
has fixed hyperparameters and no early stopping, so no validation actors are
used): train on every actor that is not held out.

    cv6   the 6 folds of results_cv/folds.json (4 test actors each, train on 20)
    luna5 the 5 folds of Luna-Jiménez et al. (2022), train on 19 or 20

Accuracy and macro-F1 are reported as the mean over folds (their aggregation)
and pooled over all 1440 clips with a 95% actor-bootstrap CI; both are averaged
over the seeds.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from .analysis import ActorBootstrap, acc_f1_from_cm, confusion
from .data import align_blendshapes, load_ravdess_blendshape_data
from .fer import fer_features, train_xgbrf

LUNA_FOLDS = [[2, 5, 14, 15, 16], [3, 6, 7, 13, 18], [10, 11, 12, 19, 20], [8, 17, 21, 23, 24], [1, 4, 9, 22]]
CLASSES8 = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]   # RAVDESS order
CLASSES7 = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]           # main study
PUBLISHED = [("Luna-Jiménez et al. (2022), OpenFace AUs + bi-LSTM", 0.6213),
             ("Luna-Jiménez et al. (2022), static MLP", 0.6022)]


def protocols(folds_json):
    with open(folds_json) as fh:
        cv6 = [f["test"] for f in json.load(fh)["folds"]]
    for name, folds in (("cv6", cv6), ("luna5", LUNA_FOLDS)):
        assert sorted(a for f in folds for a in f) == list(range(1, 25)), f"{name}: every actor once"
    return {"cv6": cv6, "luna5": LUNA_FOLDS}


def run(blendshape_dir, clips_csv, folds_json, seeds=(42, 43, 44)):
    clips = pd.read_csv(clips_csv)
    aligned, report = align_blendshapes(clips, load_ravdess_blendshape_data(blendshape_dir))
    if report["n_audio_without_face"] or report["n_clips_sharing_a_face_file"]:
        raise ValueError(f"alignment incomplete: {report}")
    X = fer_features(aligned)
    meta = clips.set_index("idx").loc[aligned.index]
    y = {8: meta["emotion_raw"].map({c: i for i, c in enumerate(CLASSES8)}).values,
         7: meta["label"].values}
    actors = meta["actor"].values
    rows = []
    for proto, folds in protocols(folds_json).items():
        for k, test in enumerate(folds):
            te = np.isin(actors, test)
            for n_cls in (8, 7):
                for seed in seeds:
                    clf = train_xgbrf(X[~te], y[n_cls][~te], seed)
                    pred = clf.predict(X[te])
                    print(f"{proto} fold {k} {n_cls}-class seed {seed}: acc {np.mean(pred == y[n_cls][te]):.3f}", flush=True)
                    rows.append(pd.DataFrame({"protocol": proto, "fold": k, "n_classes": n_cls, "seed": seed,
                                              "idx": aligned.index[te], "actor": actors[te],
                                              "label": y[n_cls][te], "pred": pred}))
    return pd.concat(rows, ignore_index=True), report


def analyse(preds, out_dir, B=10000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(out_dir, exist_ok=True)
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    boot = ActorBootstrap(sorted(preds["actor"].unique()), B=B, seed=0)
    actor_of = preds.drop_duplicates("idx").set_index("idx")["actor"]
    summary, recall_rows, cms = [], {}, {}
    for (proto, n_cls), g in preds.groupby(["protocol", "n_classes"]):
        names = CLASSES8 if n_cls == 8 else CLASSES7
        seeds = sorted(g["seed"].unique())
        per_fold = pd.DataFrame([{"fold": f, "seed": s, "acc": acc_f1_from_cm(confusion(h["label"], h["pred"], n_cls))[0],
                                  "f1": acc_f1_from_cm(confusion(h["label"], h["pred"], n_cls))[1]}
                                 for (f, s), h in g.groupby(["fold", "seed"])])
        fold_mean = per_fold.groupby("seed")[["acc", "f1"]].mean()          # their aggregation, per seed
        by_fold = per_fold.groupby("fold")[["acc", "f1"]].mean()             # averaged over seeds
        c = boot.cm_by_actor(g, seeds, n_cls, actor_of)
        (pa, pf), (ba, bf) = boot.stats(c)
        cm = c.sum((0, 1)) / len(seeds)                                     # mean counts per seed
        cms[(proto, n_cls)] = cm
        recall_rows[f"{proto}_{n_cls}"] = pd.Series(np.diag(cm) / cm.sum(1), index=names)
        summary.append({"protocol": proto, "n_classes": n_cls, "n_folds": g["fold"].nunique(), "seeds": len(seeds),
                        "fold_mean_acc": fold_mean["acc"].mean(), "fold_mean_acc_seed_sd": fold_mean["acc"].std(ddof=1),
                        "fold_sd_acc": by_fold["acc"].std(ddof=1),
                        "fold_mean_f1": fold_mean["f1"].mean(), "fold_sd_f1": by_fold["f1"].std(ddof=1),
                        "pooled_acc": pa, "pooled_acc_lo": np.percentile(ba, 2.5), "pooled_acc_hi": np.percentile(ba, 97.5),
                        "pooled_f1": pf, "pooled_f1_lo": np.percentile(bf, 2.5), "pooled_f1_hi": np.percentile(bf, 97.5),
                        "per_fold_acc": json.dumps({int(k): round(float(v), 4) for k, v in by_fold["acc"].items()})})
        if n_cls == 8:
            i_n, i_c = CLASSES8.index("neutral"), CLASSES8.index("calm")
            summary[-1].update(calm_as_neutral=cm[i_c, i_n] / cm[i_c].sum(), neutral_as_calm=cm[i_n, i_c] / cm[i_n].sum(),
                               calm_recall=cm[i_c, i_c] / cm[i_c].sum(), neutral_recall=cm[i_n, i_n] / cm[i_n].sum())
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    recall = pd.DataFrame(recall_rows)
    recall.to_csv(os.path.join(out_dir, "per_class_recall.csv"))
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as fh:
        json.dump({f"{p}_{n}": {"classes": CLASSES8 if n == 8 else CLASSES7, "mean_counts_per_seed": np.round(m, 2).tolist()}
                   for (p, n), m in cms.items()}, fh, indent=1)

    blues = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, proto, title in zip(axes, ("cv6", "luna5"), ("6-fold actor-wise CV", "Luna-Jiménez et al. 5 folds")):
        cm = cms[(proto, 8)]
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=blues, vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(8)); ax.set_xticklabels(CLASSES8, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(8)); ax.set_yticklabels(CLASSES8, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        ax.set_title(f"FER, 8 classes: {title}", fontsize=9)
    fig.suptitle("Pooled confusion, % of true class (mean over 3 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_fer8_confusion.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(out_dir, "RESULTS_FER8.md"), "w") as fh:
        fh.write(render(summary, recall, cms))
    return summary


def _p(x):
    return f"{x * 100:.1f}"


def render(summary, recall, cms):
    S = summary.set_index(["protocol", "n_classes"])
    L = ["# FER only: 8 classes (calm vs neutral) and 7 classes\n",
         "_Generated by `python -m cv.fer8`. Face model: 376 MediaPipe-blendshape statistics + XGBRFClassifier, "
         "same features and hyperparameters as the main study, seeds 42, 43, 44. Every clip, calm "
         "included, is paired with the blendshapes of its own recording._\n",
         "Both protocols hold out whole actors and train on all other actors (no validation actors: XGBRF has fixed "
         "hyperparameters and no early stopping). Fold mean = mean of the per-fold accuracies (the aggregation of "
         "Luna-Jiménez et al.), ± the sd across folds; pooled = all 1440 clips with a 95% actor-bootstrap CI. All "
         "values in %, averaged over the 3 seeds.\n",
         "| Protocol | Classes | Folds | Fold-mean acc. ± sd | Pooled acc. [95% CI] | Fold-mean macro-F1 ± sd | Pooled macro-F1 [95% CI] |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    pname = {"cv6": "6-fold actor-wise CV (this study)", "luna5": "Luna-Jiménez et al. 5 folds"}
    for (p, n), r in S.sort_index(level=[0, 1], ascending=[True, False]).iterrows():
        L.append(f"| {pname[p]} | {n} | {r.n_folds} | {_p(r.fold_mean_acc)} ± {_p(r.fold_sd_acc)} | {_p(r.pooled_acc)} "
                 f"[{_p(r.pooled_acc_lo)}, {_p(r.pooled_acc_hi)}] | {_p(r.fold_mean_f1)} ± {_p(r.fold_sd_f1)} | "
                 f"{_p(r.pooled_f1)} [{_p(r.pooled_f1_lo)}, {_p(r.pooled_f1_hi)}] |")
    L.append("\nPer-fold accuracy (8 classes): " + "; ".join(
        f"{pname[p]}: " + ", ".join(f"{v * 100:.1f}" for v in json.loads(S.loc[(p, 8), 'per_fold_acc']).values())
        for p in ("cv6", "luna5")) + ".\n")

    L.append("## Calm vs neutral\n")
    L.append("| Protocol | Calm recall | Neutral recall | Calm predicted as neutral | Neutral predicted as calm |")
    L.append("|---|---:|---:|---:|---:|")
    for p in ("cv6", "luna5"):
        r = S.loc[(p, 8)]
        L.append(f"| {pname[p]} | {_p(r.calm_recall)} | {_p(r.neutral_recall)} | {_p(r.calm_as_neutral)} | {_p(r.neutral_as_calm)} |")
    L.append("\nRAVDESS has 192 calm clips (normal and strong intensity) and 96 neutral clips (normal intensity only).\n")

    L.append("## Per-class recall (%), pooled over folds, mean over seeds\n")
    cols = [c for c in ("cv6_8", "luna5_8") if c in recall]
    L.append("| Class | " + " | ".join(pname[c.split("_")[0]] for c in cols) + " |")
    L.append("|---|" + "---:|" * len(cols))
    for cls in CLASSES8:
        L.append(f"| {cls} | " + " | ".join(_p(recall.loc[cls, c]) for c in cols) + " |")
    L.append("")

    L.append("## 8-class confusion matrices (mean counts per seed; rows true, columns predicted)\n")
    for p in ("cv6", "luna5"):
        cm = cms[(p, 8)]
        L.append(f"**{pname[p]}**\n")
        L.append("| true \\ pred | " + " | ".join(c[:4] for c in CLASSES8) + " |")
        L.append("|---|" + "---:|" * 8)
        for i, cls in enumerate(CLASSES8):
            L.append(f"| {cls} | " + " | ".join(f"{v:.0f}" for v in cm[i]) + " |")
        L.append("")
    L.append("![8-class confusion](fig_fer8_confusion.png)\n")

    r8 = S.loc[("luna5", 8)]
    L.append("## Comparison with published face-only results\n")
    L.append("| Face-only system, 8 classes, Luna-Jiménez et al. folds | Accuracy (fold mean) |")
    L.append("|---|---:|")
    for name, acc in PUBLISHED:
        L.append(f"| {name} | {acc * 100:.2f} |")
    L.append(f"| This study: MediaPipe blendshape statistics + XGBRF (3 seeds) | {_p(r8.fold_mean_acc)} "
             f"(pooled {_p(r8.pooled_acc)}, 95% CI {_p(r8.pooled_acc_lo)}–{_p(r8.pooled_acc_hi)}) |")
    diff = (r8.fold_mean_acc - PUBLISHED[0][1]) * 100
    L.append("")
    L.append(f"On the same five actor folds and the same eight classes, the blendshape model scores "
             f"{_p(r8.fold_mean_acc)}% (fold mean), {abs(diff):.1f} points {'above' if diff >= 0 else 'below'} the "
             f"AU + bi-LSTM model ({PUBLISHED[0][1] * 100:.2f}%) and "
             f"{abs(r8.fold_mean_acc - PUBLISHED[1][1]) * 100:.1f} points "
             f"{'above' if r8.fold_mean_acc >= PUBLISHED[1][1] else 'below'} the static MLP "
             f"({PUBLISHED[1][1] * 100:.2f}%). The comparison is indicative only:\n")
    L.append("- **Different face features.** This study summarises MediaPipe Face Landmarker blendshapes (52 ARKit "
             "coefficients + `_neutral`) into 376 clip-level statistics; Luna-Jiménez et al. use OpenFace facial "
             "action units.")
    L.append("- **No model selection on held-out actors here.** XGBRF runs with fixed hyperparameters "
             "(set in advance, not tuned on these folds) and no early stopping, so in these runs "
             "the held-out actors are used only for testing.")
    L.append("- **Seeds.** Our numbers are averaged over three seeds, with an actor-bootstrap CI for the pooled "
             "accuracy.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blendshapes", required=True)
    ap.add_argument("--clips", default=os.path.join(os.path.dirname(__file__), "..", "results_cv", "clips.csv"))
    ap.add_argument("--folds", default=os.path.join(os.path.dirname(__file__), "..", "results_cv", "folds.json"))
    ap.add_argument("--out", default="results_cv/fer8")
    ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*XGBRFClassifier. is deprecated", category=FutureWarning)
    preds, report = run(a.blendshapes, a.clips, a.folds)
    print("alignment:", report)
    s = analyse(preds, a.out, B=a.bootstrap)
    print(s[["protocol", "n_classes", "fold_mean_acc", "pooled_acc", "pooled_acc_lo", "pooled_acc_hi"]].round(4).to_string())
    print(f"wrote {a.out}/RESULTS_FER8.md")


if __name__ == "__main__":
    main()
