# Hearing and seeing emotion: speech + face fusion on RAVDESS

**Interactive overview: [tharencandi.net/fer-ser-fusion](https://tharencandi.net/fer-ser-fusion/)** · [Full report](https://tharencandi.net/fer-ser-fusion/report/) · [PDF](report/report.pdf)

Multimodal emotion recognition on the RAVDESS speech subset. Self-supervised speech models
(wav2vec 2.0 and emotion2vec) are combined with a facial-expression model by
temperature-calibrated late fusion, and every system is evaluated with 6-fold actor-wise
cross-validation, so each of the 24 actors is tested once on models that never saw them.

| Model | Seeds | Accuracy [95% CI] | Macro-F1 |
|---|---:|---:|---:|
| wav2vec 2.0, frozen, matched probe | 3 | 72.0% [66.8, 76.6] | 0.716 |
| emotion2vec, frozen, matched probe | 3 | 68.7% [65.3, 71.7] | 0.675 |
| wav2vec 2.0, frozen, weighted-layer probe | 1 | 66.9% [63.7, 70.0] | 0.649 |
| wav2vec 2.0, LoRA (r = 4) | 1 | 73.2% [69.1, 76.9] | 0.727 |
| emotion2vec, frozen, final-layer probe | 3 | 66.9% [63.5, 70.3] | 0.655 |
| Face: MediaPipe blendshapes + XGBoost random forest | 3 | 60.4% [56.6, 64.3] | 0.589 |
| wav2vec 2.0 matched probe + face | 3 | 77.5% [73.7, 81.4] | 0.769 |
| emotion2vec matched probe + face | 3 | 76.4% [72.8, 80.0] | 0.754 |
| wav2vec 2.0 weighted-layer probe + face | 1 | 74.4% [71.0, 77.8] | 0.730 |
| **wav2vec 2.0 LoRA + face** | 1 | **78.5% [75.0, 81.9]** | **0.778** |
| emotion2vec final-layer probe + face | 3 | 74.4% [71.1, 77.6] | 0.733 |

Accuracy pooled over all 1440 clips (24 actors), mean over seeds, with 95% confidence intervals
from a bootstrap over actors. 7 classes, calm merged into neutral.

## Findings

- **Fusion helps every speech model** (RQ1): +5.3 to +7.7 accuracy points, every confidence
  interval above zero, and positive in every test fold for three of the five speech models. The
  face model is strong where speech is weak (happy) and weak where speech is strong (surprised,
  fearful).
- **wav2vec 2.0 vs emotion2vec** (RQ2): with the same probe, recipe and tuning budget, the two
  frozen encoders are not reliably different (+3.3 points for wav2vec 2.0, 95% CI −0.6 to +7.0).
- **LoRA vs frozen wav2vec 2.0** (RQ3): the LoRA pipeline beats the frozen weighted-layer probe
  pipeline by 6.2 points alone and 4.1 points with the face model (one seed; the two pipelines
  also have different heads).
- **The face model is in the published range.** On the 5 actor folds and 8 classes of
  Luna-Jiménez et al. (2022) it reaches 57.9% (fold mean), against 62.13% for their OpenFace AU +
  bi-LSTM and 60.22% for their static MLP, with no held-out actors used for model selection
  (report Appendix B, `results_cv/fer8/`).

See `results_cv/RESULTS_CV.md` and the report for the full analysis.

## Method

- **Data:** RAVDESS speech (`xbgoose/ravdess` on Hugging Face), 6 actor-wise folds: each tests 4
  actors (2 male, 2 female), validates on 4 others and trains on 16 (`results_cv/folds.json`).
  Model selection and fusion temperatures use the validation actors only.
- **Seeds:** systems trained on cached frozen features, the face model and their fusions use
  seeds 42, 43, 44; the systems with the encoder in the training loop use seed 42.
- **Speech:** 16 kHz audio up to 5.27 s.
  - Matched probes for the fair encoder comparison: frozen
    [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (`facebook/wav2vec2-base`) and frozen
    [emotion2vec](https://arxiv.org/abs/2312.15185) (`iic/emotion2vec_base`), each with the same
    probe over all 13 hidden states (99,348 trainable parameters), the same recipe and the same
    hyperparameter grid; clips are fed at their true length.
  - Three pipeline systems, each with its own head and fixed recipe: the layer-weighted wav2vec 2.0 probe of
    [Pepino et al. (2021)](https://arxiv.org/abs/2104.03502), LoRA wav2vec 2.0 (rank 4 on
    `q_proj`/`v_proj`, after [Lashkarashvili et al. (2024)](https://arxiv.org/abs/2402.11747)),
    and an emotion2vec probe on the final layer.
- **Face:** MediaPipe blendshapes per frame (53 channels: the 52 ARKit blendshapes plus
  `_neutral`), summarised into 376 features: 7 statistics per channel plus 5 region ratios.
  Classified by `XGBRFClassifier`.
- **Fusion:** each branch's logits are divided by a positive temperature fitted on the validation
  actors and turned into probabilities with a softmax; the two are summed and the highest is the
  prediction.

## Repository

```
index.html, assets/   interactive overview (GitHub Pages); assets/data.js is generated from results_cv/
report/               report source (report.md), rendered page (index.html), PDF and figures
notebook/             ser_fer_fusion_ravdess_cv.ipynb: cross-validation (Google Colab, with outputs)
                      fer8_eval.ipynb: face model alone, 8 and 7 classes (report Appendix B)
cv/                   code for the cross-validation: data, folds, models, training, FER, fusion,
                      resumable runner, analysis
results_cv/           fold assignment, RESULTS_CV.md, tables (CSV/JSON) and figures
tests/                unit tests (python -m pytest tests -q)
```

The notebooks run on Colab. Each run's outputs are saved to Drive as it finishes, so the
cross-validation resumes after a disconnect.

**Blendshape data.** The per-frame MediaPipe blendshapes of every RAVDESS speech video (2880 CSV
files, one per video) are public: [Google Drive folder](https://drive.google.com/drive/folders/1omlBivvIL68xhqAdRY5lSjJNIO8DaIN7). To run the notebooks, add the
folder to your Drive (or download it and upload it) so that it sits at
`My Drive/ravdess-videos/blendshapes`, or point `BLENDSHAPE_DATA_PATH` in the notebook's config
cell at wherever you put it. Outside Colab, download it and pass the folder directly, e.g.
`python -m cv.fer8 --blendshapes <folder>`.

To regenerate the analysis and the page data from the saved run outputs (no training):

```
python -m cv.analysis --results "<RESULTS_DIR>" --out results_cv
python -m cv.site_data --analysis results_cv --out assets/data.js
```

To rebuild the report page:

```
pandoc report/report.md -s --mathml --shift-heading-level-by=-1 --metadata lang=en \
  --metadata pagetitle="Multimodal Speech Emotion Recognition on RAVDESS" \
  --include-before-body report/nav.html -o report/index.html
```

## Data and licence

RAVDESS: S. R. Livingstone and F. A. Russo (2018), "The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS)", *PLOS ONE* 13(5): e0196391. The 14 example clips in
`assets/audio/` are unmodified RAVDESS files (actors 01 and 02), redistributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The [blendshape files](https://drive.google.com/drive/folders/1omlBivvIL68xhqAdRY5lSjJNIO8DaIN7)
are derived from the RAVDESS videos and are shared under the same licence.
