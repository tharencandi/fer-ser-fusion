# Hearing and seeing emotion: speech + face fusion on RAVDESS

**Interactive overview: [tharencandi.net/fer-ser-fusion](https://tharencandi.net/fer-ser-fusion/)** · [Full report](https://tharencandi.net/fer-ser-fusion/report/) · [PDF](report/report.pdf)

Multimodal emotion recognition on the RAVDESS speech subset. Self-supervised speech models
(wav2vec 2.0 and emotion2vec) are combined with a facial-expression model by
temperature-calibrated late fusion, evaluated on actors unseen in training.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| wav2vec 2.0, frozen (weighted-layer linear probe) | 71.3% | 0.696 |
| wav2vec 2.0, LoRA (r = 4) | 72.5% | 0.702 |
| emotion2vec, frozen (linear probe) | 68.3% | 0.668 |
| Face: MediaPipe blendshapes + XGBoost random forest | 65.0% | 0.650 |
| wav2vec 2.0 frozen + face | 82.1% | 0.813 |
| **wav2vec 2.0 LoRA + face** | **86.7%** | **0.860** |
| emotion2vec + face | 85.8% | 0.854 |

240 test clips from 4 held-out actors, 7 classes (calm merged into neutral).

## Findings

- **Fusion helps every speech model**, by +10.8, +14.2 and +17.5 accuracy points. The face
  model is strong where speech is weak (happy, sad) and weak where speech is strong
  (neutral, surprised).
- **LoRA beats the frozen probe**, slightly alone and clearly after fusion.
- **Emotion-specific pretraining didn't win**: emotion2vec scored below both wav2vec 2.0
  pipelines on its own.

Results come from one seed and one actor-disjoint split, without confidence intervals.

## Method

- **Data:** RAVDESS speech (`xbgoose/ravdess` on Hugging Face), actor-disjoint split of
  16 / 4 / 4 actors (960 / 240 / 240 clips), seed 42.
- **Speech:** 16 kHz audio up to 5.27 s.
  - [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (`facebook/wav2vec2-base`), frozen, with
    the layer-weighted probe of [Pepino et al. (2021)](https://arxiv.org/abs/2104.03502):
    a learned weighting of all 13 hidden states and a 128-unit frame-level dense layer.
  - wav2vec 2.0 with LoRA (rank 4 on `q_proj`/`v_proj`), weighted layer sum and weight
    gating, after [Lashkarashvili et al. (2024)](https://arxiv.org/abs/2402.11747), with a
    single LayerNorm + linear classifier.
  - [emotion2vec](https://arxiv.org/abs/2312.15185) (`iic/emotion2vec_base`), frozen, with a
    small MLP head.
- **Face:** MediaPipe blendshapes per frame (53 channels: the 52 ARKit blendshapes plus
  `_neutral`), summarised into 376 features: 7 statistics per channel plus 5 region ratios.
  Classified by `XGBRFClassifier`.
- **Fusion:** each branch's logits are divided by a temperature fitted on the validation
  actors and turned into probabilities with a softmax; the two are summed and the highest
  is the prediction.

## Repository

```
index.html, assets/   interactive overview (GitHub Pages)
report/               report source (report.md), rendered page (index.html) and PDF
notebook/             experiment notebook with outputs (Google Colab)
```

The notebook was run on Colab. It reads precomputed blendshape features and model checkpoints
from Google Drive, which aren't included here.

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
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
