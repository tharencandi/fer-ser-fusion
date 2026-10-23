"""RAVDESS loading, clip metadata, audio preprocessing and blendshape alignment.

Every clip gets an explicit metadata row (actor, gender, intensity, RAVDESS file
id) so that outputs can be saved and joined by clip. The audio/blendshape merge key
uses the raw RAVDESS emotion, so `calm` and `neutral` stay apart (see
`align_blendshapes`).
"""
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# ── Constants ───────────────────────────────────────────────────────────────
RAVDESS_REQUIRED_COLUMNS = [
    "audio", "modality", "vocal_channel", "emotion", "emotional_intensity",
    "statement", "repetition", "actor", "gender",
]
INTENSITY_MAP = {1: "normal", 2: "strong"}
STATEMENT_MAP = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}
REPETITION_MAP = {1: "1st repetition", 2: "2nd repetition"}
RAVDESS_EMOTION_ID_TO_NAME = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}
EMOTION_MERGE_MAP = {"calm": "neutral"}

ARKIT_52_COLS = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
assert len(ARKIT_52_COLS) == 53


def merge_emotion(name):
    name = str(name).lower().strip()
    return EMOTION_MERGE_MAP.get(name, name)


def _inverse(mapping):
    return {str(v).lower(): k for k, v in mapping.items()}


# ── Dataset and clip table ──────────────────────────────────────────────────
def load_ravdess(dataset_id):
    """Load the RAVDESS speech set (the HF 'train' split is the whole dataset)."""
    if dataset_id == "synthetic":
        from .testing import synthetic_ravdess
        raw = synthetic_ravdess()
    else:
        from datasets import load_dataset
        raw = load_dataset(dataset_id, split="train")
    missing = [c for c in RAVDESS_REQUIRED_COLUMNS if c not in raw.column_names]
    if missing:
        raise ValueError(f"Missing expected RAVDESS columns: {missing}")
    return raw


def label_space(raw):
    """Sorted merged emotion names (calm merged into neutral)."""
    names = sorted(set(merge_emotion(e) for e in raw["emotion"]))
    label2id = {e: i for i, e in enumerate(names)}
    id2label = {i: e for e, i in label2id.items()}
    return label2id, id2label


def build_clip_table(raw, label2id):
    """One row per clip, indexed by the HF row index (`idx`).

    `clip_id` is the RAVDESS file id rebuilt from the metadata
    (03 = audio-only, 01 = speech), e.g. 03-01-06-01-02-01-12.
    """
    meta = raw.remove_columns(["audio"]).to_pandas()
    emo_ids = {v: k for k, v in RAVDESS_EMOTION_ID_TO_NAME.items()}
    int_ids, st_ids, rep_ids = _inverse(INTENSITY_MAP), _inverse(STATEMENT_MAP), _inverse(REPETITION_MAP)

    df = pd.DataFrame({"idx": np.arange(len(meta))})
    df["actor"] = meta["actor"].astype(int).values
    df["gender"] = np.where(df["actor"] % 2 == 1, "male", "female")
    df["emotion_raw"] = meta["emotion"].str.lower().str.strip().values
    df["emotion"] = [merge_emotion(e) for e in df["emotion_raw"]]
    df["label"] = [label2id[e] for e in df["emotion"]]
    df["intensity"] = meta["emotional_intensity"].str.lower().str.strip().values
    df["statement"] = meta["statement"].str.lower().str.strip().values
    df["repetition"] = meta["repetition"].str.lower().str.strip().values

    df["clip_id"] = [
        f"03-01-{emo_ids[r.emotion_raw]:02d}-{int_ids[r.intensity]:02d}-"
        f"{st_ids[r.statement]:02d}-{rep_ids[r.repetition]:02d}-{r.actor:02d}"
        for r in df.itertuples()
    ]
    if df["clip_id"].duplicated().any():
        dups = df.loc[df["clip_id"].duplicated(keep=False), "clip_id"].tolist()[:10]
        raise ValueError(f"Duplicate RAVDESS clip ids: {dups}")

    # RAVDESS: odd actor ids are male, even are female. Cross-check the dataset's own column.
    if "gender" in meta.columns:
        g = meta["gender"].astype(str).str.lower().str.strip()
        known = g.isin(["male", "female"])
        bad = known & (g != df["gender"])
        if bad.any():
            raise ValueError(f"{int(bad.sum())} clips have a gender column that contradicts actor parity")
    return df


# ── Audio ───────────────────────────────────────────────────────────────────
def load_waveforms(raw, target_sr, max_samples):
    """Resample to target_sr and truncate to max_samples. No padding (lengths vary)."""
    import torchaudio

    audio = raw.select_columns(["audio"])
    waves = []
    for i in tqdm(range(len(audio)), desc="Decoding audio"):
        item = audio[i]["audio"]
        arr = np.array(item["array"], dtype=np.float32)
        sr = item["sampling_rate"]
        if sr != target_sr:
            arr = torchaudio.functional.resample(torch.from_numpy(arr), sr, target_sr).numpy()
        waves.append(arr[:max_samples].astype(np.float32))
    return waves


def padded_inputs(waves, extractor, target_sr, max_samples, batch_size=64):
    """Pipeline-system preprocessing: zero-pad to max_samples, then the wav2vec2
    feature extractor normalises the whole padded array (padding included)."""
    out = np.zeros((len(waves), max_samples), dtype=np.float32)
    for s in range(0, len(waves), batch_size):
        chunk = [np.pad(w, (0, max_samples - len(w))) for w in waves[s:s + batch_size]]
        proc = extractor(chunk, sampling_rate=target_sr, return_tensors="np", padding=False)
        out[s:s + len(chunk)] = proc["input_values"]
    return out


def matched_normalize(wave):
    """Zero-mean unit-variance over the valid samples only (same formula as the
    wav2vec2 feature extractor, but no padding is ever added). Used for both
    backbones in the matched protocol."""
    wave = np.asarray(wave, dtype=np.float32)
    return ((wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)).astype(np.float32)


# ── Blendshapes ─────────────────────────────────────────────────────────────
def load_ravdess_blendshape_data(directory):
    """One sample per blendshape CSV, with its file name, modality and raw emotion."""
    samples = []
    filenames = sorted(f for f in os.listdir(directory) if f.endswith(".csv"))
    for filename in filenames:
        parts = filename.replace(".csv", "").split("-")
        if len(parts) != 7:
            continue
        try:
            modality, _, emo_id, intensity_id, statement_id, repetition_id, actor_id = map(int, parts)
        except ValueError:
            continue
        emotion_name = RAVDESS_EMOTION_ID_TO_NAME.get(emo_id, "unknown")
        if emotion_name == "unknown":
            continue

        df = pd.read_csv(os.path.join(directory, filename))
        df = df.drop(columns=["timestamp_ms"], errors="ignore")
        if "tongueOut" not in df.columns:
            df["tongueOut"] = 0.0
        df = df[ARKIT_52_COLS]
        features = df.values.astype(np.float32)
        assert features.shape[1] == 53, f"Column mismatch in {filename}"

        samples.append({
            "bs_filename": filename,
            "bs_modality": modality,
            "blendshape_features": features,
            "emotion": emotion_name,
            "intensity": INTENSITY_MAP.get(intensity_id, "unknown"),
            "statement": STATEMENT_MAP.get(statement_id, "unknown"),
            "repetition": REPETITION_MAP.get(repetition_id, "unknown"),
            "actor": int(actor_id),
        })
    print(f"Loaded {len(samples)} RAVDESS blendshape sequences from {directory}")
    return samples


def _make_key(emotion, intensity, statement, repetition, actor):
    return f"{emotion.lower()}|{intensity.lower()}|{statement.lower()}|{repetition.lower()}|{actor}"


def align_blendshapes(clips, bs_samples):
    """Join each audio clip to the blendshape sequence of its own recording.

    The key is (raw emotion, intensity, statement, repetition, actor). The raw
    emotion matters: neutral exists only at normal intensity, so a key built after
    merging calm into neutral would name both the neutral and the calm recording
    of the same statement and repetition. Duplicates that remain are the
    audio-video (01) and video-only (02) copies of the same recording; the first
    in sorted file order (01) is kept.
    """
    bs = pd.DataFrame(bs_samples).sort_values("bs_filename").reset_index(drop=True)
    bs["merge_key"] = [
        _make_key(str(r.emotion).lower(), r.intensity, r.statement, r.repetition, r.actor)
        for r in bs.itertuples()
    ]
    n_bs_files = len(bs)
    bs = bs.drop_duplicates(subset=["merge_key"], keep="first")

    keys = [
        _make_key(e, r.intensity, r.statement, r.repetition, r.actor)
        for e, r in zip(clips["emotion_raw"], clips.itertuples())
    ]
    a = clips[["idx", "label", "emotion", "actor"]].copy()
    a["merge_key"] = keys
    if a["merge_key"].duplicated().any():
        raise ValueError("Audio merge keys are not unique")

    bs_cols = ["merge_key", "bs_filename", "blendshape_features", "emotion", "actor"]
    merged = a.merge(bs[bs_cols], on="merge_key", how="inner", suffixes=("", "_bs"),
                     validate="many_to_one")
    merged["label_bs"] = [merge_emotion(e) for e in merged["emotion_bs"]]
    if not (merged["label_bs"] == merged["emotion"]).all():
        raise ValueError("Emotion mismatch between audio and blendshapes after alignment")
    if not (merged["actor_bs"] == merged["actor"]).all():
        raise ValueError("Actor mismatch between audio and blendshapes after alignment")

    shared = merged["bs_filename"].duplicated(keep=False)
    report = {
        "n_audio_clips": int(len(clips)),
        "n_blendshape_files": int(n_bs_files),
        "n_unique_blendshape_keys": int(len(bs)),
        "n_aligned": int(len(merged)),
        "n_audio_without_face": int(len(clips) - len(merged)),
        "n_clips_sharing_a_face_file": int(shared.sum()),
    }
    merged = merged.set_index("idx").sort_index()
    return merged[["bs_filename", "blendshape_features"]], report
