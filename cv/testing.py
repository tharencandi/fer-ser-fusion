"""Offline stand-ins for smoke tests (no Hugging Face / ModelScope / Drive access).

* `synthetic_ravdess()`  : a Dataset with the xbgoose/ravdess columns, 24 actors x 60
                           clips with the real RAVDESS design (calm/neutral, intensities,
                           statements, repetitions), short synthetic waveforms at 22.05 kHz
* `tiny_wav2vec2_config()`: a 2-layer, 32-dim wav2vec2 (random weights, HF classes)
* `tiny_emotion2vec()`   : a funasr Emotion2vec with 1 prenet + 2 blocks, 32-dim (random weights);
                           full_size=True gives the emotion2vec_base layout (4 + 8 blocks, 768-dim)
* `write_synthetic_blendshapes(clips, directory)`: CSVs named like the real files,
                           in both 01 (AV) and 02 (video-only) versions

These exercise every code path of the real pipeline; the numbers they produce mean nothing.
"""
import os

import numpy as np
import pandas as pd

from .data import ARKIT_52_COLS, INTENSITY_MAP, REPETITION_MAP, STATEMENT_MAP


def synthetic_ravdess(sr=22050, seed=0):
    from datasets import Dataset

    rng = np.random.default_rng(seed)
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    rows = {k: [] for k in ["audio", "modality", "vocal_channel", "emotion", "emotional_intensity",
                            "statement", "repetition", "actor", "gender"]}
    for actor in range(1, 25):
        for e_i, emo in enumerate(emotions):
            intensities = [1] if emo == "neutral" else [1, 2]
            for it in intensities:
                for st in (1, 2):
                    for rep in (1, 2):
                        dur = rng.uniform(0.8, 1.6)
                        t = np.arange(int(dur * sr)) / sr
                        f0 = 120 + 25 * e_i + 5 * (actor % 4)
                        wave = 0.3 * it * np.sin(2 * np.pi * f0 * t) + 0.05 * rng.standard_normal(len(t))
                        rows["audio"].append({"array": wave.astype(np.float32), "sampling_rate": sr})
                        rows["modality"].append("audio-only")
                        rows["vocal_channel"].append("speech")
                        rows["emotion"].append(emo)
                        rows["emotional_intensity"].append(INTENSITY_MAP[it])
                        rows["statement"].append(STATEMENT_MAP[st])
                        rows["repetition"].append(REPETITION_MAP[rep])
                        rows["actor"].append(str(actor))
                        rows["gender"].append("male" if actor % 2 else "female")
    return Dataset.from_dict(rows)


def tiny_wav2vec2_config():
    from transformers import Wav2Vec2Config
    return Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        conv_dim=(32,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        feat_extract_norm="group", do_stable_layer_norm=False, vocab_size=32,
    )


def tiny_emotion2vec(full_size=False):
    """Random-weight Emotion2vec. full_size=True uses the data2vec 2.0 base layout of
    emotion2vec_base (768-dim, 12 heads, 4 context-encoder blocks + 8 blocks)."""
    import torch
    from funasr.models.emotion2vec.model import Emotion2vec

    audio = dict(
        type="AUDIO", prenet_depth=1, prenet_layerdrop=0.0, prenet_dropout=0.0,
        start_drop_path_rate=0.0, end_drop_path_rate=0.0, num_extra_tokens=0,
        init_extra_token_zero=True, mask_noise_std=0.01, mask_prob_min=None, mask_prob=0.7,
        inverse_mask=False, mask_prob_adjust=0.0, keep_masked_pct=0.0, mask_length=5,
        add_masks=False, remove_masks=False, mask_dropout=0.0, encoder_zero_mask=True,
        mask_channel_prob=0.0, mask_channel_length=64, ema_local_encoder=False,
        local_grad_mult=1.0, use_alibi_encoder=True, alibi_scale=1.0, learned_alibi=False,
        alibi_max_pos=None, learned_alibi_scale=True, learned_alibi_scale_per_head=True,
        learned_alibi_scale_per_layer=False, num_alibi_heads=2, model_depth=2, decoder=None,
        extractor_mode="layer_norm",
        feature_encoder_spec="[(32, 10, 5)] + [(32, 3, 2)] * 4 + [(32,2,2)] + [(32,2,2)]",
        conv_pos_width=95, conv_pos_groups=2, conv_pos_depth=5, conv_pos_pre_ln=False,
    )
    model_conf = dict(
        loss_beta=0.0, loss_scale=None, depth=2, start_drop_path_rate=0.0, end_drop_path_rate=0.0,
        num_heads=2, norm_eps=1e-6, norm_affine=True, encoder_dropout=0.1, post_mlp_drop=0.1,
        attention_dropout=0.1, activation_dropout=0.0, dropout_input=0.0, layerdrop=0.0,
        embed_dim=32, mlp_ratio=4.0, layer_norm_first=False, average_top_k_layers=2,
        end_of_block_targets=False, clone_batch=1, normalize=True,
        modalities=dict(audio=audio),
    )
    if full_size:
        audio.update(prenet_depth=4, model_depth=8, num_alibi_heads=12, conv_pos_groups=16,
                     feature_encoder_spec="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]")
        model_conf.update(depth=8, num_heads=12, embed_dim=768, average_top_k_layers=8)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Emotion2vec(model_conf=model_conf)


def write_synthetic_blendshapes(clips, directory, seed=0):
    """One CSV per clip and per video modality (01 AV, 02 video-only), MediaPipe-like
    column order with a timestamp column and no tongueOut column."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    cols = [c for c in ARKIT_52_COLS if c != "tongueOut"]
    cols = ["timestamp_ms"] + list(rng.permutation(cols))
    for r in clips.itertuples():
        n = int(rng.integers(30, 60))
        base = rng.uniform(0, 0.3, size=(n, len(cols) - 1))
        base[:, (r.label * 5) % (len(cols) - 1)] += 0.5  # weak class signal
        df = pd.DataFrame(np.column_stack([np.arange(n) * 33.3, base]), columns=cols)
        tail = r.clip_id[2:]  # "-01-EE-II-SS-RR-AA"
        for modality in ("01", "02"):
            df.to_csv(os.path.join(directory, f"{modality}{tail}.csv"), index=False)
