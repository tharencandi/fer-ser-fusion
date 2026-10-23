"""Speech models.

Pipeline systems (each with its own head and fixed recipe; the backbone loading is
factored out so tests can swap in a tiny random model, and the hidden states are
recorded by `_HiddenStateRecorder` so they do not depend on the transformers version):
    Wav2Vec2LinearProbe   W2V2-WLP: frozen wav2vec2 + 13-layer weighted sum + frame MLP + mean pool
    Wav2Vec2LoRAProbe     W2V2-LoRA: PEFT LoRA (q_proj, v_proj, r=4) + weighted sum + gate + LN + linear
    Emotion2VecLastLayerHead
                          E2V-LP: head applied to the cached mean-pooled final-layer
                          output (see `emotion2vec_last_layer_features`)

Matched protocol (one probe class for every backbone):
    Wav2Vec2States / Emotion2VecStates
        return every hidden state the backbone exposes, shape (S, frames, D),
        for one clip at its true length (no padding ever enters the backbone,
        so pooling is over valid frames only)
    SharedProbe
        softmax-weighted sum of the S mean-pooled states -> Linear(D,128) ->
        ReLU -> Dropout -> Linear(128,C)
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

TINY = "tiny-random"


# ── Backbone loading ────────────────────────────────────────────────────────
def wav2vec2_config(model_name):
    from transformers import Wav2Vec2Config
    if model_name == TINY:
        from .testing import tiny_wav2vec2_config
        return tiny_wav2vec2_config()
    return Wav2Vec2Config.from_pretrained(model_name)


def wav2vec2_backbone(model_name, config):
    from transformers import Wav2Vec2Model
    if model_name == TINY:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            return Wav2Vec2Model(config)
    return Wav2Vec2Model.from_pretrained(model_name, config=config)


def load_emotion2vec(model_id, device="cpu", hub="ms"):
    """The bare emotion2vec nn.Module (funasr's AutoModel builds it and calls .eval())."""
    if model_id == TINY:
        from .testing import tiny_emotion2vec
        return tiny_emotion2vec().to(device).eval()
    from funasr import AutoModel
    am = AutoModel(model=model_id, hub=hub, device=str(device), disable_update=True)
    return am.model



class _RecordedLayer(nn.Module):
    """Wraps one wav2vec2 encoder layer. Applies LayerDrop with the semantics of
    transformers 4.x (a skipped layer passes its input through, so its hidden
    state repeats the previous one) and records the layer output.

    transformers 5.x leaves skipped layers out of `hidden_states` (fewer than 13
    tensors in training mode) and returns `hidden_states=None` through PEFT, which
    breaks the weighted-layer probes. Recording the states here gives the same 13
    states under any transformers version; the backbone's own LayerDrop is set to 0.
    """

    def __init__(self, layer, layerdrop, sink):
        super().__init__()
        self.layer = layer
        self.layerdrop = float(layerdrop)
        self._sink = sink

    def forward(self, hidden_states, *args, **kwargs):
        out = self.layer(hidden_states, *args, **kwargs)
        if self.training and torch.rand([]).item() < self.layerdrop:
            out = (hidden_states,) + tuple(out[1:]) if isinstance(out, tuple) else hidden_states
        self._sink.append(out[0] if isinstance(out, tuple) else out)
        return out


class _HiddenStateRecorder:
    """Installs _RecordedLayer around every encoder layer of a Wav2Vec2Model and
    returns all num_layers + 1 hidden states of the last forward pass."""

    def __init__(self, wav2vec2_model):
        self.states = []
        enc = wav2vec2_model.encoder
        p = wav2vec2_model.config.layerdrop
        wav2vec2_model.config.layerdrop = 0.0
        enc.layers = nn.ModuleList(_RecordedLayer(l, p, self.states) for l in enc.layers)
        enc.layers[0].register_forward_pre_hook(self._pre_hook)
        self.n_states = len(enc.layers) + 1

    def _pre_hook(self, module, args):
        self.states.clear()
        self.states.append(args[0])

    def collect(self):
        states = list(self.states)
        self.states.clear()
        if len(states) != self.n_states:
            raise RuntimeError(f"recorded {len(states)} hidden states, expected {self.n_states}")
        return states


# ── Pipeline systems ────────────────────────────────────────────────────────
class Wav2Vec2LinearProbe(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout_prob: float = 0.2):
        super().__init__()
        config = wav2vec2_config(model_name)
        config.output_hidden_states = True
        config.mask_time_prob = 0.05
        config.mask_time_length = 15
        config.mask_feature_prob = 0.05
        config.mask_feature_length = 12
        self.backbone = wav2vec2_backbone(model_name, config)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self._recorder = _HiddenStateRecorder(self.backbone)

        hidden_size = self.backbone.config.hidden_size
        num_total_states = self.backbone.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.ones(num_total_states))
        self.frame_transformation = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        self.backbone(input_values)
        hidden_states = torch.stack(self._recorder.collect(), dim=1)
        w = torch.softmax(self.layer_weights, dim=0)
        weighted_sum = (hidden_states * w[None, :, None, None]).sum(dim=1)
        frame_features = self.frame_transformation(weighted_sum)
        pooled = frame_features.mean(dim=1)
        return self.classifier(pooled)


class Wav2Vec2LoRAProbe(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        config = wav2vec2_config(model_name)
        config.output_hidden_states = True
        config.mask_time_prob = 0.05
        config.mask_time_length = 15
        config.mask_feature_prob = 0.05
        config.mask_feature_length = 12
        base_model = wav2vec2_backbone(model_name, config)
        peft_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.25,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(base_model, peft_config)
        self._recorder = _HiddenStateRecorder(self.backbone.get_base_model())
        hidden_size = self.backbone.config.hidden_size
        num_total_states = self.backbone.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.ones(num_total_states))
        self.weight_gate = nn.Parameter(torch.ones(hidden_size))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        self.backbone(input_values=input_values)
        hidden_states = torch.stack(self._recorder.collect(), dim=1)
        gated_hidden_states = torch.sigmoid(self.weight_gate) * hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        pooled = (gated_hidden_states * w[None, :, None, None]).sum(dim=1).mean(dim=1)
        return self.classifier(pooled)


class Emotion2VecLastLayerHead(nn.Module):
    """Head of the E2V-LP system. The frozen backbone (eval mode, no padding mask)
    is averaged over all frames of the 5.27 s padded input; that pooled vector is
    deterministic, so it is computed once by `emotion2vec_last_layer_features`
    and this head is trained on it."""

    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.25):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes),
        )

    def forward(self, pooled_x):
        return self.head(pooled_x)


@torch.no_grad()
def emotion2vec_last_layer_features(e2v_model, inputs, device, batch_size=16):
    """Mean of res['x'] over all frames of the padded, extractor-normalised input."""
    e2v_model.eval()
    out = []
    for s in tqdm(range(0, len(inputs), batch_size), desc="emotion2vec final-layer features"):
        x = torch.as_tensor(inputs[s:s + batch_size], dtype=torch.float32, device=device)
        res = e2v_model.extract_features(x, padding_mask=None)
        out.append(res["x"].mean(dim=1).float().cpu().numpy())
    return np.concatenate(out)


# ── Matched protocol ────────────────────────────────────────────────────────
class _AlwaysEval(nn.Module):
    """Backbone wrapper that stays in eval mode (no internal dropout, LayerDrop or
    SpecAugment) for both backbones."""

    def train(self, mode=True):
        return super().train(False)


class Wav2Vec2States(_AlwaysEval):
    """All hidden states of wav2vec2: CNN/positional output + each transformer layer."""

    def __init__(self, model_name):
        super().__init__()
        config = wav2vec2_config(model_name)
        config.output_hidden_states = True
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        self.model = wav2vec2_backbone(model_name, config)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = config.hidden_size
        self.num_states = config.num_hidden_layers + 1
        self.eval()

    def forward(self, wave):  # (1, T)
        hs = self.model(wave, output_hidden_states=True).hidden_states  # eval mode: no LayerDrop
        if len(hs) != self.num_states:
            raise RuntimeError(f"got {len(hs)} hidden states, expected {self.num_states}")
        return torch.stack([h.float() for h in hs], dim=0)[:, 0]  # (S, F, D); fp32 (AMP may mix dtypes)


class Emotion2VecStates(_AlwaysEval):
    """All hidden states of emotion2vec (data2vec 2.0 layout).

    The audio encoder runs CNN -> projection -> relative positional conv ->
    context encoder (LayerNorm + `prenet_depth` AltBlocks), then the model runs
    `depth` more AltBlocks; emotion2vec_base has 4 + 8 = 12 transformer blocks
    and post-LN (`layer_norm_first=False`, so no final norm). State 0 is the input
    to the first context-encoder block (the counterpart of wav2vec2's
    hidden_states[0]); states 1..12 are the outputs of the 12 blocks in order.
    They are collected with forward hooks; extract_features is unchanged.
    """

    def __init__(self, e2v_model):
        super().__init__()
        self.model = e2v_model
        for p in self.model.parameters():
            p.requires_grad_(False)
        enc = self.model.modality_encoders["AUDIO"]
        self.blocks = list(enc.context_encoder.blocks) + list(self.model.blocks)
        self.n_extra = int(enc.modality_cfg.num_extra_tokens)
        self.hidden_size = int(self.model.cfg.get("embed_dim"))
        self.num_states = len(self.blocks) + 1
        self.final_norm = self.model.norm is not None
        self._buf = None
        self.blocks[0].register_forward_pre_hook(self._pre_hook)
        for blk in self.blocks:
            blk.register_forward_hook(self._hook)
        self.eval()

    def _pre_hook(self, module, args):
        if self._buf is not None:
            self._buf.append(args[0])

    def _hook(self, module, args, output):
        if self._buf is not None:
            self._buf.append(output[0])

    def forward(self, wave):  # (1, T)
        self._buf = []
        try:
            res = self.model.extract_features(wave, padding_mask=None)
            states = self._buf
        finally:
            self._buf = None
        if len(states) != self.num_states:
            raise RuntimeError(f"captured {len(states)} states, expected {self.num_states}")
        if not self.final_norm and self.n_extra == 0:
            # the last block output is exactly what extract_features returns
            assert states[-1].shape == res["x"].shape
        return torch.stack([s[:, self.n_extra:].float() for s in states], dim=0)[:, 0]  # (S, F, D), fp32


class SharedProbe(nn.Module):
    """Same head for every backbone. Softmax-weighted sum over all hidden states,
    masked mean pooling (pooling is done per clip over its own frames before the
    weighted sum; both are linear, so the order does not matter), then an MLP."""

    def __init__(self, num_states, hidden_size, num_classes, dropout, proj_dim=128):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_states))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, pooled_states):  # (B, S, D)
        w = torch.softmax(self.layer_weights, dim=0)
        x = (pooled_states * w[None, :, None].to(pooled_states.dtype)).sum(dim=1)
        return self.head(x)


@torch.no_grad()
def compute_pooled_states(states, waves, device, desc="pooled states"):
    """Frozen-backbone cache: (N, S, D) float32, one clip at a time, fp32."""
    states.eval()
    out = np.zeros((len(waves), states.num_states, states.hidden_size), dtype=np.float32)
    for i, w in enumerate(tqdm(waves, desc=desc)):
        x = torch.as_tensor(w, dtype=torch.float32, device=device).unsqueeze(0)
        out[i] = states(x).mean(dim=1).float().cpu().numpy()
    return out


# ── Parameter accounting ────────────────────────────────────────────────────
def param_breakdown(model):
    """Trainable parameters by component, plus the frozen total."""
    groups = {"layer_weights": 0, "weight_gate": 0, "lora": 0, "head": 0}
    frozen = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen += p.numel()
            continue
        if "lora_" in name:
            groups["lora"] += p.numel()
        elif "layer_weights" in name:
            groups["layer_weights"] += p.numel()
        elif "weight_gate" in name:
            groups["weight_gate"] += p.numel()
        else:
            groups["head"] += p.numel()
    groups["trainable_total"] = sum(groups.values())
    groups["frozen"] = frozen
    return groups
