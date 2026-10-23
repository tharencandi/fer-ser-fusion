"""Temperature scaling and calibrated sum fusion.

Temperatures are fitted on the validation actors of the same fold and seed, then
applied to that fold's test logits. Runs on CPU in float64 so the fit is deterministic.

T = exp(log_t) is always positive, and L-BFGS uses a strong-Wolfe line search, so
the fit reaches the validation cross-entropy optimum (an unconstrained fit of T
with a fixed step can diverge to T < 0, which inverts the probabilities).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

METHOD = "T = exp(log_t), L-BFGS with strong-Wolfe line search, float64, validation cross-entropy"


class TemperatureScaler(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.log_t = nn.Parameter(torch.zeros(1, dtype=torch.float64, device=device))

    @property
    def temperature(self):
        return self.log_t.exp()

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        logits_t = torch.tensor(np.asarray(logits), dtype=torch.float64, device=self.device)
        labels_t = torch.tensor(np.asarray(labels), dtype=torch.long, device=self.device)
        optimizer = optim.LBFGS([self.log_t], lr=1.0, max_iter=500, line_search_fn="strong_wolfe")
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self

    def calibrate(self, logits):
        with torch.no_grad():
            return self.forward(torch.tensor(np.asarray(logits), dtype=torch.float64, device=self.device)).cpu().numpy()


def calibrated_sum_fusion(audio_val, visual_val, val_labels, audio_test, visual_test, weights=(1.0, 1.0)):
    """Returns fused probabilities on test and the two fitted temperatures."""
    ts_a = TemperatureScaler().fit(audio_val, val_labels)
    ts_v = TemperatureScaler().fit(visual_val, val_labels)
    pa = torch.softmax(torch.tensor(ts_a.calibrate(audio_test)), dim=-1)
    pv = torch.softmax(torch.tensor(ts_v.calibrate(visual_test)), dim=-1)
    fused = weights[0] * pa + weights[1] * pv
    return fused.numpy().astype(np.float32), float(ts_a.temperature.item()), float(ts_v.temperature.item())
