"""Seeding, data loaders, the training loop and prediction.

`train_model`: AdamW +
CosineAnnealingLR stepped per epoch, (class-weighted) cross-entropy, AMP in the
training pass only, validation in fp32, early stopping on validation loss and
restoring the best-validation weights.
"""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Reproducibility ─────────────────────────────────────────────────────────
def enable_determinism():
    """Call once, before CUDA is first used."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights_for(labels, num_classes):
    """Inverse-frequency class weights."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    return (len(labels) / (num_classes * counts)).astype(np.float32)


# ── Data ────────────────────────────────────────────────────────────────────
class IndexedDataset(Dataset):
    """Items are (xs[i], labels[i]) for i in `indices` (HF row indices)."""

    def __init__(self, xs, labels, indices):
        self.xs, self.labels, self.indices = xs, labels, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        return torch.as_tensor(np.asarray(self.xs[i]), dtype=torch.float32), int(self.labels[i])


def _collate_stack(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_loader(xs, labels, indices, batch_size, shuffle=False, seed=0):
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(
        IndexedDataset(xs, labels, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=0,
        collate_fn=_collate_stack,
    )


def _to(x, device):
    return x.to(device, non_blocking=True)


# ── Training ────────────────────────────────────────────────────────────────
def build_training_components(model, recipe, class_weights=None, device="cpu"):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"])
    weight = None
    if recipe.get("use_class_weights", True) and class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    return optimizer, scheduler, criterion


def _trainable_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _load_trainable_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n])


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
                epochs, patience, device, use_amp=True, name="model", verbose=True):
    use_amp = bool(use_amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    def run_epoch(loader, is_train):
        model.train(is_train)
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = _to(x, device), y.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(is_train):
                if is_train and use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = criterion(logits.float(), y)
                    if is_train:
                        loss.backward()
                        optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        return total_loss / n, correct / n

    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    epochs_no_improve, history = 0, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    t0 = time.time()
    epoch = 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, True)
        vl_loss, vl_acc = run_epoch(val_loader, False)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        for k, v in zip(history, (tr_loss, tr_acc, vl_loss, vl_acc, lr)):
            history[k].append(float(v))
        if verbose:
            print(f"  [{name}] epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | "
                  f"val {vl_loss:.4f}/{vl_acc:.3f} | lr {lr:.2e} | {time.time() - t0:.0f}s")
        if vl_loss < best_val_loss:
            best_val_loss, best_epoch, best_state = vl_loss, epoch, _trainable_state(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [{name}] early stop; best epoch {best_epoch}")
                break
    if best_state is not None:
        _load_trainable_state(model, best_state)
    return {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epoch),
        "history": history,
    }


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(_to(x, device)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)
