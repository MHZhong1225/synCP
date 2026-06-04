from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.progress import progress


@dataclass(frozen=True)
class SyntheticDataset:
    x: np.ndarray
    y: np.ndarray
    known_group: np.ndarray
    hidden_groups: np.ndarray


@dataclass(frozen=True)
class SplitData:
    train: np.ndarray
    calib0: np.ndarray
    discovery: np.ndarray
    mvcal: np.ndarray
    test: np.ndarray


def make_synthetic_medical_data(
    n: int = 12000,
    dim: int = 32,
    num_classes: int = 4,
    seed: int = 7,
) -> SyntheticDataset:
    """Create a medical-like classification problem with hidden hard slices.

    The hidden slices are feature-space regions with class-dependent score
    ambiguity. Known groups are correlated with, but do not fully reveal, the
    hidden slices.
    """
    rng = np.random.default_rng(seed)
    class_means = rng.normal(0.0, 2.2, size=(num_classes, dim))
    y = rng.integers(0, num_classes, size=n)
    x = class_means[y] + rng.normal(0.0, 1.0, size=(n, dim))

    hidden_groups = np.zeros((n, 3), dtype=bool)
    group_specs = [
        (0, 1, 0, 0.28, 1.85),
        (1, 2, 1, 0.22, 1.65),
        (2, 3, 2, 0.18, 1.45),
    ]
    for group_id, source_class, target_class, prob, strength in group_specs:
        idx = np.where(y == source_class)[0]
        selected = idx[rng.random(len(idx)) < prob]
        hidden_groups[selected, group_id] = True
        direction = class_means[target_class] - class_means[source_class]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        x[selected] += strength * direction + rng.normal(0.0, 0.55, size=(len(selected), dim))

    known_group = np.zeros(n, dtype=np.int64)
    known_group[hidden_groups[:, 0] | (rng.random(n) < 0.18)] = 1
    known_group[hidden_groups[:, 1] | (rng.random(n) < 0.12)] = 2
    known_group[(known_group == 0) & (rng.random(n) < 0.15)] = 3

    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    return SyntheticDataset(x=x.astype(np.float32), y=y.astype(np.int64), known_group=known_group, hidden_groups=hidden_groups)


def make_splits(n: int, seed: int = 7) -> SplitData:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cuts = np.cumsum([0.50, 0.15, 0.15, 0.10])
    a, b, c, d = (cuts * n).astype(int)
    return SplitData(
        train=idx[:a],
        calib0=idx[a:b],
        discovery=idx[b:c],
        mvcal=idx[c:d],
        test=idx[d:],
    )


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x


def train_classifier(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    num_classes: int,
    device: str = "cuda",
    seed: int = 7,
    epochs: int = 45,
    batch_size: int = 256,
) -> MLPClassifier:
    torch.manual_seed(seed)
    model = MLPClassifier(x.shape[1], num_classes).to(device)
    dataset = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in progress(range(epochs), desc="train synthetic", leave=False):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_and_embed(
    model: MLPClassifier,
    x: np.ndarray,
    device: str = "cuda",
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    embeddings = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
    for (xb,) in progress(loader, desc="embed synthetic", total=len(loader), leave=False):
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        embeddings.append(model.embed(xb).cpu().numpy())
    return np.concatenate(probs), np.concatenate(embeddings).astype(np.float32)
