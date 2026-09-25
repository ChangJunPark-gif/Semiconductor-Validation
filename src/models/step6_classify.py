"""Step 6: closed-set defect-pattern CNN and spatial-feature baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.step1_yield import load_legacy_pickle
from src.data.step2_visualize import CMAP, NORM
from src.features.step4_features import PCA_FEATURES


LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
INPUT_FEATURES = ["fail_rate", *PCA_FEATURES]
SIZE = 48


def map_digest(wafer_map: np.ndarray) -> str:
    array = np.asarray(wafer_map, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(np.asarray(array.shape, dtype=np.int32).tobytes())
    h.update(array.tobytes())
    return h.hexdigest()


def resize_map(wafer_map: np.ndarray, size: int = SIZE) -> np.ndarray:
    array = np.asarray(wafer_map, dtype=np.uint8)
    if array.ndim != 2 or not np.isin(array, [0, 1, 2]).all():
        raise ValueError("Expected 2D wafer map with states 0, 1, 2")
    height, width = array.shape
    side = max(height, width)
    padded = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - height) // 2, (side - width) // 2
    padded[y0:y0 + height, x0:x0 + width] = array
    return np.asarray(Image.fromarray(padded).resize((size, size), Image.Resampling.NEAREST))


def cross_split_keep(hashes: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Keep earlier split on exact-hash collision; retain same-split repetitions."""
    keep = np.ones(len(hashes), dtype=bool)
    removed = {"validation": 0, "test": 0}
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        idx = np.flatnonzero(splits == split)
        for i in idx:
            if split != "train" and hashes[i] in seen:
                keep[i] = False
                removed[split] += 1
        seen.update(hashes[idx])
    return keep, removed


class SmallCNN(nn.Module):
    def __init__(self, classes: int = len(LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def channels(maps: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack((maps == 1, maps == 2), axis=1).astype(np.float32))


def predict_cnn(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    outputs = []
    start = time.perf_counter()
    with torch.inference_mode():
        for (xb,) in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    elapsed = time.perf_counter() - start
    return np.concatenate(outputs), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/wafer_features.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument("--data-output", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-output", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.data_output.mkdir(parents=True, exist_ok=True)
    args.report_output.mkdir(parents=True, exist_ok=True)
    figures = args.report_output / "figures"
    figures.mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = pd.read_csv(args.features, usecols=["source_row", "lot_id", "failure_type", "split", *INPUT_FEATURES])
    frame = frame[frame.failure_type.isin(LABELS)].copy().reset_index(drop=True)
    cache = args.data_output / "step6_labeled_maps.npz"
    if cache.exists():
        stored = np.load(cache)
        maps = stored["maps"]
        hashes = stored["hashes"]
        if not np.array_equal(stored["source_rows"], frame.source_row.to_numpy()):
            raise ValueError("Map cache does not match feature rows")
    else:
        raw = load_legacy_pickle(args.raw)
        maps = np.empty((len(frame), SIZE, SIZE), dtype=np.uint8)
        hashes = np.empty(len(frame), dtype="U64")
        for i, source_row in enumerate(frame.source_row):
            wafer_map = np.asarray(raw.iloc[int(source_row)].waferMap)
            maps[i] = resize_map(wafer_map)
            hashes[i] = map_digest(wafer_map)
        np.savez(cache, maps=maps, hashes=hashes, source_rows=frame.source_row.to_numpy())
        del raw
    # The CNN sees resized maps: distinct originals can become identical inputs.
    original_keep, _ = cross_split_keep(hashes, frame.split.to_numpy())
    input_hashes = np.array([hashlib.sha256(wafer_map.tobytes()).hexdigest() for wafer_map in maps])
    input_keep, _ = cross_split_keep(input_hashes, frame.split.to_numpy())
    keep = original_keep & input_keep
    removed = {name: int(((frame.split == name) & ~keep).sum()) for name in ("validation", "test")}
    frame = frame.loc[keep].reset_index(drop=True)
    maps = maps[keep]
    hashes = hashes[keep]
    labels = frame.failure_type.map({name: i for i, name in enumerate(LABELS)}).to_numpy(dtype=np.int64)
    split = frame.split.to_numpy()
    tr, va, te = (np.flatnonzero(split == name) for name in ("train", "validation", "test"))
    if any(len(indices) == 0 for indices in (tr, va, te)):
        raise ValueError("An evaluation split is empty")
    if set(frame.lot_id.iloc[tr]) & set(frame.lot_id.iloc[va]) or set(frame.lot_id.iloc[tr]) & set(frame.lot_id.iloc[te]) or set(frame.lot_id.iloc[va]) & set(frame.lot_id.iloc[te]):
        raise AssertionError("Lot leakage between splits")

    baseline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
    baseline.fit(frame.loc[tr, INPUT_FEATURES], labels[tr])
    joblib.dump(baseline, args.data_output / "step6_logistic.joblib")
    start = time.perf_counter()
    baseline_pred = baseline.predict(frame.loc[te, INPUT_FEATURES])
    baseline_seconds = time.perf_counter() - start

    x = channels(maps)
    dataset = TensorDataset(x)
    def loader(indices: np.ndarray, shuffle: bool = False) -> DataLoader:
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_counts = np.bincount(labels[tr], minlength=len(LABELS))
    weights = np.sqrt(len(tr) / np.maximum(class_counts, 1))
    weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    train_dataset = TensorDataset(x[tr], torch.from_numpy(labels[tr]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    history = []
    best_f1 = -1.0
    best_epoch = 0
    checkpoint = args.data_output / "step6_cnn.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        val_logits, _ = predict_cnn(model, loader(va), device)
        val_f1 = f1_score(labels[va], val_logits.argmax(axis=1), labels=np.arange(len(LABELS)), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total_loss / len(tr), "validation_macro_f1": val_f1})
        print(f"epoch {epoch}: loss={total_loss / len(tr):.4f}, val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.state_dict(), "labels": LABELS, "size": SIZE, "epoch": epoch}, checkpoint)
    pd.DataFrame(history).to_csv(args.report_output / "step6_training_history.csv", index=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    logits, cnn_seconds = predict_cnn(model, loader(te), device)
    cnn_pred = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pd.DataFrame({"source_row": frame.source_row.iloc[te].to_numpy(), "true_label": [LABELS[i] for i in labels[te]],
                  "cnn_prediction": [LABELS[i] for i in cnn_pred], "cnn_confidence": probabilities.max(axis=1),
                  "baseline_prediction": [LABELS[i] for i in baseline_pred]}).to_csv(args.report_output / "step6_test_predictions.csv", index=False)
    metrics = {}
    for name, pred, seconds in (("CNN", cnn_pred, cnn_seconds), ("logistic", baseline_pred, baseline_seconds)):
        result = classification_report(labels[te], pred, labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0)
        metrics[name] = {"macro_f1": result["macro avg"]["f1-score"], "accuracy": result["accuracy"],
                         "milliseconds_per_wafer": seconds * 1000 / len(te), "per_class": result}
        ConfusionMatrixDisplay.from_predictions(labels[te], pred, labels=np.arange(len(LABELS)), display_labels=LABELS,
                                                xticks_rotation=45, cmap="Blues", colorbar=False)
        plt.title(f"{name}: test confusion matrix")
        plt.tight_layout()
        plt.savefig(figures / f"step6_{name.lower()}_confusion.png", dpi=150)
        plt.close()
    (args.report_output / "step6_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    errors = np.flatnonzero(cnn_pred != labels[te])
    ranked = errors[np.argsort(-probabilities[errors].max(axis=1))[:12]]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, position in zip(axes.flat, ranked):
        index = te[position]
        ax.imshow(maps[index], cmap=CMAP, norm=NORM, interpolation="nearest")
        ax.set_title(f"{LABELS[labels[index]]} → {LABELS[cnn_pred[position]]}\nconf {probabilities[position].max():.2f}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(ranked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "step6_error_gallery.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"split_counts": {name: int((split == name).sum()) for name in ("train", "validation", "test")},
                      "duplicate_removed": removed, "best_epoch": best_epoch,
                      "validation_macro_f1": best_f1, "test_macro_f1": metrics["CNN"]["macro_f1"],
                      "baseline_macro_f1": metrics["logistic"]["macro_f1"], "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
