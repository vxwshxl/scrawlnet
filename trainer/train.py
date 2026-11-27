"""Train scrawlnet's CNN on MNIST and export it for the browser.

Run:  uv run train.py            (≈1–3 min on Apple Silicon)
Out:  ../public/model/weights.bin   raw float32, little-endian, layers in order
      ../public/model/model.json    layer shapes + byte offsets + training stats
      ../public/model/samples.json  20 test digits with PyTorch's logits: the site's
                                    "try a real digit" button, and the fixture that
                                    scripts/check-parity.ts checks the TS forward pass against
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "public" / "model"
MEAN, STD = 0.1307, 0.3081  # MNIST pixel mean/std; the browser normalises the same way
EPOCHS = 8
SEED = 7


class ScrawlNet(nn.Module):
    """28×28 → conv(16) → pool → conv(32) → pool → dense(64) → 10 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 16 × 28 × 28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32 × 14 × 14
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # → 16 × 14 × 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # → 32 × 7 × 7
        x = torch.flatten(x, 1)                     # → 1568
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    confusion = np.zeros((10, 10), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            np.add.at(confusion, (y.cpu().numpy(), pred.cpu().numpy()), 1)
    return loss_sum / total, correct / total, confusion


def main() -> None:
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    norm = transforms.Normalize((MEAN,), (STD,))
    # Augmentation: people draw digits rotated, off-centre and at odd sizes,
    # so the training images are randomly nudged the same way.
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=8),
        transforms.ToTensor(),
        norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    data_dir = ROOT / "data"
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = ScrawlNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader))

    history = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = run_correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * y.numel()
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
        test_loss, test_acc, confusion = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "trainLoss": round(run_loss / seen, 4),
            "trainAcc": round(run_correct / seen, 4),
            "testLoss": round(test_loss, 4),
            "testAcc": round(test_acc, 4),
        }
        history.append(row)
        print(row, f"{time.time() - t0:.0f}s")

    export(model.cpu().eval(), history, confusion, test_set)


def export(model: ScrawlNet, history, confusion, test_set) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layers, blobs, offset = [], [], 0
    for name in ["conv1", "conv2", "fc1", "fc2"]:
        layer = getattr(model, name)
        entry = {"name": name}
        for kind in ["weight", "bias"]:
            arr = getattr(layer, kind).detach().numpy().astype("<f4")
            entry[kind] = {"shape": list(arr.shape), "offset": offset, "length": int(arr.size)}
            blobs.append(arr.tobytes())
            offset += arr.nbytes
        layers.append(entry)
    (OUT / "weights.bin").write_bytes(b"".join(blobs))

    params = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "scrawlnet",
        "input": [1, 28, 28],
        "normalize": {"mean": MEAN, "std": STD},
        "params": params,
        "bytes": offset,
        "layers": layers,
        "history": history,
        "testAcc": history[-1]["testAcc"],
        "confusion": confusion.tolist(),
    }
    (OUT / "model.json").write_text(json.dumps(meta, indent=1))

    # Parity fixtures: first 20 test digits (already normalised) + PyTorch logits.
    samples = []
    with torch.no_grad():
        for i in range(20):
            x, y = test_set[i]
            logits = model(x.unsqueeze(0))[0]
            samples.append({
                "label": int(y),
                "input": [round(v, 6) for v in x.flatten().tolist()],
                "logits": [round(v, 6) for v in logits.tolist()],
            })
    (OUT / "samples.json").write_text(json.dumps(samples))
    print(f"exported {params:,} params ({offset / 1024:.0f} KiB) → {OUT}")


if __name__ == "__main__":
    main()
