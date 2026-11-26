"""Train the main JointFuseNet architecture on prepared MATLAB splits."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from pytorch_msssim import ssim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset

from infer import ROOT, physical_volume, read_volumes
from model import JointFuseNet, RESISTIVITY_RANGE, VELOCITY_RANGE


class VolumePairs(Dataset):
    """Match data<ID>.mat to model<ID>.mat within one split."""

    def __init__(self, split):
        self.pairs = sorted(
            (path, split / "model" / f"model{path.name[4:]}")
            for path in (split / "data").glob("data*.mat")
        )
        if not self.pairs:
            raise ValueError(f"No MATLAB inputs found in {split / 'data'}")
        for _, target in self.pairs:
            if not target.is_file():
                raise FileNotFoundError(target)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        observation, target = self.pairs[index]
        return read_volumes(observation, "data")[0], read_volumes(target, "model")[0]


def joint_loss(outputs, target, stats):
    loss = 0.0
    for channel, (prediction, truth, limits) in enumerate(
        zip(outputs, (target[:, :1], target[:, 1:2, :, :30]), (VELOCITY_RANGE, RESISTIVITY_RANGE))
    ):
        lower, upper = limits
        unit_prediction = (prediction - lower) / (upper - lower)
        unit_truth = ((physical_volume(truth, stats, channel) - lower) / (upper - lower)).clamp(0, 1)
        loss += mse_loss(unit_prediction, unit_truth)
        loss += 0.1 * (1 - ssim(unit_prediction, unit_truth, data_range=1))
    return loss


def run_epoch(network, loader, device, stats, optimizer=None, scheduler=None):
    training = optimizer is not None
    network.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for observation, target in loader:
            observation, target = observation.to(device), target.to(device)
            outputs = network(observation[:, :1], observation[:, 1:2, :, :30])
            loss = joint_loss(outputs, target, stats)
            if not torch.isfinite(loss):
                raise ValueError("Encountered a non-finite training or validation loss.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--statistics", type=Path, default=ROOT / "examples/mean_std.json")
    parser.add_argument("--output", type=Path, default=ROOT / "training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) < 1:
        parser.error("Batch size and epoch count must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    stats = json.loads(args.statistics.read_text(encoding="utf-8"))["model"]
    training, validation = [
        DataLoader(
            VolumePairs(args.dataset / split),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            drop_last=True,
        )
        for split in ("train", "valid")
    ]
    if not len(training) or not len(validation):
        parser.error("Each split must contain at least one batch. Reduce --batch-size.")
    network = JointFuseNet().to(args.device)
    if torch.device(args.device) == torch.device("cuda:0") and torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(training)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1 - step / total_steps) ** 0.9)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(network, training, args.device, stats, optimizer, scheduler)
        print(f"Epoch {epoch}: training loss {train_loss:.6g}", flush=True)
        if epoch == 1 or epoch % 5 == 0:
            validation_loss = run_epoch(network, validation, args.device, stats)
            print(f"Epoch {epoch}: validation loss {validation_loss:.6g}", flush=True)
            state = {"epoch": epoch, "state_dict": network.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, args.output / "checkpoint.pth.tar")
            if validation_loss < best:
                best = validation_loss
                torch.save(state, args.output / "model_best.pth.tar")


if __name__ == "__main__":
    main()
