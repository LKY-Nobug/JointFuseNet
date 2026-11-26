"""Run JointFuseNet on a normalized MATLAB sample and save physical volumes."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import torch

from model import JointFuseNet


ROOT = Path(__file__).resolve().parent
VOLUME_SHAPE = (2, 30, 30, 120)


def read_volumes(path, key):
    """Read the stored channel-first volumes and put tunnel distance on axis H."""
    volume = loadmat(path)[key]
    if volume.shape != VOLUME_SHAPE or not np.isfinite(volume).all():
        raise ValueError(f"{path}: expected finite {key} with shape {VOLUME_SHAPE}")
    return torch.from_numpy(volume.astype(np.float32)).permute(0, 1, 3, 2).unsqueeze(0)


def load_network(path):
    # The original checkpoint stores its validation loss as a NumPy float64 scalar.
    numpy_scalar_types = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(numpy_scalar_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model = JointFuseNet()
    model.load_state_dict(state, strict=True)
    return model.eval()


def physical_volume(values, statistics, channel):
    return values * statistics[f"std{channel}"] + statistics[f"mean{channel}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/data0.mat")
    parser.add_argument("--target", type=Path, help="Optional normalized model MAT file")
    parser.add_argument("--statistics", type=Path, default=ROOT / "examples/mean_std.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/jointfusenet-se-maxpool.pt")
    mode.add_argument("--smoke-test", action="store_true", help="Run with deterministic fresh weights")
    parser.add_argument("--output", type=Path, default=ROOT / "results/predictions.mat")
    parser.add_argument("--device", default="cpu", help="PyTorch device, such as cpu or cuda:0")
    args = parser.parse_args()

    observe = read_volumes(args.input, "data").to(args.device)
    if args.smoke_test:
        torch.manual_seed(0)
        network = JointFuseNet().eval()
    else:
        network = load_network(args.checkpoint)
    network = network.to(args.device)
    with torch.inference_mode():
        velocity, resistivity = network(observe[:, :1], observe[:, 1:2, :, :30])
    predictions = {"velocity_m_s": velocity, "resistivity_ohm_m": resistivity}
    if any(not torch.isfinite(value).all() for value in predictions.values()):
        raise ValueError("JointFuseNet produced non-finite predictions.")

    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))["model"]
    output = {
        name: value.cpu().numpy()[0, 0] for name, value in predictions.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, output)
    summary = {
        "mode": "fresh_weights" if args.smoke_test else "checkpoint",
        "outputs": {name: {"shape": list(value.shape)} for name, value in output.items()},
    }

    if args.target is not None:
        target = read_volumes(args.target, "model").to(args.device)
        targets = (target[:, :1], target[:, 1:2, :, :30])
        for channel, ((name, prediction), truth) in enumerate(zip(predictions.items(), targets)):
            residual = prediction - physical_volume(truth, statistics, channel)
            summary["outputs"][name]["physical_rmse"] = residual.square().mean().sqrt().item()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
