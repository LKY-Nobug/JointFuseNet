# JointFuseNet

JointFuseNet maps seismic images and coarse resistivity models to 3D velocity and resistivity estimates for tunnel ahead prospecting. Cross-attention exchanges features within the shared seismic and electrical coverage. The velocity output extends over the full 120 m axial grid and the resistivity output covers the first 30 m.

## Installation

Use Python 3.12. The example runs on a CPU. For GPU use, install the CUDA build of PyTorch 2.6.0 that matches your system before installing the remaining dependencies.

```sh
python -m venv .venv
# Activate the environment before installing.
python -m pip install -r requirements.txt
```

## Run the synthetic example

### Download the pretrained weights

The pretrained checkpoint is distributed as an asset of the GitHub release [v1.0.0](https://github.com/LKY-Nobug/Joint-inversion/releases/tag/v1.0.0). Download `jointfusenet-se-maxpool.pt` from the release assets and place it at `checkpoints/jointfusenet-se-maxpool.pt` in the repository. The inference script looks for the checkpoint at this path by default.

Verify the downloaded file with SHA-256:

```text
93df7fb7a0601545ac81f0c0a06939b16e6a166400bc45a45b663b5dfed6dd06  jointfusenet-se-maxpool.pt
```

On Linux or macOS, run `sha256sum checkpoints/jointfusenet-se-maxpool.pt`. On Windows PowerShell, run `Get-FileHash .\checkpoints\jointfusenet-se-maxpool.pt -Algorithm SHA256`.

```sh
python infer.py --target examples/model0.mat
```

The command loads the pretrained weights from `checkpoints/jointfusenet-se-maxpool.pt` and writes `results/predictions.mat` with velocity in m/s and resistivity in ohm m. The expected sizes are `(30, 120, 30)` for velocity and `(30, 30, 30)` for resistivity. The second output axis is the tunnel axis. Use `--smoke-test` to check the interfaces with deterministic fresh weights.

The examples are the first ten cases of the prepared synthetic test split in numeric filename order (IDs 0 through 9). Each `data<ID>.mat` has a matching `model<ID>.mat`. The following commands run all ten cases in Bash:

```sh
for id in {0..9}; do
    python infer.py --input "examples/data${id}.mat" --target "examples/model${id}.mat" \
        --output "results/predictions${id}.mat" || exit 1
done
```

The MATLAB input contains a `data` array of shape `(2, 30, 30, 120)`. Channel 0 is a seismic image and channel 1 is a coarse resistivity model. The optional reference file uses the same shape and contains a `model` array with velocity and resistivity labels. Supply z-score normalized arrays as in the example. For arrays in original units, subtract `mean0` or `mean1` and divide by the matching `std0` or `std1` in the `data` or `model` section of `examples/mean_std.json`. The script converts the stored axis order for the network and crops the electrical input and resistivity reference to the shared 30 m.

Use `--checkpoint` to supply another checkpoint for the same architecture. The optional reference file enables RMSE reporting in physical units.

```sh
python infer.py --checkpoint /path/to/main-model-checkpoint.pth.tar --target examples/model0.mat
```

Use `python infer.py --help` for options. Set `--device cuda:0` to run on a CUDA device. Set `--output PATH` to change the output destination.

## Train the main architecture

Prepare normalized MATLAB pairs in `train/data`, `train/model`, `valid/data`, and `valid/model`, with corresponding names such as `data0.mat` and `model0.mat`. Compute the channel means and standard deviations on the training split and apply the same statistics to all splits. Store them with the keys in `examples/mean_std.json`.

```sh
python train.py /path/to/prepared_dataset --statistics /path/to/mean_std.json --seed 888
```

The training entry point uses the main architecture, AdamW, per-batch polynomial learning-rate decay, gradient clipping, and the MSE plus SSIM objective from the source training script. It supports CUDA and uses the visible GPUs for data parallel training when selected through `cuda:0`. The original batch size of 100 was intended for multi-GPU training. Reduce `--batch-size` for smaller datasets and devices. The training and validation loaders use only complete batches. The example uses the seed specified in the source training script. The exported checkpoint records the epoch and optimizer settings. Run `python train.py --help` for the remaining options.

The repository uses two axial multi-attention blocks (AMABs), AdamW weight decay of `1e-2`, an initial learning rate of `1e-4`, a batch size of 100, and 200 epochs. Training maps predictions and reference volumes to unit intervals for MSE and SSIM. Use `--weight-decay` to change the optimizer setting.

## Reproducibility and licensing

The examples start from prepared inversion inputs and cover ten synthetic test cases. Full training and test-set evaluation require the original prepared data splits. The bundled weights apply to this two-block implementation.

The bundled weights contain the model tensors from the supplied epoch 200 `model_best.pth.tar`. The checkpoint loads strictly into the two-block model derived from `JointResFuseNet_nodrop_multiatten_softmax.py`. The resistivity head uses the declared lower bound of 375 ohm m in place of the undefined `vmin` attribute in the original source. Both heads map a sigmoid output into physical units.

The SHA-256 of the release checkpoint `jointfusenet-se-maxpool.pt` is `93df7fb7a0601545ac81f0c0a06939b16e6a166400bc45a45b663b5dfed6dd06`.

The source code is licensed under the MIT License. See LICENSE for details. The synthetic data in examples/ and model weights in checkpoints/ are outside the scope of this code license.
