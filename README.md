<h1 align="center">DRWKV: Focusing on Object Edges for Low-Light Image Enhancement</h1>

<p align="center">
  Xuecheng Bai<sup>1,*</sup> &nbsp;
  Yuxiang Wang<sup>2,*</sup> &nbsp;
  Boyu Hu<sup>3</sup> &nbsp;
  Qinyuan Jie<sup>1</sup> &nbsp;
  Chuanzhi Xu<sup>2,&dagger;</sup> &nbsp;
  Hongru Xiao<sup>4</sup> &nbsp;
  Kechen Li<sup>5</sup> &nbsp;
  Vera Chung<sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>Shenyang Ligong University &nbsp;
  <sup>2</sup>The University of Sydney &nbsp;
  <sup>3</sup>University of International Business and Economics<br>
  <sup>4</sup>Tongji University &nbsp;
  <sup>5</sup>Nanjing University of Aeronautics and Astronautics<br>
  <sup>*</sup>Equal contribution &nbsp;
  <sup>&dagger;</sup>Corresponding author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.18594">arXiv</a> |
  <a href="https://openaccess.thecvf.com/content/WACV2026/papers/Bai_DRWKV_Focusing_on_Object_Edges_for_Low-Light_Image_Enhancement_WACV_2026_paper.pdf">Paper PDF</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img width="1000" src="README_files/Overall.png" alt="DRWKV overview">
</p>

## Abstract

Low-light image enhancement remains challenging when severe illumination degradation breaks object edges and fine structures. DRWKV introduces Global Edge Retinex (GER), Evolving WKV Attention, Bilateral Spectrum Aligner (Bi-SAB), and MS2-Loss to improve edge continuity, luminance-chrominance alignment, and artifact suppression for low-light restoration.

## News

- 2026-05-08: Training code has been updated to use the paper-style MS2-Loss components.
- Pretrained checkpoints and standalone inference/evaluation scripts are not included in this repository yet.

## Highlights

- Global Edge Retinex decomposition with illumination, edge, noise, artifact, and reflection-related auxiliary outputs.
- Evolving WKV Attention with spiral scanning for spatial edge continuity.
- Wavelet-based downsampling and cross-feature enhancement modules for multi-scale restoration.
- MS2-Loss with reconstruction, edge sparsity, illumination smoothness, artifact suppression, and parameter regularization terms.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Checkpoints](#checkpoints)
- [Inference and Evaluation](#inference-and-evaluation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

This project uses a custom CUDA WKV operator, so a CUDA-capable PyTorch environment and a working `nvcc` compiler are required.

The code has been smoke-tested on WSL with:

- Python 3.10
- PyTorch 2.9.0+cu130
- CUDA available through WSL

Create or activate your environment:

```bash
conda create -n drwkv python=3.10 -y
conda activate drwkv
```

Install PyTorch following your CUDA version from the official PyTorch instructions, then install the project dependencies:

```bash
pip install timm einops thop tqdm pyyaml pillow tensorboardX
pip install pytorch_wavelets PyWavelets
```

Optional dependencies for legacy loss utilities:

```bash
pip install pytorch-ssim
```

The CUDA extension is compiled automatically when `Block.py` is imported:

```bash
python - <<'PY'
from Block import Net
print("DRWKV import succeeded")
PY
```

## Data Preparation

The current data loader expects paired low-light and normal-light images under each dataset split directory.

Supported low-light folder names:

```text
input, low, lowlight, lq
```

Supported target folder names:

```text
target, gt, high, normal, hq
```

Example layout for LOL-v2 real:

```text
/path/to/LOLv2/Real_captured/
  Train/
    input/
      0001.png
      0002.png
    target/
      0001.png
      0002.png
  Test/
    input/
      0001.png
    target/
      0001.png
```

Image pairs are matched by filename. Edit the dataset paths in one of the config files before training:

- `configs/LOL_v2_real.yaml`
- `configs/LOL_v1.yaml`

Important fields:

```yaml
TRAINING:
  TRAIN_DIR: '/path/to/train/split'
  VAL_DIR: '/path/to/val/split'
  SAVE_DIR: '/path/to/checkpoints'
```

## Training

Train on LOL-v2 real with the default config:

```bash
python train.py \
  --gpu_id 0 \
  --model_name LOL_v2_real \
  --yml_path configs/LOL_v2_real.yaml \
  --epochs 500
```

Train on LOL-v1:

```bash
python train.py \
  --gpu_id 0 \
  --model_name LOL_v1 \
  --yml_path configs/LOL_v1.yaml \
  --epochs 500
```

Resume training by setting `RESUME: True` in the config. The trainer will load the latest checkpoint ending in `_latest.pth` from:

```text
<SAVE_DIR>/<model_name>/models/
```

Load pretrained weights manually:

```bash
python train.py \
  --gpu_id 0 \
  --model_name LOL_v2_real_finetune \
  --yml_path configs/LOL_v2_real.yaml \
  --pretrain_weights /path/to/checkpoint.pth
```

Training logs are written to:

```text
<SAVE_DIR>/<model_name>/log/
```

Model checkpoints are written to:

```text
<SAVE_DIR>/<model_name>/models/
```

## MS2-Loss

The training script uses `MS2Loss` from `loss.py`. Its default weights follow the loss weighting described in the paper:

```yaml
LOSS:
  LAMBDA_RECON: 1.0
  LAMBDA_SPARSE: 0.01
  LAMBDA_SMOOTH: 0.1
  LAMBDA_ARTIFACT: 0.05
  LAMBDA_REG: 0.0001
  EDGE_AWARE_WEIGHT: 10.0
  ARTIFACT_TV_WEIGHT: 0.1
```

The model returns auxiliary tensors during training through:

```python
outputs = model(input_tensor, return_aux=True)
loss, loss_items = criterion(outputs, target_tensor, input_tensor)
```

`loss_items` contains TensorBoard-ready components:

```text
loss/recon, loss/sparse, loss/smooth, loss/artifact, loss/reg, loss/total
```

## Checkpoints

Pretrained checkpoints are not included in this repository yet.

During training, the following checkpoints are saved automatically:

- `model_latest.pth`
- `model_bestPSNR.pth`
- `model_bestSSIM.pth`

Place downloaded or self-trained weights anywhere convenient and pass the path with `--pretrain_weights`.

## Inference and Evaluation

A standalone inference script is not included yet. For quick integration in your own script:

```python
import torch
from Block import Net

model = Net(
    channels=3,
    out_dim=3,
    dim=16,
    img_size=[512, 256, 128],
    num_blocks=[4, 8],
).cuda().eval()

checkpoint = torch.load("/path/to/model_bestPSNR.pth", map_location="cuda")
model.load_state_dict(checkpoint["state_dict"], strict=True)

with torch.no_grad():
    enhanced = model(low_light_tensor.cuda())
```

The training script reports validation PSNR and SSIM when paired validation data is available.

## Project Structure

```text
DRWKV/
  Block.py                         # DRWKV network, WKV attention, GER-related outputs
  loss.py                          # MS2-Loss and legacy loss utilities
  train.py                         # training entry point
  params.py                        # backbone variant and WKV extension usage
  utils.py                         # stochastic depth utility
  configs/
    LOL_v1.yaml
    LOL_v2_real.yaml
  cuda/
    wkv_op.cpp
    wkv_cuda.cu
  custom_utils/
    dataset_utils.py
    data_loaders/lol.py
    warmup_scheduler/scheduler.py
  README_files/
    Overall.png
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'pytorch_wavelets'`

Install the wavelet dependencies:

```bash
pip install pytorch_wavelets PyWavelets
```

### CUDA extension build fails

Check that `nvcc` is available and matches the CUDA toolchain used by PyTorch:

```bash
nvcc --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

### Dataset path error

Make sure `TRAIN_DIR` and `VAL_DIR` point to split roots containing paired low-light and target folders. The filenames inside both folders must match.

### No pretrained model

Pretrained weights are not currently distributed in this repository. Train from scratch or provide your own checkpoint through `--pretrain_weights`.

## Contact

If you have questions, please contact Chuanzhi Xu or Xuecheng Bai:

- xhuanzhi.xu@sydney.edu.au
- bai_xuecheng@163.com

## Citation

If you find this code helpful in your research or work, please cite:

```bibtex
@misc{bai2025drwkvfocusingobjectedges,
      title={DRWKV: Focusing on Object Edges for Low-Light Image Enhancement},
      author={Xuecheng Bai and Yuxiang Wang and Boyu Hu and Qinyuan Jie and Chuanzhi Xu and Hongru Xiao and Kechen Li and Vera Chung},
      year={2025},
      eprint={2507.18594},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.18594},
}
```

## License

This repository is released under the Apache-2.0 license. See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgement

This project is built with reference to the following excellent works:

- [RWKV](https://github.com/BlinkDL/RWKV-LM)
- [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
- [RetinexMamba](https://github.com/YhuoyuH/RetinexMamba)
- [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)

We thank the authors for their open-source contributions.
