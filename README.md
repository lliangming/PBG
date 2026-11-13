# PBG: Physics-Inspired and Boundary-Aware Infrared Gas Video Segmentation

**Tong Zuo, Yuan Liu, Junqi Bai, Xiaodong Kuang, Tong Jiang, Xiubao Sui, and Qian Chen**

PBG segments gas plumes in infrared videos by combining scene context, short-term temporal evolution, and boundary-aware decoding. It uses the current frame and two historical frames, without requiring a pre-acquired gas-free background image.

[Overview](#overview) | [Results](#manuscript-results) | [Installation](#installation) | [Data](#datasets-and-data-preparation) | [Training](#training) | [Evaluation](#evaluation) | [Citation](#citation)

![Overall architecture of PBG, including SPTE, the MiT encoder, and BGDH.](assets/pbg_framework.png)

*Architecture figure from the manuscript: (a) the overall pipeline, (b) Scene-aware Physics-Inspired Temporal Enhancement, and (c) the Boundary-aware Gas Decoding Head.*

## Manuscript results

The following values are reported in Tables 1 and 2 of the manuscript. **They have not been re-evaluated for the current implementation without the two scalar coefficients.** All metrics are percentages.

| Dataset | mIoU | mF | mS | mP |
|---|---:|---:|---:|---:|
| SimGas | 87.19 | 92.77 | 92.64 | 92.94 |
| RIGVS | 86.28 | 92.10 | 90.34 | 94.02 |

The manuscript reports 14.82M parameters and 67.13 GFLOPs for PBG. These are manuscript reference values; this source release does not include the benchmark scripts, trained checkpoints, or dataset files used to produce the tables.

## Installation

The code uses the MMCV 1.x runner and parallel APIs, with a bundled subset of MMSegmentation. Do not install a separate MMSegmentation package over the local `mmseg` module.

| Component | Local verification environment |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.0.0 |
| CUDA runtime | 11.8 |
| MMCV-full | 1.7.2 |
| timm | 0.9.5 |
| MMEngine | 0.10.7 |
| Built-in `mmseg` version identifier | 0.11.0 |

The manuscript describes training on an NVIDIA RTX A5000 with 24 GB VRAM. Training and evaluation use an NVIDIA GPU and require the MMCV deformable-convolution operator.

Run the following commands from the repository root in a dedicated environment:

```bash
conda create -n pbg python=3.8 -y
conda activate pbg
python -m pip install torch
python -m pip install mmcv-full
python -m pip install -r requirements.txt
python -m pip install -e . 
```

The installation example follows the [official PyTorch version pairing](https://pytorch.org/get-started/previous-versions/) and the [OpenMMLab CUDA 11.8 / PyTorch 2.0 wheel index](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html). Local checks used an existing environment with TorchVision 0.15.0; the example uses the official 0.15.1 pairing and has not been tested by reinstalling a fresh environment.

Check the imports and command-line interfaces:

```bash
python -c "import torch, mmcv; from mmcv.ops import DeformConv2d; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"
python tools/train.py --help
python tools/test.py --help
```

Use MMCV-full with working CUDA operators. The inherited implementation has a convolutional fallback when deformable convolution cannot be imported, which changes the model structure.

## Datasets and data preparation

The manuscript evaluates PBG on the following video-disjoint splits:

| Dataset | Source | Total videos | Annotated frames | Training videos | Evaluation videos |
|---|---|---:|---:|---:|---:|
| SimGas | Synthetic infrared gas videos | 31 | 12,128 | 20 | 11 |
| RIGVS | Real infrared gas videos collected for the study | 165 | 8,145 | 129 | 36 |

RIGVS was recorded at 320 x 256 resolution and 25 fps with a cooled infrared camera and a 3.2-3.4 micrometer narrow-band filter. Dataset images, masks, and video-ID split lists are **not included** in this source release. Split sizes alone do not reconstruct the exact experimental partitions.

Organize each dataset using the layout expected by the video reader:

```text
data/
├── SimGas/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   └── data/
│       ├── video_001/
│       │   ├── origin/
│       │   │   ├── image_1.jpg
│       │   │   └── image_2.jpg
│       │   └── mask/
│       │       ├── image_1.png
│       │       └── image_2.png
│       └── ...
└── RIGVS/
    └── ...                  # Same directory structure
```

- Each split file contains one video directory name per line, such as `video_001`. Use no empty lines and retain a final newline.
- Default images are `.jpg`; masks are same-stem `.png` files containing class indices: `0` for background, `1` for gas, and `255` for ignored pixels.
- Frames are ordered by the last number in each filename stem, so `image_2.jpg` precedes `image_10.jpg`. Missing numeric IDs or duplicate IDs within one video raise an error. Keep only image files in `origin/`.
- The default three-frame window uses ordered image positions `[t-10, t-5, t]`. The stride counts available images, not differences between filename numbers.
- When history is insufficient, the model uses the current image and bypasses temporal enhancement for that sample. No future frames are used.
- Training samples one target window per video; validation and testing visit every target frame. The training loader drops incomplete batches, so fewer than four training videos require reducing `data.samples_per_gpu`.

Use the dataset's actual protocol to prepare the split lists. `val.txt` is used by the training validation hook. `--split test` explicitly selects `test.txt` for evaluation.

### Pretrained backbone

Obtain ImageNet-pretrained MiT-B1 weights from the [SegFormer project](https://github.com/NVlabs/SegFormer#training) and place them at:

```text
pretrained/mit_b1.pth
```

A different location can be selected with `--options model.pretrained=/path/to/mit_b1.pth`. Setting `model.pretrained=None` is suitable for a code smoke test, but does not reproduce pretrained initialization.

Full-model evaluation requires a trained PBG checkpoint; MiT-B1 backbone weights alone are insufficient. Neither pretrained nor trained weights are bundled with this repository.

## Training

All commands below run from the repository root. Paths supplied to `--data-root`, `--work-dir`, or checkpoint arguments can be absolute or relative to the working directory.

```bash
# SimGas
python tools/train.py --config configs/psbg_full.py --data-root ./data/SimGas --work-dir ./work_dirs/simgas --seed 207

# RIGVS
python tools/train.py --config configs/psbg_full.py --data-root ./data/RIGVS --work-dir ./work_dirs/rigvs --seed 207
```

`--data-root` updates all three dataset entries. The configuration's legacy default is `./data/gasvid`; the commands above override it explicitly.

## Acknowledgments and license

This implementation builds on [SegFormer](https://github.com/NVlabs/SegFormer) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). The local `mmseg` version identifier is inherited from the original project and does not indicate a complete upstream distribution.

The inherited NVIDIA Source Code License for SegFormer is retained in [LICENSE](LICENSE). Refer to that file for the terms, and preserve the upstream copyright and attribution notices when redistributing the code.
