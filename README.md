# PSBG

**Physics-Inspired Temporal Enhancement and Boundary-Aware Decoding for Infrared Gas Video Segmentation**

PSBG is a framework for segmenting faint, semi-transparent gas plumes in infrared videos. It combines **Scene-aware Physics-inspired Temporal Enhancement (SPTE)** with a **Boundary-aware Gas Decoding Head (BGDH)** to integrate temporal evolution, scene memory, and boundary guidance. We evaluate PSBG on **SimGas** and **RIGVS**, our real-world infrared gas video dataset with pixel-level annotations.


## Framework

![Overall architecture of PSBG](assets/framework.png)

Given a current frame and two historical frames, SPTE combines **Scene Memory (SM)** with **Physics-inspired Temporal Evolution (PITE)** to enhance the encoder input. Its scene-memory pathway uses references from the same video, without requiring a pre-acquired gas-free background image. A **MiT-B1** encoder extracts multi-scale features, and BGDH coordinates boundary-guided feature refinement with boundary-controlled logit diffusion to generate the segmentation map.

The physics-inspired evolution operates in learned feature space; its outputs are not calibrated fluid parameters.

## Experimental Results

### Performance and Model Complexity

![Radar-chart comparison on SimGas and RIGVS](assets/radar-comparison.png)

Comparison of segmentation performance and model complexity on **SimGas** (left) and **RIGVS** (right). Arrows indicate the preferred direction for each metric.

### Qualitative Comparisons

**SimGas**

![Qualitative comparison on the SimGas test set](assets/qualitative-simgas.png)

**RIGVS**

![Qualitative comparison on the RIGVS test set](assets/qualitative-rigvs.png)

Each row shows one example. From left to right: input, ground truth (GT), SeaFormer, SegFormer, Gasformer, GasSeg, FGSTP, and PSBG (Ours). In these figures, **white**, **red**, and **green** indicate true positives, false positives, and false negatives, respectively. Red boxes in the input images mark gas regions of interest.

## Getting Started (After Code Release)

The code release will contain the complete PSBG model and the dependencies required for training, data processing, and evaluation. All commands below are intended to run from the repository root.

### Repository Structure

```text
.
├── README.md
├── assets/                       # Figures displayed in this README
├── configs/
│   └── psbg_full.py               # Standalone full-model configuration
├── mmseg/
│   ├── apis/                     # Training, validation, and testing
│   ├── core/                     # Metrics, evaluation hooks, and EMA
│   ├── datasets/                 # Video loading, frame ordering, and augmentation
│   ├── models/                   # Backbone, decoder, and losses
│   ├── ops/                      # Required operations
│   └── utils/                    # Logging and environment utilities
├── tools/
│   ├── train.py
│   └── test.py
├── requirements/
│   └── runtime.txt
├── requirements.txt
├── setup.py
├── LICENSE
└── .gitignore
```

The bundled `mmseg/` directory is a subset of the original framework, so a separate MMSegmentation installation is not required. The boundary-aware decoder is implemented as `GasPhysEdgeHead`. Some legacy `TDMGas` class names are retained for compatibility with existing full-model checkpoint parameter names. The bundled framework retains the version identifier `0.11.0`; it is not a complete upstream MMSegmentation distribution.

### Environment and Installation

The reference environment reported for this implementation is:

| Component | Version |
| --- | --- |
| Operating system | Windows |
| Python | 3.8.20 |
| PyTorch | 2.0.0 |
| CUDA | 11.8 |
| MMCV-full | 1.7.2 |
| timm | 0.9.5 |
| MMEngine | 0.10.7 |

The implementation uses the MMCV 1.x runner and parallel APIs. Use **MMCV-full 1.7.2** with CUDA operators for the complete model; MMCV 2.x is incompatible with these APIs.

Installation example for Python 3.8 and CUDA 11.8:

```bash
conda create -n psbg python=3.8 -y
conda activate psbg
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv-full==1.7.2 --only-binary=mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The installation example uses TorchVision 0.15.1, while the existing reference environment reported TorchVision 0.15.0. This exact installation sequence has not been verified in a separate clean environment. Installation references: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) and the [OpenMMLab wheel index](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html).

Check the environment and command-line entry points:

```bash
python -c "import torch, mmcv; from mmcv.ops import DeformConv2d; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"
python tools/train.py --help
python tools/test.py --help
```

Training and evaluation require an NVIDIA GPU. Although the implementation includes a convolution fallback when deformable convolution is unavailable, use MMCV-full with CUDA operators to preserve the intended model structure and checkpoint compatibility.

### Data Preparation

Prepare each dataset locally using the following structure:

```text
data/gasvid/
├── train.txt
├── val.txt
├── test.txt
└── data/
    ├── video_001/
    │   ├── origin/
    │   │   ├── image_1.jpg
    │   │   ├── image_2.jpg
    │   │   └── ...
    │   └── mask/
    │       ├── image_1.png
    │       ├── image_2.png
    │       └── ...
    └── ...
```

Frames are sorted by their numeric frame IDs. The temporal sampling stride counts positions in the sorted image sequence. To use another dataset, follow the same directory structure and pass its root directory through `--data-root`.

### Pretrained Weights

Obtain the ImageNet-pretrained **MiT-B1** weights from the [SegFormer upstream project](https://github.com/NVlabs/SegFormer#training) and place them at:

```text
pretrained/mit_b1.pth
```

To use another local path, add `--options model.pretrained=/path/to/mit_b1.pth` to the training command. For a random-initialization code check, use `--options model.pretrained=None`; this changes the initialization used for training.

Evaluation requires a **full-model checkpoint** produced by training. The MiT-B1 pretrained backbone weights alone cannot be used as a complete PSBG checkpoint.

### Training

```bash
python tools/train.py --config configs/psbg_full.py --data-root ./data/gasvid --work-dir ./work_dirs/gasvid --seed 207
```

`--data-root` updates the training, validation, and test dataset paths. When switching datasets, also use a separate `--work-dir`. Relative paths are resolved from the working directory where the command is run.

The default configuration includes:

| Setting | Value |
| --- | --- |
| Input window | Current frame + two historical frames |
| Temporal sampling stride | 5 ordered image positions |
| Training iterations | 160,000 |
| Optimizer | AdamW |
| Base learning rate | `6e-5` |
| Warmup iterations | 1,500 |
| Validation and checkpoint interval | 10,000 iterations |
| EMA decay | `0.999` |
| Segmentation losses | OHEM cross-entropy + Dice |
| Additional supervision | Boundary loss |

See `configs/psbg_full.py` for the complete settings. Training saves `config.py`, logs, and checkpoints in the working directory.

Resume training from a checkpoint:

```bash
python tools/train.py --config configs/psbg_full.py --data-root ./data/gasvid --work-dir ./work_dirs/gasvid --resume-from ./work_dirs/gasvid/iter_100000.pth --options model.pretrained=None
```

Resuming restores the model and optimizer states. EMA shadow parameters are not saved separately and are reinitialized from the restored model parameters.

### Testing and Evaluation

Use the configuration saved during training to evaluate the videos listed in `test.txt`:

```bash
python tools/test.py --config ./work_dirs/gasvid/config.py --checkpoint ./work_dirs/gasvid/iter_160000.pth --data-root ./data/gasvid --split test --metrics-out ./outputs/gasvid/metrics.json --mask-dir ./outputs/gasvid/masks
```

Alternatively, set `--config configs/psbg_full.py`. The test entry point disables backbone pretraining and loads the full-model weights from `--checkpoint`.

- **Split:** The default configuration uses `val.txt`. Specify `--split test` to evaluate `test.txt`.
- **Metrics:** Evaluation prints per-class IoU, Recall, Precision, and Dice. The JSON summary contains `mIoU`, `mRecall`, `mPrecision`, and `mDice`, averaged over the background and gas classes and reported on a 0-100 percentage scale with two decimal places.
- **Masks:** `--mask-dir` saves class-index PNG files under `<mask-dir>/result_submission/<video>/<frame>.png`.
- **Overlays:** Add `--show-dir ./outputs/gasvid/overlays` to export visualization overlays.
- **Difference maps:** Add `--diff-dir ./outputs/gasvid/diffs` to export prediction-versus-label difference maps.
- **Raw predictions:** Add `--out ./outputs/gasvid/predictions.pkl` to save per-frame predictions as a pickle file. Pickle output is disabled by default.

## Acknowledgements and License

This implementation builds on [SegFormer](https://github.com/NVlabs/SegFormer) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). The code release will retain the inherited license and the copyright and source notices in the included code. Please consult the `LICENSE` file distributed with the code release for the applicable terms.

The upstream SegFormer reference is:

```bibtex
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

Citation details for PSBG will be added after publication.
