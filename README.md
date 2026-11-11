# PSBG

红外气体视频分割完整模型的训练与评估代码。模型包含场景记忆（SM）、物理引导时序编码（PITE）、MiT-B1 骨干，以及带边界细化与各向异性扩散的 `GasPhysEdgeHead` 解码头。

本仓库仅保留完整模型及其必要的训练、数据处理、评估依赖。内置的 `mmseg/` 是从原工程整理的框架子集，不需要另外安装 MMSegmentation。源码中的部分 TDMGas 类名予以保留，以兼容原有完整模型检查点的参数名称。

## 目录

```text
.
├── configs/psbg_full.py       # 独立的完整模型配置
├── mmseg/
│   ├── apis/                # 训练、验证与测试
│   ├── core/                # 指标、评估钩子与 EMA
│   ├── datasets/            # 视频读取、数字帧号排序和数据增强
│   ├── models/              # 完整骨干、解码头和损失函数
│   ├── ops/                 # 插值等必要操作
│   └── utils/               # 日志与环境信息
├── tools/train.py
├── tools/test.py
├── requirements/runtime.txt
├── requirements.txt
├── setup.py
├── LICENSE
└── .gitignore
```

数据集、预训练权重、训练检查点、日志和实验输出均需在本地自行准备，不随源码提供。

## 环境与安装

本次本地验证使用 Windows、Python 3.8.20、PyTorch 2.0.0、CUDA 11.8、MMCV-full 1.7.2、timm 0.9.5 和 MMEngine 0.10.7。该代码使用 MMCV 1.x 的 runner 和 parallel 接口，不适用于 MMCV 2.x。

以下为 Python 3.8 + CUDA 11.8 的安装示例。先进入仓库根目录，再在独立环境中执行：

```bash
conda create -n psbg python=3.8 -y
conda activate psbg
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv-full==1.7.2 --only-binary=mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

PyTorch 与 TorchVision 的安装组合来自 [PyTorch 官方旧版本说明](https://pytorch.org/get-started/previous-versions/)，MMCV wheel 来自 [OpenMMLab 官方索引](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html)。本机已有环境的 TorchVision 为 0.15.0；上面的安装示例采用官方配套的 0.15.1，未另建环境重装验证。


```bash
python -c "import torch, mmcv; from mmcv.ops import DeformConv2d; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"
python tools/train.py --help
python tools/test.py --help
```

正式训练和测试使用 NVIDIA GPU。原实现带有可变形卷积不可导入时的卷积回退路径；为保持完整模型结构和检查点一致，应使用包含 CUDA 算子的 MMCV-full。

## 数据准备

将数据集按以下结构放在本地，例如 `data/gasvid/`；IGV 等数据也使用相同结构，并通过 `--data-root` 指定：

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

## 预训练权重

从 [SegFormer 上游项目](https://github.com/NVlabs/SegFormer#training) 获取 ImageNet 预训练的 MiT-B1 权重，并放到本地：

```text
pretrained/mit_b1.pth
```

也可以在训练时用 `--options model.pretrained=/path/to/mit_b1.pth` 指定文件。若仅检查随机初始化的代码流程，可设置 `--options model.pretrained=None`；该设置不等价于正式的预训练初始化实验。

完整模型测试需要训练生成的完整检查点，MiT-B1 预训练权重不能直接作为完整模型检查点使用。本源码包不附带训练权重或评估结果。

## 训练

所有相对路径均相对于运行命令时的工作目录；下面的命令在仓库根目录执行。

```bash
python tools/train.py --config configs/psbg_full.py --data-root ./data/gasvid --work-dir ./work_dirs/gasvid --seed 207
```

`--data-root` 同时更新训练、验证和测试数据路径。切换到 IGV 时，将其改为实际 IGV 根目录，并使用另一个 `--work-dir` 即可。也可以直接修改配置顶部的 `data_root`。

默认设置为 160000 次迭代、AdamW、学习率 `6e-5`、1500 次 warmup、每 10000 次迭代验证并保存检查点，以及 decay 为 `0.999` 的 EMA。分割损失为 OHEM 交叉熵与 Dice，另含边界监督；完整参数见 `configs/psbg_full.py`。

训练在输出目录保存 `config.py`、日志和检查点。配置使用 Python 标准库序列化，保留元组等配置类型，并避免旧版 MMCV 配置格式化器对 YAPF 版本及 Windows 路径的依赖。

恢复训练示例：

```bash
python tools/train.py --config configs/psbg_full.py --data-root ./data/gasvid --work-dir ./work_dirs/gasvid --resume-from ./work_dirs/gasvid/iter_100000.pth --options model.pretrained=None
```

检查点恢复会加载模型及优化器状态。原有 EMA 钩子的影子参数不单独保存，恢复时会从恢复后的模型参数重新初始化。

## 测试与评估

优先使用训练时保存的配置。以下命令在 `test.txt` 指定的视频上评估：

```bash
python tools/test.py --config ./work_dirs/gasvid/config.py --checkpoint ./work_dirs/gasvid/iter_160000.pth --data-root ./data/gasvid --split test --metrics-out ./outputs/gasvid/metrics.json --mask-dir ./outputs/gasvid/masks
```

如果使用源码配置，将 `--config` 改为 `configs/psbg_full.py`。测试入口会关闭骨干预训练权重加载，只读取 `--checkpoint` 指定的完整模型权重。

- 默认配置沿用验证划分，即 `val.txt`；`--split test` 显式选择 `test.txt`。
- 评估输出每类 IoU、Recall、Precision、Dice，并将两类平均后的 `mIoU`、`mRecall`、`mPrecision`、`mDice` 写入 JSON。汇总指标沿用原工程的百分数尺度（0–100，保留两位小数），不是气体单类别指标。
- `--mask-dir` 导出类别索引 PNG，路径为 `<mask-dir>/result_submission/<video>/<frame>.png`。
- `--show-dir ./outputs/gasvid/overlays` 导出叠加图；`--diff-dir ./outputs/gasvid/diffs` 导出预测与标签的差异图。
- `--out ./outputs/gasvid/predictions.pkl` 可选保存逐帧预测，默认不保存 pickle。

## 来源与许可证

本工程沿用原项目的 [LICENSE](LICENSE)，并保留所复制代码中的版权与来源声明。上游项目为 [SegFormer](https://github.com/NVlabs/SegFormer) 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)。内置 `mmseg` 的版本标识沿用原工程的 `0.11.0`，不代表完整的上游发行版。

使用限制请查阅许可证全文。引用上游 SegFormer 时可使用：

```bibtex
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
