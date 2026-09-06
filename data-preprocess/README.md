# SPVC 退化视频对制作指南

本目录只保留一个入口脚本：

```text
data-preprocess/
├── README.md
└── render_degradations.py
```

脚本负责加载 DriveStudio 的场景 checkpoint，渲染退化 RGB 帧，并通过
DriveStudio 的 `save_videos` 保存视频和逐帧图片。随后将同一场景、同一组
相机、同一时间范围的退化输出与干净参考输出配对，即可得到 SPVC 所需的
degraded/reference video pair。

## DriveStudio 与 OmniRe

本流程使用的 DriveStudio 是一个面向城市场景重建和仿真的 3D Gaussian
Splatting 代码库；其中的官方 OmniRe 实现用于从驾驶日志重建动态城市场景，
并提供多相机渲染能力。退化视频由这些场景模型的不同训练状态或不同相机
可见性产生，而不是由一个额外的视频生成模型产生。

- DriveStudio 代码库：<https://github.com/ziyc/drivestudio>
- OmniRe 项目主页：<https://ziyc.github.io/omnire/>
- OmniRe 论文：<https://arxiv.org/abs/2408.16760>

论文引用：

```bibtex
@article{chen2024omnire,
    title={OmniRe: Omni Urban Scene Reconstruction},
    author={Chen, Ziyu and Yang, Jiawei and Huang, Jiahui and Lutio, Riccardo de
        and Esturo, Janick Martinez and Ivanovic, Boris and Litany, Or
        and Gojcic, Zan and Fidler, Sanja and Pavone, Marco and Song, Li
        and Wang, Yue},
    journal={arXiv preprint arXiv:2408.16760},
    year={2024}
}
```

## 环境和输入

`render_degradations.py` 会动态导入完整的 DriveStudio，因此本目录不包含
DriveStudio 源码、CUDA 扩展、数据集或 checkpoint。请在已经能够运行
DriveStudio 的环境中执行：

```bash
cd /path/to/drivestudio
export PYTHONPATH=$(pwd)
python -c "from datasets.driving_dataset import DrivingDataset; from models.video_utils import render_images, save_videos"
```

脚本要求：

1. `--drivestudio-root` 下存在 `datasets/driving_dataset.py`、
   `models/video_utils.py` 和 DriveStudio 的其他依赖；
2. `--checkpoint-root/<scene>/` 下存在 `config.yaml` 以及对应 checkpoint；
3. `config.yaml` 中的数据路径指向已经处理好的 DriveStudio 数据；
4. 为了获得逐帧目录，配置中的 `logging.save_seperate_video` 应设为 `true`。

checkpoint 根目录的标准形式为：

```text
checkpoint-root/
└── <scene>/
    ├── config.yaml
    ├── checkpoint_01000.pth     # 欠拟合模型
    └── checkpoint_final.pth     # 正常 40000-step 模型
```

仅有原始图片或仅有 DriveStudio 数据目录不能直接执行渲染；每个场景都必
须有与 checkpoint 一起保存的 `config.yaml`。

## 三类退化 checkpoint 的训练方式

以下约定是本项目制作退化数据时使用的训练策略。训练入口仍使用
DriveStudio 的 `tools/train.py` 和 OmniRe 配置；本目录只负责后续渲染。

### 1. Underfitting：训练 1000 steps

使用正常 OmniRe 配置和完整训练相机，但把 `trainer.optim.num_iters` 设为
`1000`。训练结束后使用 `checkpoint_01000.pth` 渲染。该 checkpoint 保留了
明显的欠拟合状态，用于制作 underfitting 退化：

```bash
cd /path/to/drivestudio
export PYTHONPATH=$(pwd)
python tools/train.py \
  --config_file configs/omnire.yaml \
  --output_root /path/to/checkpoints \
  --project spvc_degradation \
  --run_name <dataset>_<scene>_underfitting \
  dataset=<dataset>/<num_cams>cams \
  data.scene_idx=<scene> \
  data.start_timestep=0 \
  data.end_timestep=-1 \
  trainer.optim.num_iters=1000
```

请确认保存频率允许在 1000 step 保存 checkpoint；标准配置通常会生成
`checkpoint_01000.pth`。

### 2. Cross-view：三相机训练，剩余三相机渲染

对于六相机数据，先只选择三个相机训练 OmniRe，再用训练得到的场景模型
渲染没有参与训练的另外三个相机。例如训练相机为 `[0, 1, 2]`，渲染相机
为 `[3, 4, 5]`。训练阶段使用三相机数据配置：

```bash
python tools/train.py \
  --config_file configs/omnire.yaml \
  --output_root /path/to/checkpoints \
  --project spvc_degradation \
  --run_name <dataset>_<scene>_cross_view \
  dataset=<dataset>/3cams \
  data.scene_idx=<scene> \
  data.start_timestep=0 \
  data.end_timestep=-1 \
  trainer.optim.num_iters=40000
```

渲染时通过 `--cameras` 指定未参与训练的相机。下面的例子对应训练
`[0, 1, 2]`、渲染 `[3, 4, 5]`；实际相机编号应与数据集配置一致：

```bash
python data-preprocess/render_degradations.py \
  --mode cross-reference \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --cameras 3,4,5
```

脚本中的 `cross-reference` 就是这里所说的 cross-view 渲染模式。若不传
`--cameras`，脚本默认使用 `[0, 3, 4]`；要复现“三相机训练、剩余三相机
渲染”，应显式传入未训练的相机列表。

### 3. Random mask：正常 checkpoint 加随机遮挡

Random mask 不重新训练模型。先按正常 OmniRe 配置训练 `40000` steps，使用
生成的 `checkpoint_final.pth` 渲染，然后对保存的 RGB 帧逐张施加随机矩形、
椭圆或多边形遮挡。使用固定 `--seed` 可以复现同一组遮挡：

```bash
python data-preprocess/render_degradations.py \
  --mode random-mask \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --seed 0
```

该模式默认查找 `checkpoint_final.pth`。随机遮挡是在
`<scene>_rgbs/` 中已经保存的图片上完成的；如果要得到遮挡后的 MP4，必须
使用这些遮挡后的 PNG 重新编码，不能直接使用遮挡前写出的 MP4。

## 渲染退化视频和图片

### Underfitting

```bash
python data-preprocess/render_degradations.py \
  --mode underfitting \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/underfitting_checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset>
```

该模式默认读取每个场景的 `checkpoint_01000.pth`。

### Cross-view

```bash
python data-preprocess/render_degradations.py \
  --mode cross-reference \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/cross_view_checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --cameras 3,4,5
```

### Random mask

```bash
python data-preprocess/render_degradations.py \
  --mode random-mask \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/normal_40000_checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --seed 0
```

可以用 `--scenes 001 002` 只处理指定场景；也可以用 `--checkpoint` 指定
不同于默认名称的 checkpoint。`--device auto` 会在有 CUDA 时使用 GPU。

输出目录形式为：

```text
degraded_outputs/
├── underfitting_data/<dataset>/<scene>_rgbs.mp4
├── underfitting_data/<dataset>/<scene>_rgbs/000_000.png
├── cross_reference_data/<dataset>/<scene>_rgbs.mp4
├── cross_reference_data/<dataset>/<scene>_rgbs/000_000.png
└── random_mask_data/<dataset>/<scene>_rgbs/000_000.png
```

当 `logging.save_seperate_video=true` 时，`<scene>_rgbs/` 是逐时间戳、逐相机
保存的 PNG 目录，文件名形如 `<timestamp>_<camera>.png`。如果配置关闭了
单独视频保存，可能只得到拼接视频而没有可供 random mask 处理的帧目录。

## 制作退化—参考视频对

对每个场景，保持下面四项完全一致：

- scene ID；
- 相机列表及相机顺序；
- 起止时间戳和帧数；
- FPS、分辨率和视频布局。

然后把退化渲染结果作为 `control/degraded video`，把原始传感器视频或正常
40000-step OmniRe 渲染结果作为 `reference/GT video`。推荐按相同文件名建立
配对关系：

```text
degraded/<mode>/<dataset>/<scene>_rgbs.mp4
reference/<dataset>/<scene>_rgbs.mp4
```

如果训练或渲染使用了六相机中的不同子集，必须先确认两侧视频的相机和时间
轴仍然一一对应；cross-view 的“未训练相机”应与参考视频中的同一相机对应。

Random mask 的配对流程是：

1. 用 `checkpoint_final.pth` 渲染并保存逐帧 PNG；
2. 对 `<scene>_rgbs/` 中的 PNG 执行脚本内置的随机遮挡；
3. 将遮挡后的 PNG 编码为 control video；
4. 用同一 scene、相机和时间窗口的干净视频作为 reference video。

例如，对单个相机的遮挡帧重新编码：

```bash
ffmpeg -framerate 12 \
  -i /path/to/random_mask_data/<dataset>/<scene>_rgbs/%03d_0.png \
  -c:v libx264 -pix_fmt yuv420p \
  /path/to/pairs/random_mask/<scene>_camera0.mp4
```

如果需要保持多相机拼接布局，应按照 DriveStudio 的 `layout` 对同一时间戳
的多张相机 PNG 重新拼接后再编码。

## 参数速查

```text
--mode              cross-reference | underfitting | random-mask
--drivestudio-root  DriveStudio 仓库根目录
--checkpoint-root   直接包含各 scene 子目录的 checkpoint 根目录
--output-root       退化视频和图片输出根目录
--dataset-name      nuscenes | pandaset | waymo
--checkpoint        可选，覆盖默认 checkpoint 文件名
--scenes            可选，只处理指定 scene
--cameras           可选，覆盖渲染相机列表
--device            auto | cuda | cpu
--fps               输出 FPS，默认 12
--seed              random-mask 的随机种子，默认 0
```

本目录不包含 checkpoint、原始数据或 DriveStudio 依赖；发布或复现时请按
各数据集和 DriveStudio 的许可证要求准备这些外部资源，并引用 OmniRe。
