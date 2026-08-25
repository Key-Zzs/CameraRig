# CameraRig

CameraRig 是面向机器人系统的单物理相机采集、标定参数管理、验证和重放 Python 工具库。

## 职责范围

一个 `CameraSession` 或 `CameraDriver` 实例只表示一个物理相机。本库的职责边界包括设备发现与
生命周期、设备内部各数据流、设备内部时间信息、SDK 原厂内参、设备内部光学数据流坐标变换、
单相机固定安装外参、质量验证以及采集与重放产物。

## 非目标

CameraRig 不负责多相机分组或同步、相机间标定、机器人正向运动学、点云构建、FFS、
DepthAnything、建图、融合或 TSDF 重建。下游可以消费 CameraRig 产物，但 CameraRig 不反向
依赖下游。

## 安装

CameraRig 要求 Python 3.10 或更高版本。

```bash
python -m pip install -e ".[dev]"
```

如需 D435i 采集和 PNG 预览，请安装官方 RealSense wheel 与可视化可选依赖：

```bash
python -m pip install -e ".[dev,realsense,viz]"
```

物理设备序列号应只保存在私有且被忽略的配置中。公开示例刻意使用
`REPLACE_WITH_DEVICE_SERIAL` 占位符。

## 命令行

```bash
camera-rig --help
camera-rig --version
python -m camera_rig --help
```

验证仓库内严格的单相机契约示例：

```bash
camera-rig config validate --config configs/examples/single_camera_contract.yaml
```

验证带版本的相机 bundle：

```bash
camera-rig artifact validate --bundle path/to/camera_bundle.json
```

以只读方式发现并检查 D435i：

```bash
camera-rig device list --driver realsense
camera-rig device inspect --config .local/configs/d435i.yaml --show-profiles
camera-rig device smoke --config .local/configs/d435i.yaml --cycles 5 \
  --report .local/reports/device-smoke.json
```

从 pipeline 实际启用的 profiles 导出原厂参数，然后验证、保存并离线重放原始数据：

```bash
camera-rig calibration factory export \
  --config .local/configs/d435i.yaml \
  --output .local/artifacts/factory_calibration.json
camera-rig capture validate-streams \
  --config .local/configs/d435i.yaml --frames 300 \
  --report .local/reports/stream-validation.json
camera-rig capture snapshot \
  --config .local/configs/d435i.yaml --frames 30 \
  --output .local/artifacts/sequence
camera-rig replay validate --artifact .local/artifacts/sequence
```

配置根节点使用单数 `camera`。未知字段、复数 `cameras` 根节点、未知数据流、非法尺寸或帧率以及
非字符串序列号都会被拒绝，不会被静默转换。

## Python API

```python
import camera_rig
from camera_rig.capture import CameraSession, ReplayCameraSession
from camera_rig.config import load_config

print(camera_rig.__version__)

config = load_config("private-d435i.yaml")
with CameraSession.from_config(config) as camera:
    frame = camera.capture()

with ReplayCameraSession.from_artifact("capture-artifact") as replay:
    restored = replay.capture()
```

硬件无关契约位于 `camera_rig.core`，目标插件接口位于 `camera_rig.targets`，确定性产物工具位于
`camera_rig.artifacts`。导入这些模块不需要 RealSense 或 OpenCV。

## 产物

`CameraBundle` 是面向一个物理相机的带版本顶层 JSON 契约，可包含设备身份、数据流配置、各流
内参、设备内部光学坐标变换、深度比例、可选固定安装标定、质量结果和来源信息。JSON 写入具有
确定性和原子性，读取持久化变换时会重新验证。

采集产物保存原始 `uint8` RGB、原始 `uint16` 深度、左右两路原始 `uint8` 红外、逐流时间
元数据、原厂参数产物以及 SHA-256。manifest 中所有路径均相对产物根目录。PNG 仅用于诊断，
重放会先校验完整产物，再从 NPZ 恢复原始 `CameraFrame`，且不导入或依赖 RealSense SDK。

```text
capture-artifact/
├── manifest.json
├── factory_calibration.json
├── checksums.sha256
├── frames/
│   ├── frame_000000.npz
│   └── frame_000000.meta.json
└── previews/
    ├── frame_000000_color.png
    ├── frame_000000_depth.png
    ├── frame_000000_ir_left.png
    ├── frame_000000_ir_right.png
    └── mosaic.png
```

## 坐标约定

向量采用列向量，坐标系为右手系，长度单位为米，时间单位为纳秒。刚体变换采用 4 x 4 齐次
SE(3) 矩阵并命名为 `T_target_from_source`，其数学含义是
`p_target = T_target_from_source @ p_source`。持久化的变换记录必须显式包含
`source_frame` 和 `target_frame`。

## 许可证

本项目采用 Apache License 2.0，详见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

## 致谢与引用

致谢信息见 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)，引用元数据见
[CITATION.cff](CITATION.cff)。

## English documentation

See [README.md](README.md) for the English documentation.
