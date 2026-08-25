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

验证带版本的 synthetic 或后续相机 bundle：

```bash
camera-rig artifact validate --bundle path/to/camera_bundle.json
```

配置根节点使用单数 `camera`。未知字段、复数 `cameras` 根节点、未知数据流、非法尺寸或帧率以及
非字符串序列号都会被拒绝，不会被静默转换。

## Python API

```python
import camera_rig

print(camera_rig.__version__)
```

硬件无关契约位于 `camera_rig.core`，目标插件接口位于 `camera_rig.targets`，确定性产物工具位于
`camera_rig.artifacts`。导入这些模块不需要 RealSense 或 OpenCV。

## 产物

`CameraBundle` 是面向一个物理相机的带版本顶层 JSON 契约，可包含设备身份、数据流配置、各流
内参、设备内部光学坐标变换、深度比例、可选固定安装标定、质量结果和来源信息。JSON 写入具有
确定性和原子性，读取持久化变换时会重新验证。

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
