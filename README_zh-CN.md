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

如需生成和检测 ChArUco 标定板，请安装对应可选依赖：

```bash
python -m pip install -e ".[charuco]"
```

如需完整的固定相机一键配置运行环境：

```bash
python -m pip install -e ".[dev,provision]"
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

生成标准可打印 ChArUco 板、检测单张图像，或验证 snapshot 产物中的全部彩色帧：

```bash
camera-rig target generate \
  --config configs/targets/charuco_a4_v1.yaml \
  --output .local/targets/charuco_a4_v1
camera-rig target detect \
  --target .local/targets/charuco_a4_v1/target_spec.json \
  --image image.png \
  --output .local/reports/detection.json \
  --overlay .local/overlays/detection.png
camera-rig target validate-artifact \
  --target .local/targets/charuco_a4_v1/target_spec.json \
  --artifact .local/artifacts/sequence \
  --stream color \
  --policy uncertainty_validated \
  --report .local/reports/target-validation.json \
  --overlays .local/overlays/target-validation
```

使用一个严格 YAML 配置固定 D435i。非硬件 `--dry-run` 只检查输入和依赖；live viability
preflight 会打开相机并复用 `provision fixed` 的采集、逐帧 gate、共享位姿求解和最终质量 evaluator，
但绝不发布 bundle 或 fixed provision：

```bash
camera-rig provision fixed \
  --config .local/configs/fixed_provision.yaml \
  --output .local/artifacts/fixed_camera \
  --dry-run
camera-rig provision preflight \
  --config .local/configs/fixed_provision.yaml \
  --report .local/reports/fixed-provision-preflight.json \
  --overlays .local/overlays/fixed-provision-preflight \
  --evidence-root .local/validation/structured-gate/camera_a/repeat_01
```

`uncertainty_validated_v1` 的显式状态是 `HOLD`，不是 frozen release preset。指定
`--evidence-root` 时，live preflight 会保留私有 capture/evaluation 证据；即使候选数值检查
通过，也会报告 `UNCERTAINTY_VALIDATED_PRESET_NOT_RELEASED` 与 `would_publish=false`。
`provision fixed` 会拒绝为该 HOLD 策略构建 canonical CameraBundle。保留的 capture 只能用于
离线验证，不能当作 provision。

可在不修改 capture、calibration 或 provision artifact 的前提下，对某次保留采集执行内存中的
K/D/target 反事实分析：

```bash
camera-rig calibration evaluate-model-counterfactuals \
  --detection-report .local/validation/structured-gate/camera_a/repeat_01/target/detection_report.json \
  --factory-calibration .local/validation/structured-gate/camera_a/repeat_01/factory/factory_calibration.json \
  --output .local/validation/structured-gate/camera_a/repeat_01/model-counterfactuals.json
```

报告中的位姿变化是相对 retained-data baseline 的敏感度，不是 ground-truth pose bias；该输出
只能作为分析证据，不能据此 release 策略。

固定相机工作流把 `workspace` 明确定义为持久化的 ChArUco 目标板坐标系，并在验证通过的
`CameraBundle` 中输出 `T_workspace_from_<camera>/ir_left_optical`。采集期间相机和目标板都保持
固定；多帧用于衡量检测与位姿重复性，而不是重新标定内参。详见
[固定相机外参](docs/fixed-camera-calibration.md)、
[固定相机配置](docs/fixed-camera-provisioning.md) 与
[标定质量](docs/calibration-quality.md)。

候选验证可设置 `target.detection_policy: uncertainty_validated`，它选择历史 v1 HOLD profile，
当前不具备 production provision 资格。独立命名的 `uncertainty_validated_v2` structured
policy 同样保持 HOLD；当前代码没有 authenticated release loader。未来 release 尝试还必须
提交预注册 manifest，并让未开启的 holdout 满足全部界限。coverage 仍用于操作员
引导和标定板尺寸诊断，但 coverage 不等于位姿精度，在此策略中也不是硬门槛。核心验收改为检测
完整性、PnP、灾难性 scalar 重投影上限、缩放 Jacobian 可观测性、有界条件位姿不确定度、
平面位姿歧义、时间重复性、split-half 稳定性和原生深度 sanity。重投影残差已经进入 covariance 的
像素噪声估计，因此 `uncertainty_validated` 不会再把 legacy 0.5/1.0 px precision 阈值作为 primary
hard gate；当前 1.5/2.0 px gross RMSE/p95 候选门只用于拦截灾难性投影错误。候选 structured
diagnostic 使用空间 holdout 预测、工程幅值下限和确定性整向量 permutation null；最终主
diagnostic 先按物理 corner ID 对重复观测求均值，逐帧 structure 仍只作诊断。scalar magnitude
和条件 covariance 都不能证明 K、D 或 target geometry 正确。
低 coverage 并不保证 PASS；它只是不再因 coverage 数值本身
拒绝一个实际可观的位姿。`legacy_strict` 与 `pose_validated` 的历史语义保持不变。

target preflight 与 provision preflight 回答不同问题。对于 uncertainty policy，前者的
`NUMERICAL_PASS RELEASE_HOLD` 只说明目标检测与位姿可观测，
不保证 raw-stream、fixed-frame 数量/比例、最终重投影、重复性、split-half 或原生深度会通过。
preset 仍为 HOLD 时必须在 `target preflight -> provision preflight` 后停止。只有显式绑定
release manifest hash 的 structured preset 成为 `RELEASED` 后，才能继续；该路径必须由未来
版本新增 authenticated criteria/holdout loader 后实现，当前代码无法开启
`provision fixed -> provision validate`。

合成开发覆盖 500 x 700 mm、
5 x 7、100 mm 方格、75 mm marker、`DICT_4X4_50` 大板。但大板不会自动 PASS：marker 像素
尺度、角点定位、位姿不确定度以及所有物理检查仍必须通过。原始数据流验证是独立前置门槛，
`uncertainty_validated` 不会绕过或修改它。

当前真实 structured-gate 实验只使用已固定的 A4 target；
`REAL_500X700_STRUCTURED_GATE_VALIDATION=DEFERRED`。

`TargetDetector` 是硬件无关的插件契约。ChArUco 实现返回图像点、稳定 point ID、持久化的
目标板 canonical 点和二维质量指标，不估计目标位姿或相机外参。打印和验证细节见
[docs/charuco-target.md](docs/charuco-target.md)。

配置根节点使用单数 `camera`。未知字段、复数 `cameras` 根节点、未知数据流、非法尺寸或帧率以及
非字符串序列号都会被拒绝，不会被静默转换。

## Python API

`camera_rig.api` 是稳定的下游接口。消费者代码不应依赖包内目录结构。

```python
import camera_rig
from camera_rig.api import CameraSession, ReplayCameraSession, load_camera_config

print(camera_rig.__version__)

config = load_camera_config("private-d435i.yaml")
with CameraSession.from_config(config) as camera:
    frame = camera.capture()

with ReplayCameraSession.from_artifact("capture-artifact") as replay:
    restored = replay.capture()
```

无需依赖完整固定相机产物的内部文件布局即可加载：

```python
from camera_rig.api import load_provisioned_camera_bundle

bundle = load_provisioned_camera_bundle("fixed-camera-artifact")
fixed = bundle.fixed_mount_calibration
if fixed is None:
    raise RuntimeError("camera is not fixed-calibrated")
T_workspace_from_camera = fixed.T_parent_from_camera_reference
```

导入 `camera_rig.api` 不需要 RealSense 或 OpenCV。详见[公开 API](docs/public-api.md)、
[稳定性策略](API_STABILITY.md)与[下游集成指南](docs/downstream-integration.md)。

## 产物

`CameraBundle` 是面向一个物理相机的带版本顶层 JSON 契约，可包含设备身份、数据流配置、各流
内参、设备内部光学坐标变换、深度比例、可选的固定安装标定、质量结果和来源信息。固定相机一键配置
始终填充并验证该安装记录。JSON 写入具有确定性和原子性，读取持久化变换时会重新验证。

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
