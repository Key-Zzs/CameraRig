# 固定相机 bootstrap 资格

权威 A4 production bootstrap 必须使用真实 target-metrology 回执。对水平和垂直方向的长基线
做重复测量，并与 resolved target geometry 比较。尺度容差按“下游允许平移误差 / 最大工作
距离”预注册，再扣除测量不确定度。回执保存双轴尺度、各向异性、重复性、仪器、时间、来源和
PASS/FAIL。

必须在获得读数前冻结 acceptance。先用下游平移误差预算与计划最大工作距离运行
`camera-rig target metrology-policy-create` 并保存 policy 文件；之后才可运行
`camera-rig target metrology-create --acceptance-policy <frozen-policy.json>` 输入水平/垂直重复
读数。测量命令不再提供阈值覆盖，因此不能在看到 FAIL 后放宽阈值重跑。

回执会内嵌完整且已验证的 policy 及其 SHA-256，且回执时间必须晚于 policy 时间。v2 bootstrap
provision 同时携带独立 policy 文件与回执，并强制二者内容及哈希一致。

若既有实体测量已知，但原始重复读数、量具身份、分辨率和不确定度没有留存，操作人可以明确
授权人工豁免。使用 `camera-rig target metrology-waiver-create`，并逐字保留授权声明。该命令
生成独立的 `camera-rig.target-metrology-manual-waiver.v1` schema：记录报告尺寸，绑定冻结的
policy 与靶标，把缺失测量字段保留为 null，并将机器门禁标记为 `WAIVED_NOT_EVALUATED`。
报告尺寸与标称值一致时可获得 `PASS`，但这始终是操作人授权的例外，不得表述为机器验证过的
物理计量证据。

Native RealSense depth 是独立 metric source。Bootstrap evaluator 使用 factory depth scale、
depth intrinsics 与内部 stream transforms，对比 PnP predicted target plane 和 measured depth，
报告 support、signed/absolute residual、robust plane offset/normal、distance-scale ratio、
board-local distribution 及逐帧/汇总统计。支持不足、projection 不支持、非有限深度或
plane/scale 不通过都必须 FAIL。

发布的 v2 provision 包含 target-scale policy、metrology、metric-depth 与 bootstrap
qualification 回执。
CameraBundle 权限固定为 `qualification_state=BOOTSTRAP_QUALIFIED`、
`qualification_scope=bootstrap_only`、`production_authoritative=false`；它只是下游 multi-pose
的 initializer，不是 production 多相机标定。

Factory-vs-diagnostic-refit intrinsic health 在多姿态阶段评估，refit 绝不修改 factory K/D。
只有 bounded 参数变化在 untouched holdout 上同时达到绝对、相对和 paired-consistency 改善，
factory model 才标记为 `SUSPECT`；证据不足为 `INSUFFICIENT_EVIDENCE`。

Structured residual 永久为 `diagnostic_only`。通用 planar hard gate 明确为
`NOT_SUPPORTED_DUE_TO_PLANAR_IDENTIFIABILITY_LIMIT`，不能影响资格决策。
