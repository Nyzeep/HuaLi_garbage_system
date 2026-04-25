# 测试覆盖清单

> 本文档用于汇总当前项目的测试覆盖范围，方便快速了解各模块的回归保护情况、已验证的关键能力以及后续可继续补强的方向。
>
> 当前 `tests/` 目录共包含 **30 个测试文件**，覆盖从底层算法、服务编排、数据持久化、统计分析，到 API、SSE / WebSocket、视频任务链路、配置验证、兼容层、Celery 任务注册等多个层次。

---

## 1. 测试概览

当前测试类型包括：

- **单元测试**：独立函数和类的行为验证
- **服务层测试**：业务逻辑编排和依赖协作
- **API 接口测试**：HTTP 端点请求响应验证
- **实时流测试**：SSE / WebSocket 交互验证
- **视频任务与处理链路集成测试**：端到端流程验证
- **数据库统计与并发场景测试**：数据一致性和并发安全
- **Rust 桥接层测试**：跨语言调用和回退机制
- **属性测试（Hypothesis）**：随机输入下的稳定性验证
- **配置与常量测试**：配置项默认值、路径解析、常量语义验证
- **兼容层测试**：历史兼容层的委托行为与无模型安全行为
- **Celery 任务注册测试**：异步任务注册与发现验证

### 1.1 测试文件清单

| 测试文件 | 测试目标 | 测试类型 |
|---|---|---|
| `tests/test_validators.py` | 输入校验函数 | 单元测试 + 属性测试 |
| `tests/test_alert_policy.py` | 告警冷却策略服务 | 单元测试 |
| `tests/test_scene_service.py` | 场景分析服务 | 单元测试 |
| `tests/test_inference_service.py` | 模型推理服务 | 单元测试 + Mock |
| `tests/test_rendering_service.py` | 图像渲染服务 | 单元测试 |
| `tests/test_detection_service.py` | 检测编排服务 | 单元测试 + Mock |
| `tests/test_api_detection.py` | 检测 API 端点 | API 测试 |
| `tests/test_stream_interfaces.py` | SSE / WebSocket 接口 | 实时流测试 |
| `tests/test_video_cooldown.py` | 视频告警冷却逻辑 | 单元测试 |
| `tests/test_tracker.py` | IoU 目标追踪器 | 单元测试 |
| `tests/test_alarm.py` | 连续帧告警引擎 | 单元测试 |
| `tests/test_detection_engine.py` | 检测适配器 | 单元测试 |
| `tests/test_video_pipeline_integration.py` | 视频处理流水线 | 集成测试 |
| `tests/test_video_task_integration.py` | 视频任务调度 | 集成测试 |
| `tests/test_rust_bridge.py` | Rust 桥接层 | 单元测试 + Mock |
| `tests/test_record_service.py` | 记录持久化服务 | 单元测试 + 数据库 |
| `tests/test_statistics_service.py` | 统计计算服务 | 单元测试 + 数据库 |
| `tests/test_concurrency.py` | 并发写入场景 | 并发测试 |
| `tests/test_detector.py` | 旧 detector 兼容层与无模型安全行为 | 兼容性测试 |
| `tests/test_legacy_compat.py` | 旧 cooldown / inference 兼容层委托行为 | 兼容性测试 |
| `tests/test_video_service_alert_history.py` | 视频告警历史裁剪与容量上限 | 单元测试 |
| `tests/test_backends_postprocess.py` | ONNX 后处理与坐标反算 | 单元测试 |
| `tests/test_celery_task_registration.py` | Celery 任务注册与发现 | 集成测试 |
| `tests/test_config_model_paths.py` | 配置项模型路径与遗留兼容 | 单元测试 |
| `tests/test_constants_bin_types.py` | 垃圾分类常量语义 | 单元测试 |
| `tests/test_onnx_iobinding.py` | ONNX Runtime IOBinding 加速路径 | 单元测试 + Mock |
| `tests/test_runtime_paths.py` | 运行时目录路径解析 | 单元测试 |
| `tests/test_rust_geometry_utils.py` | Rust 坐标反算与批量 IoU 匹配 | 单元测试 + Mock |
| `tests/test_rust_nms_and_phash.py` | Rust scored NMS 与感知哈希 | 单元测试 + Mock |
| `tests/test_video_micro_batching.py` | 视频微批大小自适应调整 | 单元测试 |

---

## 2. 按模块的覆盖情况

### 2.1 输入校验与边界控制

#### `tests/test_validators.py`

**覆盖内容：**

| 函数 | 验证点 |
|---|---|
| `validate_upload_size()` | 文件大小上限校验、边界值处理 |
| `validate_skip_frames()` | 跳帧参数范围校验、非法值修正 |
| `validate_pagination()` | 分页参数边界、最大值限制 |
| `validate_image_bytes()` | 图像解码验证、格式检查 |

**测试重点：**

- 上限 / 下限边界行为
- 非法值自动修正逻辑
- 极端输入下的系统稳定性
- Hypothesis 属性测试验证随机输入

**关键用例：**

```python
# 示例：validate_upload_size 边界测试
assert validate_upload_size(b"x" * (200 * 1024 * 1024), max_mb=200)  # 通过
assert_raises(ValidationError, validate_upload_size, b"x" * (201 * 1024 * 1024), max_mb=200)
```

---

### 2.2 告警冷却策略

#### `tests/test_alert_policy.py`

**覆盖内容：**

- 图片 / Base64 场景下的告警冷却逻辑
- 同一目标在冷却窗口内重复触发的抑制行为
- 不同类别 / 不同位置目标的独立性
- 内存历史缓存的更新与清理机制
- 本地状态文件持久化与重启恢复

**测试重点：**

- 同类重复告警是否被正确抑制
- 冷却窗口到期后是否恢复触发
- 历史记录的自动裁剪策略
- 状态文件加载与过期清理

**场景覆盖：**

| 场景 | 预期行为 |
|---|---|
| 同一位置同类目标在冷却期内 | 抑制告警 |
| 同一位置同类目标冷却期后 | 允许告警 |
| 不同位置同类目标 | 独立冷却，互不影响 |
| 同一位置不同类别目标 | 独立冷却，互不影响 |

---

### 2.3 场景分析

#### `tests/test_scene_service.py`

**覆盖内容：**

- 空检测结果时的 `normal` 场景判定
- 火情、烟雾、溢出、一般告警的优先级判断
- `alert_count` / `normal_count` / `total` 统计准确性
- `alert_types` 去重逻辑
- `timestamp` 字段存在性与格式验证

**场景优先级：**

```
fire > smoke > overflow > general_alert > normal
```

**测试重点：**

- 场景分类优先级正确性
- 统计字段数值准确性
- 输出结构完整性验证

---

### 2.4 推理服务

#### `tests/test_inference_service.py`

**覆盖内容：**

- 无可用模型时返回空结果（无随机演示检测，避免虚假告警）
- 单模型推理结果映射正确性
- 多模型并行推理结果合并逻辑
- 多模型结果跨模型去重逻辑
- 分模型自适应置信度阈值
- 忽略未加载后端的过滤机制
- `class_mapping` 重映射功能
- `source_model` 标识保留

**测试重点：**

- 推理返回字段完整性
- 多模型合并逻辑正确性
- 后端加载状态过滤准确性

**Mock 策略：**

- 使用 Mock 模型替代真实 ONNX/Ultralytics 模型
- 验证服务层逻辑独立于实际推理引擎

---

### 2.5 检测服务编排

#### `tests/test_detection_service.py`

**覆盖内容：**

- `models_loaded` 透传注册表状态
- `detect()` 调用推理并应用冷却策略
- `detect_raw()` 跳过冷却直接检测
- `draw_boxes()` 委托渲染服务
- `analyze_scene()` 委托场景服务
- `build_response()` 的图像与非图像分支处理

**测试重点：**

- 服务间协作关系正确性
- 编排层返回结构一致性
- 冷却逻辑调用顺序验证

---

### 2.6 渲染服务

#### `tests/test_rendering_service.py`

**覆盖内容：**

- 返回原图副本，不修改输入图像（无副作用）
- 检测框与标签背景绘制正确性
- 告警框样式（颜色、线宽）生效
- 已知类别使用内置英文标签
- 未知类别使用自定义 `class_name` 兜底

**测试重点：**

- 图像副作用控制（输入不可变）
- 颜色与标注绘制准确性
- 标签兜底行为验证

---

### 2.7 API 层检测接口

#### `tests/test_api_detection.py`

**覆盖端点：**

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/detect/image` | POST | 图片上传检测 |
| `/api/detect/base64` | POST | Base64 图像检测 |
| `/api/detect/video` | POST | 视频任务提交 |
| `/api/tasks/{task_id}` | GET | 视频任务状态查询 |

**测试重点：**

- 图片 / Base64 检测接口返回结构正确性
- 预警记录写入调用验证
- 视频任务提交成功响应
- 视频任务状态查询准确性

---

### 2.8 SSE / WebSocket 接口

#### `tests/test_stream_interfaces.py`

**覆盖端点：**

| 端点 | 类型 | 说明 |
|---|---|---|
| `/api/tasks/{task_id}/stream` | SSE | 视频任务进度流 |
| `/api/alerts/stream` | SSE | 告警列表自动刷新 |
| `/api/statistics/stream` | SSE | 统计数据自动刷新 |
| `/api/ws/camera` | WebSocket | 摄像头实时检测 |

**测试重点：**

- SSE 首次推送格式与字段完整性
- 任务完成状态下的 `result_video` / `stats` 输出
- WebSocket 图像检测返回结构
- 非法图片输入的错误返回

**验证要点：**

- SSE 事件格式符合规范（`data:` / `event:` 前缀）
- WebSocket 连接生命周期管理
- 异常输入的优雅处理

---

### 2.9 视频冷却与去重

#### `tests/test_video_cooldown.py`

**覆盖内容：**

- IoU 计算的对称性与范围正确性（0.0 - 1.0）
- IoU 自身一致性（相同框 IoU = 1.0）
- 视频冷却逻辑的 Python 参考实现
- 火情 / 烟雾 1 秒冷却策略
- 垃圾 / 溢出 3 秒冷却策略
- 相同目标与不同位置目标的区分
- 历史缓存过期清理机制

**测试重点：**

- 视频场景的告警抑制规则
- 类别分组冷却策略正确性
- 历史窗口维护机制

**冷却分组：**

| 类别 | 冷却时间 |
|---|---|
| fire, smoke | 1000 ms |
| garbage, overflow | 3000 ms |

---

### 2.10 追踪器

#### `tests/test_tracker.py`

**覆盖内容：**

- Track 目标 ID 分配逻辑
- 同帧 / 跨帧目标 ID 保持
- 遮挡丢失与恢复机制
- 跟踪队列清理策略
- IoU 匹配逻辑（阈值 0.3）
- Kalman 滤波器初始化与预测/校正路径
- Rust 批量 IoU 匹配回退到 Python 路径

**测试重点：**

- ID 稳定性（同一目标跨帧 ID 不变）
- Kalman 追踪器初始化与预测/校正路径
- 丢帧恢复能力（最多 10 帧）
- 追踪器状态迁移正确性

**状态机：**

```
active -> lost (连续未匹配) -> removed (超过阈值)
```

---

### 2.11 连续帧告警引擎

#### `tests/test_alarm.py`

**覆盖内容：**

- 连续帧达到阈值后触发告警
- 触发后持续输出告警状态
- 多目标独立性（互不干扰）
- 无目标时不触发
- `min_consecutive_frames=1` 的即时触发行为

**测试重点：**

- 连续帧语义正确性
- 多目标互不干扰验证
- 阈值边界测试

---

### 2.12 检测适配器

#### `tests/test_detection_engine.py`

**覆盖内容：**

- 原始检测结果适配为 `Detection` 对象
- 缺省字段兜底策略
- `infer()` 调用 detector 并适配输出

**测试重点：**

- 模型输出结构兼容性
- 适配层稳定性

---

### 2.13 Rust 桥接层

#### `tests/test_rust_bridge.py`

**覆盖内容：**

- Rust HTTP 服务可用性检测
- 健康检查逻辑（已知答案测试）
- HTTP 请求/响应处理
- `filter_boxes()` 输出解析
- `dedupe_events()` 输出解析
- 服务不可用时的 Python 回退
- 并发调用安全性

**测试重点：**

- HTTP 通信稳定性
- 失败回退能力
- 线程安全验证

**Mock 策略：**

- Mock HTTP 服务响应
- 测试服务不可用时的回退路径

---

### 2.14 Rust 几何工具

#### `tests/test_rust_geometry_utils.py`

**覆盖内容：**

- `invert_letterbox_bbox()` 坐标反算正确性
- `batch_iou_match()` 批量 IoU 匹配行为
- 边界参数传递与结果解析

**测试重点：**

- 坐标反算数学正确性
- 批量匹配结果格式一致性

**Mock 策略：**

- 使用 DummyRustBridge 模拟 Rust 几何接口，验证 Python 侧调用逻辑与参数传递

---

### 2.15 Rust NMS 与感知哈希

#### `tests/test_rust_nms_and_phash.py`

**覆盖内容：**

- `non_max_suppression()` scored NMS 去重行为
- `perceptual_hash()` 感知哈希输出一致性
- `hamming_distance()` 汉明距离计算正确性

**测试重点：**

- NMS 在高重叠场景下的置信度优先保留逻辑
- 感知哈希对微小像素变化的敏感度

**Mock 策略：**

- 使用 DummyRustBridge 模拟 NMS 与哈希接口，验证上层调用逻辑

---

### 2.16 ONNX 后端后处理

#### `tests/test_backends_postprocess.py`

**覆盖内容：**

- ONNX 模型输出后处理中的坐标映射正确性
- letterbox 预处理后的坐标反算回原始图像空间
- 置信度阈值过滤行为

**测试重点：**

- 后处理输出的 `bbox` 坐标是否准确映射到原始图像空间
- 置信度过滤边界行为

---

### 2.17 ONNX IOBinding 加速路径

#### `tests/test_onnx_iobinding.py`

**覆盖内容：**

- `OnnxYoloBackend` 在支持 IOBinding 时优先使用 `run_with_iobinding`
- IOBinding 失败时自动回退到 `session.run()`
- 固定批量模型在 `predict_batch()` 中逐帧回退处理

**测试重点：**

- IOBinding 成功路径与回退路径的正确切换
- 批量推理在固定输入形状模型上的兼容性处理

**Mock 策略：**

- Mock `onnxruntime.InferenceSession` 的 `io_binding()`、`run_with_iobinding()`、`run()` 方法

---

### 2.18 记录服务

#### `tests/test_record_service.py`

**覆盖内容：**

- 视频任务 `upsert_video_task()` 的创建与更新
- 视频任务 `update_video_task()` 的状态修改
- `get_video_task()` 的查询行为
- `list_alerts()` 的分页 / 状态过滤 / 倒序返回
- `list_classes()` 的类别与图标元数据

**测试重点：**

- 数据写入与更新正确性
- 查询与过滤行为准确性
- 业务字典返回结构完整性

---

### 2.19 统计服务

#### `tests/test_statistics_service.py`

**覆盖内容：**

- `build_statistics()` 的总检测数、总告警数、今日告警数统计
- 按小时统计 `hourly_alerts`
- 按类别统计 `class_stats`
- 类别统计按数量降序排序

**测试重点：**

- 统计口径一致性
- 类别分布准确性
- 排序规则稳定性

---

### 2.20 并发场景

#### `tests/test_concurrency.py`

**覆盖内容：**

- 多线程并发更新同一视频任务记录
- `VideoTaskRecord` 的状态与消息最终可落库
- 并发执行下无异常抛出

**测试重点：**

- 并发写入稳定性
- SQLite / Session 使用方式的健壮性

**测试方法：**

```python
# 多线程并发更新
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(update_task, task_id) for _ in range(100)]
```

---

### 2.21 视频任务调度层

#### `tests/test_video_task_integration.py`

**覆盖内容：**

- `run_video_task()` 成功路径
- `run_video_task()` 失败路径
- 进度更新写库
- 输入文件删除策略
- `process_video_task()` Celery 任务封装
- `update_state(PROGRESS)` 调用

**测试重点：**

- 视频任务生命周期完整性
- 任务状态与进度同步
- Celery 包装行为验证

---

### 2.22 视频处理链路集成

#### `tests/test_video_pipeline_integration.py`

**覆盖内容：**

- `VideoProcessingService.process_video()` 端到端处理
- 跳帧策略下的视频处理行为
- 进度回调机制
- 输出视频文件生成
- `UpgradePipeline` 的追踪与告警元数据附加

**测试重点：**

- 视频逐帧主链路完整性
- 输出结果正确性
- 升级流水线的附加语义

---

### 2.23 视频告警历史管理

#### `tests/test_video_service_alert_history.py`

**覆盖内容：**

- `alert_history` 时间窗口裁剪机制
- 每类告警历史容量上限控制
- 长视频处理下历史列表无界增长防护

**测试重点：**

- 历史记录过期清理的时效性
- 容量上限触发的丢弃策略

---

### 2.24 视频微批处理

#### `tests/test_video_micro_batching.py`

**覆盖内容：**

- `_adaptive_micro_batch_size()` 根据检测/告警密度动态调整微批大小
- 静态场景（无检测、无告警）下扩大微批以提升吞吐
- 告警场景下缩小微批以降低延迟

**测试重点：**

- 微批大小切换边界正确性
- 自适应策略的灵敏度

---

### 2.25 配置与模型路径

#### `tests/test_config_model_paths.py`

**覆盖内容：**

- `Settings.garbage_int8_onnx_model` 默认路径拼写正确性（`garbage.int8.onnx`）
- 遗留拼写 `garbege.int8.onnx` 的自动回退机制

**测试重点：**

- 配置默认值准确性
- 遗留兼容路径的透明回退

---

### 2.26 运行时路径

#### `tests/test_runtime_paths.py`

**覆盖内容：**

- `uploads_dir` 默认指向 `RUNTIME_DIR / "uploads"`
- 运行时目录与仓库目录的隔离性

**测试重点：**

- 路径解析一致性
- 开发与生产环境路径隔离

---

### 2.27 垃圾分类常量

#### `tests/test_constants_bin_types.py`

**覆盖内容：**

- `BIN_TYPES` 中各类别语义正确性
- 可回收/有害/厨余/其他垃圾桶故意共享同一 `class_id`（0）的设计意图
- 溢出告警独立 `class_id`（1）

**测试重点：**

- 常量语义与设计意图一致性
- UI 展示类别与检测类别的映射正确性

---

### 2.28 Celery 任务注册

#### `tests/test_celery_task_registration.py`

**覆盖内容：**

- `process_video_task` 是否在 Celery 应用自动发现后正确注册
- 任务名称完整路径验证

**测试重点：**

- 异步任务注册完整性
- 自动发现机制有效性

---

### 2.29 历史兼容层

#### `tests/test_detector.py`

**覆盖内容：**

- 旧 `detector.py` 兼容层在无模型时的安全行为
- 彻底移除 `_fake_detect()` 后不再生成随机检测结果
- 返回空结果避免虚假告警污染数据库

**测试重点：**

- 无模型安全模式语义正确性
- 兼容层委托行为稳定性

#### `tests/test_legacy_compat.py`

**覆盖内容：**

- 旧 `alert_cooldown.py` 委托 `AlertPolicyService` 的行为
- 旧 `inference.py` 委托 `InferenceService` / `backends.py` 的行为
- `DeprecationWarning` 正确发出

**测试重点：**

- 兼容层委托链路的完整性
- 废弃警告提示有效性

---

## 3. 测试运行指南

### 3.1 运行所有测试

```bash
# 基本运行
pytest

# 详细输出
pytest -v

# 带覆盖率报告
pytest --cov=app --cov-report=html --cov-report=term

# 并行运行（需安装 pytest-xdist）
pytest -n auto
```

### 3.2 运行特定测试

```bash
# 运行单个测试文件
pytest tests/test_detection_service.py

# 运行单个测试类
pytest tests/test_validators.py::TestValidators

# 运行单个测试方法
pytest tests/test_validators.py::TestValidators::test_validate_upload_size

# 按关键字匹配
pytest -k "test_alert"
```

### 3.3 调试测试

```bash
# 失败时进入 PDB
pytest --pdb

# 显示完整差异
pytest -vv

# 仅运行上次失败的测试
pytest --lf

# 显示最慢的测试
pytest --durations=10
```

### 3.4 环境要求

- Python 3.10+
- 安装 `requirements.txt` 中的所有依赖
- （可选）Redis 服务（用于 Celery 相关测试）
- （可选）Rust 工具链（用于 Rust 桥接层测试）

---

## 4. 当前测试覆盖总结

### 4.1 已覆盖较好的部分

| 模块 | 覆盖程度 | 说明 |
|---|---|---|
| 输入校验 | ⭐⭐⭐⭐⭐ | 边界值、属性测试完整 |
| 检测编排 | ⭐⭐⭐⭐⭐ | 服务协作、Mock 覆盖充分 |
| 图像渲染 | ⭐⭐⭐⭐⭐ | 副作用控制、绘制验证完整 |
| 场景分析 | ⭐⭐⭐⭐⭐ | 优先级、统计验证完整 |
| 视频冷却策略 | ⭐⭐⭐⭐⭐ | 分组冷却、历史维护完整 |
| 追踪器 | ⭐⭐⭐⭐⭐ | ID 稳定性、Kalman 路径、恢复能力完整 |
| 连续帧告警 | ⭐⭐⭐⭐⭐ | 阈值、多目标验证完整 |
| Rust 桥接 | ⭐⭐⭐⭐☆ | HTTP 路径覆盖，PyO3 需补充 |
| Rust 几何/NMS/哈希 | ⭐⭐⭐⭐☆ | Mock 覆盖上层调用，真实 Rust 路径需补充 |
| API 检测接口 | ⭐⭐⭐⭐⭐ | 端点、响应结构完整 |
| SSE / WebSocket | ⭐⭐⭐⭐☆ | 基础交互覆盖，复杂场景可补充 |
| 视频任务调度 | ⭐⭐⭐⭐⭐ | 生命周期、状态同步完整 |
| 记录与统计服务 | ⭐⭐⭐⭐⭐ | CRUD、统计口径完整 |
| 并发写入 | ⭐⭐⭐⭐☆ | 基础并发覆盖，高并发可补充 |
| ONNX 后处理/IOBinding | ⭐⭐⭐⭐⭐ | 坐标反算、加速路径、回退逻辑完整 |
| 配置与常量 | ⭐⭐⭐⭐⭐ | 路径解析、遗留兼容、常量语义完整 |
| 兼容层 | ⭐⭐⭐⭐⭐ | 无模型安全、委托行为完整 |
| Celery 任务注册 | ⭐⭐⭐⭐⭐ | 任务发现与注册验证完整 |
| 视频微批处理 | ⭐⭐⭐⭐⭐ | 自适应策略边界验证完整 |
| 视频告警历史 | ⭐⭐⭐⭐⭐ | 时间窗口裁剪、容量上限验证完整 |

### 4.2 仍可继续增强的部分

| 方向 | 优先级 | 说明 |
|---|---|---|
| 真实视频样本回归测试 | 中 | 使用真实视频文件验证端到端流程 |
| WebSocket 多轮交互测试 | 中 | 验证长时间连接稳定性 |
| `AlertPolicyService` 复杂边界 | 低 | 极端时间窗口、大量历史记录 |
| `RecordService` 大数据量测试 | 低 | 百万级记录分页与统计性能 |
| 真实模型推理集成测试 | 低 | 轻量级模型验证推理链路 |
| PyO3 直接调用测试 | 中 | 补充 PyO3 路径的专项测试 |
| ResNet18 颜色分类测试 | 中 | 补充 bin_color_service 的单元测试与 Mock 测试 |
| 视频编码器自动选择测试 | 低 | 验证硬件编码器回退逻辑 |

---

## 5. Rust 调用路径基准测试

### 5.1 基准测试文件

#### `benchmarks/rust_call_path_benchmark.py`

**测试内容：**

- 纯 Python 路径基线
- Rust HTTP 服务路径
- PyO3 扩展模块路径
- 单帧 / 批量场景下的延迟与吞吐统计
- 结果 JSON 输出

**测试指标：**

| 指标 | 说明 |
|---|---|
| 平均延迟 | 单次调用耗时 |
| P50 | 50 分位延迟 |
| P95 | 95 分位延迟 |
| 吞吐量 | 每秒处理次数 |

**运行方式：**

```bash
# 运行基准测试
python benchmarks/rust_call_path_benchmark.py

# 指定批量大小和迭代次数
python benchmarks/rust_call_path_benchmark.py --batch-size 32 --iterations 2000
```

**预期结果（参考）：**

| 路径 | 单帧延迟 | 批量延迟（32） | 推荐场景 |
|---|---|---|---|
| 纯 Python | ~0.1 ms | ~3.2 ms | 无 Rust 环境 |
| PyO3 | ~0.04 ms | ~1.3 ms | 高频热路径（推荐） |
| HTTP | ~5 ms | ~150 ms | 独立进程调试 |

> 注：HTTP 路径因序列化和网络开销，仅适合低频或调试场景。

---

## 6. 测试设计原则

### 6.1 单元测试原则

1. **独立性**：每个测试用例独立运行，不依赖其他测试
2. **可重复性**：相同输入始终产生相同输出
3. **快速执行**：单元测试应在毫秒级完成
4. **单一职责**：每个测试只验证一个概念

### 6.2 Mock 使用原则

1. **外部依赖 Mock**：数据库、HTTP 服务、文件系统
2. **复杂计算 Mock**：模型推理、图像处理
3. **不 Mock 被测对象**：被测单元内部逻辑应真实执行

### 6.3 集成测试原则

1. **端到端验证**：覆盖完整用户场景
2. **真实依赖**：使用真实数据库、Redis（测试实例）
3. **环境隔离**：测试数据不污染生产环境

### 6.4 兼容性测试原则

1. **委托验证**：确保兼容层正确委托到新实现
2. **行为不变性**：兼容层不应改变原有行为的语义
3. **废弃提示**：兼容层应发出适当的 `DeprecationWarning`

---

## 7. 建议的后续补测方向

如果继续增强测试覆盖，建议优先补充：

### 7.1 高优先级

1. **PyO3 路径专项测试**
   - 验证 `huali_garbage_core` 模块加载
   - 测试 PyO3 接口的输入输出一致性
   - 验证与 HTTP 路径的结果一致性

2. **真实视频回归测试**
   - 准备小型测试视频文件
   - 验证端到端处理流程
   - 检查输出视频质量

3. **ResNet18 颜色分类测试**
   - 验证 `BinColorService` 的模型加载与预测路径
   - 验证分类标签映射正确性
   - Mock 测试置信度过滤逻辑

### 7.2 中优先级

4. **WebSocket 压力测试**
   - 长时间连接稳定性
   - 高频消息处理能力
   - 并发连接支持

5. **模型加载失败场景**
   - ONNX 加载失败回退到 PT
   - 所有模型加载失败的演示模式
   - 模型热切换

6. **视频编码器自动选择测试**
   - 模拟硬件编码器不可用场景
   - 验证 `libx264` 回退路径

### 7.3 低优先级

7. **前端 E2E 测试**
   - 页面加载与交互
   - 文件上传流程
   - 实时检测结果展示

8. **性能回归测试**
   - 基准测试结果固化
   - CI 中自动性能对比
   - 性能下降告警

---

## 8. 结论

当前项目测试已经覆盖从底层算法、服务编排、数据持久化、统计分析，到 API、SSE / WebSocket、视频任务链路、配置验证、兼容层、Celery 任务注册的多个层次，具备较完整的回归保护能力。

### 质量评估

| 维度 | 评分 | 说明 |
|---|---|---|
| 完整性 | 92/100 | 核心链路覆盖充分，新增 ONXX 后处理、配置验证、常量语义、微批处理、告警历史等均已覆盖；部分边界场景可补充 |
| 准确性 | 95/100 | 测试断言准确，Mock 使用合理 |
| 可维护性 | 87/100 | 测试结构清晰，部分复杂测试可拆分 |
| 执行速度 | 80/100 | 大部分测试快速，集成测试可优化 |
| 文档化 | 92/100 | 测试意图明确，注释充分，test.md 与代码结构同步 |

**总体评价**：对于以演示和研究验证为主的系统来说，当前测试覆盖已经相当完整；后续若继续强化真实视频回归、PyO3 路径测试、ResNet18 颜色分类测试，以及高并发场景测试，整体稳定性与性能评估能力还可以进一步提升。
