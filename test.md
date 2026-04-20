# 测试覆盖清单

> 本文档用于汇总当前项目的测试覆盖范围，方便快速了解各模块的回归保护情况、已验证的关键能力以及后续可继续补强的方向。

## 1. 测试概览

当前 `tests/` 目录已覆盖项目的主要核心链路，测试类型包括：

- 单元测试
- 服务层测试
- API 接口测试
- SSE / WebSocket 交互测试
- 视频任务与处理链路集成测试
- 数据库统计与并发场景测试
- Rust 桥接层测试
- 属性测试（Hypothesis）

### 当前测试文件列表

- `tests/test_validators.py`
- `tests/test_alert_policy.py`
- `tests/test_scene_service.py`
- `tests/test_inference_service.py`
- `tests/test_rendering_service.py`
- `tests/test_detection_service.py`
- `tests/test_api_detection.py`
- `tests/test_stream_interfaces.py`
- `tests/test_video_cooldown.py`
- `tests/test_tracker.py`
- `tests/test_alarm.py`
- `tests/test_detection_engine.py`
- `tests/test_video_pipeline_integration.py`
- `tests/test_video_task_integration.py`
- `tests/test_rust_bridge.py`
- `tests/test_record_service.py`
- `tests/test_statistics_service.py`
- `tests/test_concurrency.py`
- `tests/__init__.py`

---

## 2. 按模块的覆盖情况

### 2.1 输入校验与边界控制

#### `tests/test_validators.py`
覆盖内容：
- 上传文件大小校验 `validate_upload_size()`
- 跳帧参数校验 `validate_skip_frames()`
- 分页参数校验 `validate_pagination()`
- 边界值行为验证
- Hypothesis 属性测试，验证随机输入下的稳定性

测试重点：
- 上限 / 下限边界
- 非法值自动修正
- 极端输入不崩溃

---

### 2.2 告警冷却策略

#### `tests/test_alert_policy.py`
覆盖内容：
- 图片 / Base64 场景下的告警冷却逻辑
- 同一目标在冷却窗口内重复触发的抑制行为
- 不同类别 / 不同位置目标的独立性
- 内存历史缓存的更新与清理

测试重点：
- 同类重复告警是否被抑制
- 冷却窗口到期后是否恢复触发
- 历史记录是否正确裁剪

---

### 2.3 场景分析

#### `tests/test_scene_service.py`
覆盖内容：
- 空检测结果时的 `normal` 场景
- 火情、烟雾、溢出、一般告警的优先级判断
- `alert_count` / `normal_count` / `total` 统计
- `alert_types` 去重
- `timestamp` 字段存在性与格式

测试重点：
- 场景分类优先级
- 统计字段准确性
- 输出结构完整性

---

### 2.4 推理服务

#### `tests/test_inference_service.py`
覆盖内容：
- 无可用模型时返回空结果
- 单模型推理结果映射正确
- 多模型并行推理结果合并正确
- 忽略未加载后端
- `class_mapping` 重映射
- `source_model` 标识保留

测试重点：
- 推理返回字段完整
- 多模型合并逻辑
- 后端加载状态过滤

---

### 2.5 检测服务编排

#### `tests/test_detection_service.py`
覆盖内容：
- `models_loaded` 透传注册表状态
- `detect()` 调用推理并应用冷却
- `detect_raw()` 跳过冷却
- `draw_boxes()` 委托渲染服务
- `analyze_scene()` 委托场景服务
- `build_response()` 的图像与非图像分支

测试重点：
- 服务间协作关系
- 编排层返回结构
- 冷却逻辑调用顺序

---

### 2.6 渲染服务

#### `tests/test_rendering_service.py`
覆盖内容：
- 返回的是原图副本，不修改输入图像
- 框与标签背景被正确绘制
- 告警框样式生效
- 已知类别使用内置英文标签
- 未知类别使用自定义 `class_name`

测试重点：
- 图像副作用控制
- 颜色与标注绘制
- 标签兜底行为

---

### 2.7 API 层检测接口

#### `tests/test_api_detection.py`
覆盖内容：
- `POST /api/detect/image`
- `POST /api/detect/base64`
- `POST /api/detect/video`
- `GET /api/tasks/{task_id}`

测试重点：
- 图片 / Base64 检测接口返回结构正确
- 预警记录写入调用正确
- 视频任务提交成功
- 视频任务状态查询正确

---

### 2.8 SSE / WebSocket 接口

#### `tests/test_stream_interfaces.py`
覆盖内容：
- `GET /api/tasks/{task_id}/stream` 视频任务 SSE
- `GET /api/alerts/stream` 告警列表 SSE
- `GET /api/statistics/stream` 统计 SSE
- `GET /api/ws/camera` WebSocket 摄像头检测

测试重点：
- SSE 首次推送格式与字段完整性
- 任务完成状态下的 `result_video` / `stats` 输出
- WebSocket 图像检测返回结构
- 非法图片输入的错误返回

---

### 2.9 视频冷却与去重

#### `tests/test_video_cooldown.py`
覆盖内容：
- IoU 计算的对称性与范围正确性
- IoU 自身一致性
- 视频冷却逻辑的 Python 参考实现
- 火情 / 烟雾 1 秒冷却
- 垃圾 / 溢出 3 秒冷却
- 相同目标与不同位置目标的区分
- 历史缓存过期清理

测试重点：
- 视频场景的告警抑制规则
- 类别分组冷却策略
- 历史窗口维护

---

### 2.10 追踪器

#### `tests/test_tracker.py`
覆盖内容：
- Track 目标 ID 分配
- 同帧 / 跨帧目标保持
- 遮挡丢失与恢复
- 跟踪队列清理
- IoU 匹配逻辑

测试重点：
- ID 稳定性
- 丢帧恢复能力
- 追踪器状态迁移

---

### 2.11 连续帧告警引擎

#### `tests/test_alarm.py`
覆盖内容：
- 连续帧达到阈值后触发告警
- 触发后持续输出告警
- 多目标独立性
- 无目标时不触发
- `min_consecutive_frames=1` 的即时触发行为

测试重点：
- 连续帧语义
- 多目标互不干扰
- 阈值边界

---

### 2.12 检测适配器

#### `tests/test_detection_engine.py`
覆盖内容：
- 原始检测结果适配为 `Detection` 对象
- 缺省字段兜底
- `infer()` 调用 detector 并适配输出

测试重点：
- 模型输出结构兼容性
- 适配层稳定性

---

### 2.13 Rust 桥接层

#### `tests/test_rust_bridge.py`
覆盖内容：
- Rust 二进制可用性检测
- 健康检查逻辑
- JSONL 协议通信
- Rust 输出成功 / 失败处理
- `filter_boxes()` 输出解析
- `dedupe_events()` 输出解析
- 子进程异常处理与关闭逻辑
- 并发调用锁保护

测试重点：
- 跨进程通信稳定性
- 失败回退能力
- 线程安全

---

### 2.14 记录服务

#### `tests/test_record_service.py`
覆盖内容：
- 视频任务 `upsert_video_task()` 的创建与更新
- 视频任务 `update_video_task()` 的状态修改
- `get_video_task()` 的查询行为
- `list_alerts()` 的分页 / 状态过滤 / 倒序返回
- `list_classes()` 的类别与图标元数据

测试重点：
- 数据写入与更新正确性
- 查询与过滤行为
- 业务字典返回结构

---

### 2.15 统计服务

#### `tests/test_statistics_service.py`
覆盖内容：
- `build_statistics()` 的总检测数、总告警数、今日告警数统计
- 按小时统计 `hourly_alerts`
- 按类别统计 `class_stats`
- 类别统计按数量降序排序

测试重点：
- 统计口径一致性
- 类别分布准确性
- 排序规则稳定性

---

### 2.16 并发场景

#### `tests/test_concurrency.py`
覆盖内容：
- 多线程并发更新同一视频任务记录
- `VideoTaskRecord` 的状态与消息最终可落库
- 并发执行下无异常抛出

测试重点：
- 并发写入稳定性
- SQLite / Session 使用方式的健壮性

---

### 2.17 视频任务调度层

#### `tests/test_video_task_integration.py`
覆盖内容：
- `run_video_task()` 成功路径
- `run_video_task()` 失败路径
- 进度更新写库
- 输入文件删除策略
- `process_video_task()` Celery 任务封装
- `update_state(PROGRESS)` 调用

测试重点：
- 视频任务生命周期
- 任务状态与进度同步
- Celery 包装行为

---

### 2.18 视频处理链路集成

#### `tests/test_video_pipeline_integration.py`
覆盖内容：
- `VideoProcessingService.process_video()` 端到端处理
- 跳帧策略下的视频处理行为
- 进度回调机制
- 输出视频文件生成
- `UpgradePipeline` 的追踪与告警元数据附加

测试重点：
- 视频逐帧主链路
- 输出结果正确性
- 升级流水线的附加语义

---

## 3. 当前测试覆盖总结

### 已覆盖较好的部分
- 输入校验
- 检测编排
- 图像渲染
- 场景分析
- 视频冷却策略
- 追踪器
- 连续帧告警
- Rust 桥接
- API 检测接口
- SSE / WebSocket 交互
- 视频任务调度与处理
- 记录与统计服务
- 并发写入场景

### 仍可继续增强的部分
- 更复杂的真实视频样本回归测试
- 更完整的 WebSocket 多轮交互测试
- `AlertPolicyService` 的更多边界场景
- `RecordService` 在更高并发和更大数据量下的表现测试
- 真实模型推理的轻量集成测试

---

## 4. Rust 调用路径基准测试

#### `benchmarks/rust_call_path_benchmark.py`
新增内容：
- 纯 Python 路径基线
- 长驻 Rust HTTP 服务路径
- PyO3 扩展模块路径（可选）
- 单帧 / 批量场景下的延迟与吞吐统计
- 结果 JSON 输出

测试重点：
- 比较不同 Rust 调用路径的平均延迟、P50、P95、吞吐量
- 为 PyO3 / 多子进程方案提供量化依据

> 该脚本当前是基准测试入口，不属于自动化单元测试，但已纳入本文档，便于统一查看 Rust 调用路径的性能评估手段。

---

## 5. 建议的后续补测方向

如果继续增强测试覆盖，建议优先补：

1. 真实视频样本的最小端到端回归测试
2. WebSocket 多轮消息交互测试
3. `AlertPolicyService` 复杂边界场景测试
4. `RecordService` 大数据量分页与统计测试
5. 模型加载失败与回退路径的集成测试
6. 基准测试结果固化与趋势对比（例如记录到 CI 产物）

---

## 6. 结论

当前项目测试已经覆盖从底层算法、服务编排、数据持久化、统计分析，到 API、SSE / WebSocket、视频任务链路的多个层次，具备较完整的回归保护能力。对于以演示和研究验证为主的系统来说，这样的测试覆盖已经相当完整；后续若继续强化真实视频回归、高并发场景测试，以及 Rust 调用路径的量化基准，整体稳定性与性能评估能力还可以进一步提升。
