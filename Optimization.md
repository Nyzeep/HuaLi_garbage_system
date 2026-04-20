# Optimization Log

> 说明：本文件用于记录项目中所有新增功能、删除功能、重构优化、架构调整、性能改进以及重要风险提示。
>
> 记录要求：每次发生变更时，均需追加一条带时间的记录，便于回溯与审计。

---

## 2026-04-21 变更记录

### 1. 架构分析与实施方案输出
- 完成对当前项目架构、功能模块、代码实现与业务需求的系统性阅读与分析。
- 输出两套架构优化方向：
  - Rust + Python 混合技术栈方案
  - 纯 Python 技术栈方案
- 最终根据用户偏好，明确采用“先纯 Python 重构，再局部 Rust 加速”的渐进路线。

### 2. Harness Engineering 方案设计
- 设计了适用于 Vibe Coding 场景的约束模型框架。
- 明确了以下内容：
  - 全局规则
  - 模块规则
  - 任务规则
  - 错误模型
  - 验证流程
  - 回滚机制
- 输出了 `harness/rules.md` 与 `harness/task_contract.md` 的初稿内容与落地思路。

### 3. Subagent 方案讨论与收敛
- 在方案层面完成了多个专业化 subagent 的职责划分建议。
- 后续根据用户反馈，相关内容已收敛到当前主线任务，不再继续展开独立提示词方案。

### 4. 文件结构草案输出
- 提供了目标目录结构草案，包括：
  - `app/core/`
  - `app/domain/`
  - `app/services/`
  - `app/infrastructure/`
  - `harness/`
  - `tests/`
  - `rust/`
- 明确了各层职责边界和迁移顺序。

### 5. 核心代码骨架落地
- 新建并写入以下文件：
  - `app/core/exceptions.py`
  - `app/core/responses.py`
  - `app/core/constants.py`
  - `app/services/rendering_service.py`
  - `app/services/scene_service.py`
  - `app/services/alert_policy_service.py`
  - `app/infrastructure/ml/backends.py`
  - `app/infrastructure/ml/model_registry.py`
  - `app/services/inference_service.py`
  - `app/services/detection_service.py`
  - `app/dependencies.py`
- 完成检测服务的初步拆分：
  - 推理服务
  - 场景分析服务
  - 告警策略服务
  - 渲染服务
  - 模型注册表
- 通过依赖组装方式保留了旧接口兼容性。

### 6. Rust 核心层初始化
- 新建 Rust 工程骨架：
  - `rust/Cargo.toml`
  - `rust/src/lib.rs`
  - `rust/src/main.rs`
- 在 Rust 侧优先实现了核心几何与去重算法：
  - `BBox`
  - `iou`
  - `filter_overlapping_boxes`
  - `TrackEvent`
  - `dedupe_track_events`
- 新增 Rust 命令行 JSON 桥接模式，用于 Python 子进程调用 Rust 的高频计算能力。
- 新建 Python Rust 桥接层：
  - `app/infrastructure/ml/rust_bridge.py`
- 将 `VideoProcessingService` 中的 IoU 计算改造为可优先调用 Rust 的桥接接口，并保留 Python 回退实现。

### 7. 质量检查
- 对已新增和修改的核心文件执行了 lint 检查。
- 检查结果：无 lint 错误。

### 8. 后续待办（已全部完成，见下方 2026-04-21 第二次变更记录）

---

## 2026-04-21 第二次变更记录

### 1. Rust 核心层去重与代码结构优化
- **变更类型**：重构、消除重复代码
- **变更内容**：重写 `rust/src/main.rs`，移除其中对 `BBox`、`iou`、`filter_overlapping_boxes`、`dedupe_track_events` 的重复实现；改为通过 `use huali_garbage_core::*` 直接引用 `lib.rs` 中的公开接口。
- **影响范围**：`rust/src/main.rs`
- **风险说明**：无破坏性变更，Rust JSON-RPC 行为不变。
- **验证结果**：`cargo build` 编译通过（目标二进制逻辑不变）。

### 2. RustBridge 高层类型化 API
- **变更类型**：功能增强
- **变更内容**：在 `app/infrastructure/ml/rust_bridge.py` 中新增两个类型化封装方法：
  - `filter_boxes(boxes, threshold)` — 调用 Rust `filter_overlapping_boxes`，返回过滤后的 bbox 列表。
  - `dedupe_events(events, cooldown_ms, iou_threshold)` — 调用 Rust `dedupe_track_events`，批量去重告警事件。
- **影响范围**：`app/infrastructure/ml/rust_bridge.py`
- **风险说明**：仅新增方法，不影响现有 `call()` 接口。

### 3. 视频告警冷却迁移至 Rust 批量去重
- **变更类型**：性能优化、架构迁移
- **变更内容**：重构 `app/services/video_service.py` 中的 `_apply_video_alert_cooldown`：
  - 原实现：每帧 O(n × history) Python 循环调用 IoU。
  - 新实现：分火/烟（1s 冷却）和垃圾/溢出（3s 冷却）两个分组，分别调用 `rust_bridge.dedupe_events()` 批量处理；Rust 不可用时自动回退 Python 路径（`_apply_video_alert_cooldown_python`，原逻辑保留）。
- **影响范围**：`app/services/video_service.py`
- **风险说明**：Rust 可用路径与 Python 路径均经过独立测试；分组冷却语义与原逻辑完全一致。

### 4. 修复 tasks.py 中 DetectionService 构造错误
- **变更类型**：Bug 修复
- **变更内容**：`tasks.py` 中 `run_video_task` 使用 `DetectionService(settings=settings)` 实例化，与重构后的 `DetectionService(deps: DetectionServiceDeps)` 构造器不匹配，导致运行时崩溃。修复方式：改为调用 `get_detection_service()` 工厂函数（与 FastAPI 依赖注入路径保持一致）。
- **影响范围**：`app/tasks.py`
- **风险说明**：`get_detection_service()` 使用 `@lru_cache`，Celery worker 进程中模型只加载一次，无性能问题。

### 5. 实现真实 IoU 匹配追踪器
- **变更类型**：功能完善、架构升级
- **变更内容**：重写 `app/upgrade/tracker.py`：
  - 原实现：每帧为所有检测框分配新的自增 ID（无实际追踪逻辑）。
  - 新实现：贪心 IoU 匹配追踪器：每帧检测框与活跃轨迹按 class_id 分组进行最优 IoU 匹配（阈值 0.3）；匹配成功的轨迹保留 ID 并更新 bbox/置信度；未匹配的检测框新建轨迹；连续 10 帧未匹配的轨迹自动退出。
- **影响范围**：`app/upgrade/tracker.py`
- **风险说明**：`TrackEngine.update()` 接口签名不变，上层 `UpgradePipeline` 无需修改。`AlarmEngine` 的 `_seen_count` 现在是真实的跨帧轨迹 ID，`min_consecutive_frames` 判断语义更准确。

### 6. 后续待办
- 编译 Rust release 二进制（`cargo build --release`），确保 `rust/target/release/huali_garbage_core.exe` 存在以激活 Rust 路径。
- 考虑将 `filter_boxes` 应用于 `InferenceService.detect()` 做跨模型 NMS（当多个模型检测到同一对象时去重）。
- 持续在本文件追加变更记录。

---

## 2026-04-21 第三次变更记录

### 1. 视频检测慢根因分析
- **变更类型**：分析记录
- **根因**：`RustBridge.call()` 原实现每次调用都通过 `subprocess.run()` 重新启动一个 Rust 进程，Windows 上进程冷启动耗时 50–200ms，每帧都承担这笔开销。这使得"Rust 加速"实际上比纯 Python 更慢。
- **次要根因**：两个推理模型（garbage、fire）串行执行，浪费了可并行的 GPU/CPU 时间；`available()` 每帧都做一次文件系统 stat。

### 2. Rust 改为持久化子进程（JSONL 流式通信）
- **变更类型**：性能优化、架构改造
- **影响范围**：`rust/src/main.rs`、`app/infrastructure/ml/rust_bridge.py`
- **变更内容**：
  - `rust/src/main.rs`：改为 JSONL 流式循环模式——从 stdin 逐行读取 JSON 请求，逐行写回响应，直到 stdin 关闭。替换了原来"读全部 stdin → 处理 → 退出"的单次调用模式。
  - `rust_bridge.py`：重构为持久化子进程模式：
    - 新增 `_process`（`subprocess.Popen`）和 `_lock`（`threading.Lock`）字段
    - `call()` 通过 `_ensure_process()` 懒启动并复用同一个进程，写入一行 JSON、读取一行响应
    - 进程异常退出时自动重启（下次 `call()` 触发）
    - 新增 `close()` 和 `__del__` 确保进程在服务结束时被清理
  - `available()`：结果首次检查后缓存，消除每帧文件系统 stat 开销
- **性能预期**：视频告警冷却的 Rust 路径从每帧 ~100ms 降至 ~1–5ms（消除进程启动开销）
- **风险说明**：Python fallback 路径不变；`_lock` 保证多线程安全；进程死亡时自动重启，不影响系统可用性。

### 3. 新增 Rust 健康检查功能
- **变更类型**：功能新增、可观测性提升
- **影响范围**：`app/infrastructure/ml/rust_bridge.py`、`app/dependencies.py`、`app/api/routes.py`、`app/schemas.py`
- **变更内容**：
  - `rust_bridge.py`：新增 `health_check()` 方法，执行已知答案测试（Box A [0,0,100,100] 与 Box B [50,50,150,150] 的 IoU，期望值 ≈ 0.142857），返回 `{available, healthy, error, latency_ms}` 字典
  - `dependencies.py`：新增 `get_rust_bridge()`（`@lru_cache` 单例），供 FastAPI 依赖注入
  - `schemas.py`：新增 `RustStatus` Pydantic 模型；`SystemStatusResponse` 新增 `rust: RustStatus` 字段
  - `routes.py`：`GET /api/status` 注入 `RustBridge` 依赖，响应中加入 `rust` 字段，前端/运维可直接判断 Rust 是否正常工作
- **验证方式**：`GET /api/status` 返回 `rust.available`、`rust.healthy`、`rust.latency_ms`

### 4. 多模型并行推理
- **变更类型**：性能优化
- **影响范围**：`app/services/inference_service.py`
- **变更内容**：`InferenceService.detect()` 原实现串行遍历所有已加载模型。改为当模型数 ≥ 2 时，使用 `ThreadPoolExecutor` 并行执行各模型的 `predict()`，结果合并后返回。
- **性能预期**：两模型并行时推理耗时从 T₁+T₂ 降至 max(T₁, T₂)，约节省 30–60ms/帧（视模型大小与硬件而定）
- **风险说明**：ONNX Runtime 和 Ultralytics 各自维护独立 session，线程间无共享状态，并发安全。

### 5. 后续待办
- 执行 `cargo build --release` 编译新版 Rust 二进制（JSONL 流式模式），激活持久化子进程优化路径。
- 考虑将视频编码从 CPU libx264 改为 GPU 硬件编码器（NVENC / QuickSync），进一步降低帧处理延迟。
- 对 `TrackEngine` 引入匈牙利算法替代贪心 IoU 匹配，提升多目标场景准确性。



---

## 2026-04-22 变更记录

### 1. 视频任务进度：轮询 → SSE
- **变更类型**：性能优化、架构改进
- **变更内容**：
  - 新增后端 SSE 端点 `GET /api/tasks/{task_id}/stream`：服务端每 500ms 查询一次任务状态，仅在状态/进度/消息发生变化时才推送事件；任务完成或失败时自动关闭流，完成事件附带 `result_video` 与 `stats` 字段。
  - `video.html`：移除 `pollInterval`（`setInterval` 1.5s 轮询），改为 `watchTaskStatus()` 函数用 `EventSource` 订阅 SSE；`stopDetection()` / `finalizeDetectionUi()` 关闭 EventSource 而非 clearInterval。
  - `detection.html`（视频子功能）：同上，用 `watchVideoTaskStatus()` + `activeVideoEventSource` 替换原 `while` 循环轮询。
- **影响范围**：`app/api/routes.py`、`app/templates/video.html`、`app/templates/detection.html`
- **风险说明**：SSE 连接在任务完成/失败时由服务端主动关闭；用户手动停止时前端 `close()` 断开；后台任务执行不受影响。

### 2. 摄像头实时检测：HTTP 轮询 → WebSocket
- **变更类型**：性能优化、架构改进
- **变更内容**：
  - 新增后端 WebSocket 端点 `WS /api/ws/camera`：接收客户端 JSON（含 base64 图像），调用 `DetectionService.detect()`，返回检测结果 JSON。
  - `detection.html`：新增 `cameraWs` 变量；`startAutoDetect()` 改为建立 WebSocket 连接，`ws.onopen` 后启动 1s 定时帧捕获；`captureAndDetect()` 通过 `ws.send()` 发送帧；`stopAutoDetect()` 关闭 WebSocket；`ws.onmessage` 收到结果后直接更新 UI。
  - 移除 `detectBase64()` 函数中对 `POST /api/detect/base64` 的调用（WebSocket 路径已替代）。
- **影响范围**：`app/api/routes.py`、`app/templates/detection.html`
- **风险说明**：WebSocket 连接失败时 `onerror` 回调自动调用 `stopAutoDetect()`，UI 回归待机状态；`POST /api/detect/base64` 端点保留，供单帧手动检测与非 WebSocket 场景使用。

### 3. 告警记录与统计数据：轮询 → SSE
- **变更类型**：架构改进
- **变更内容**：
  - 新增后端 SSE 端点 `GET /api/alerts/stream`（支持 `page`/`per_page`/`status` 参数）：每 10s 推送一次当前分页数据。
  - 新增后端 SSE 端点 `GET /api/statistics/stream`：每 10s 推送一次统计数据。
  - `alerts.html`：`autoRefreshInterval` 改为 `autoRefreshEs`（`EventSource`）；`changeFilter()` 参数变更时关闭旧 SSE 并以新参数重新建立连接。
  - `statistics.html`：`autoRefreshInterval` 改为 `autoRefreshEs`（`EventSource`）。
- **影响范围**：`app/api/routes.py`、`app/templates/alerts.html`、`app/templates/statistics.html`
- **风险说明**：SSE 连接出错时 `onerror` 回调关闭连接，停止自动刷新；现有 `GET /api/alerts` 和 `GET /api/statistics` REST 端点保留。

---

## 2026-04-22 第二次变更记录

### 1. 视频编码器自动选择与兼容回退
- **变更类型**：性能优化、兼容性增强
- **变更内容**：重构 `app/services/video_service.py` 的输出写入逻辑，新增视频编码器候选池自动选择机制：优先尝试 `libx264` 之外的硬件/快速编码器候选（`h264_nvenc`、`h264_qsv`、`h264_amf`），在不可用或写入失败时自动回退到 `libx264`；若编码失败导致 ffmpeg 生成临时输出路径，服务会自动切换到回退编码器并尽量恢复输出文件。
- **影响范围**：`app/services/video_service.py`
- **风险说明**：在不支持硬件编码器的机器上仍可回退到原有 `libx264` 路径，兼容性保持不变；极少数编码中途失败场景下会重新写入输出文件，耗时略有增加但不影响任务可用性。
- **验证结果**：已完成代码检查，`app/services/video_service.py` 通过 lint。

### 4. 视频处理关键帧 + 跟踪分离优化
- **变更类型**：性能优化、架构改进
- **变更内容**：重构 `app/services/video_service.py` 的视频处理主循环：
  - 将原本按 `skip_frames` 直接跳过的粗粒度逻辑，调整为“关键帧检测 + 中间帧复用”的策略。
  - 新增 `_is_keyframe()` 判定，关键帧仍走完整的 `detect_raw()` → 冷却去重 → 升级追踪 → 画框流程。
  - 非关键帧复用上一关键帧的渲染结果写入视频，避免重复推理和重复后处理。
  - 保留告警冷却、升级追踪与输出编码回退逻辑，不改变原有业务语义。
- **影响范围**：`app/services/video_service.py`
- **风险说明**：关键帧间隔增大时，目标快速变化场景可能存在轻微位置滞后；但检测精度本身不降低，且在非关键帧上会保持最近一次有效结果。
- **验证结果**：代码已完成重构，后续通过 lint 与实际视频样例验证输出一致性。

---

## 2026-04-22 第三次变更记录

### 1. Rust 桥接切换为独立 REST 微服务
- **变更类型**：架构改造、性能优化
- **变更内容**：将 Rust 侧入口改为 Axum REST 服务，暴露 `/health`、`/v1/iou`、`/v1/filter-boxes`、`/v1/dedupe-events`；Python 侧 `RustBridge` 改为基于 `urllib.request` 的 HTTP bridge，不再依赖本地子进程 stdin/stdout 协议。
- **影响范围**：`rust/src/main.rs`、`app/infrastructure/ml/rust_bridge.py`、`app/config.py`、`app/dependencies.py`
- **风险说明**：部署时需要额外启动 Rust 服务；服务不可用时将回退到 Python 路径，功能保持可用但性能下降。
- **验证结果**：已完成代码切换，待通过自动化测试与手工启动链路进一步验证。

### 2. 视频任务服务接入统一 RustBridge 配置
- **变更类型**：重构、配置收敛
- **变更内容**：`VideoProcessingService` 支持注入 `RustBridge`，并在 `app/dependencies.py` 中新增 `get_video_processing_service()`，让 Web 与 Celery 侧统一复用配置化的 Rust 服务地址与超时设置。
- **影响范围**：`app/services/video_service.py`、`app/dependencies.py`、`app/tasks.py`
- **风险说明**：统一单例后，调用配置更一致；如果 `.env` 中 `RUST_SERVICE_URL` 配置错误，视频任务和状态检测都会同时受影响。
- **验证结果**：代码已完成修改，待通过任务链路测试验证。

### 3. 测试与启动脚本同步 REST 微服务模型
- **变更类型**：测试修复、运维改进
- **变更内容**：重写 `tests/test_rust_bridge.py`，改为验证 HTTP 路由、错误处理、请求体映射与线程锁行为；更新 `tests/test_video_pipeline_integration.py`，改为通过 mock `dedupe_events()` 触发 Python fallback；更新 `start_queue.bat`，在 Windows 一键启动中加入 Rust REST 服务窗口，并向 Web/Celery 进程显式注入 `RUST_SERVICE_URL`。
- **影响范围**：`tests/test_rust_bridge.py`、`tests/test_video_pipeline_integration.py`、`start_queue.bat`
- **风险说明**：`start_queue.bat` 现在要求先存在 release 版 Rust 可执行文件，否则会直接提示并退出。
- **验证结果**：测试文件与启动脚本已同步，待执行 pytest 进行最终确认。

### 4. PyO3 tuple 通道与 batch 性能验证
- **变更类型**：性能优化、接口收敛
- **变更内容**：为 Rust 核心库补充 PyO3 绑定，并将 Python ↔ Rust 的数据边界收敛为轻量 tuple 通道；`bbox` 采用 `(x1, y1, x2, y2)`，`event` 采用 `(class_id, bbox, timestamp_ms)`，减少对象属性访问与字典序列化开销。
- **影响范围**：`rust/src/lib.rs`、`benchmarks/rust_call_path_benchmark.py`、`README.md`
- **验证方式**：使用 `benchmarks/rust_call_path_benchmark.py` 分别在 single 与 batch 模式下对 pure Python、PyO3、Rust HTTP 路径进行对比。
- **结果摘要**：
  - single 模式：`filter_boxes` 在 PyO3 路径与 pure Python 接近，`dedupe_events` 仍受边界转换成本影响，提升有限；
  - batch 模式（`--batch-size 32 --iterations 2000`）：PyO3 对 `filter_boxes` 提升最明显，`0.1081 ms -> 0.0391 ms`，约 2.7x；`dedupe_events` 由 `0.0697 ms -> 0.0553 ms`，约 1.26x；
  - 进一步迭代后，`filter_boxes` 保持稳定优势（约 `0.0408 ms`），`dedupe_events` 变化不大（约 `0.0570 ms`），说明该路径更受事件解析与边界转换成本影响。
- **风险说明**：PyO3 是热路径主方案，HTTP 微服务仍保留为兼容/调试备用；`dedupe_events` 的收益已接近当前数据规模下的上限，后续优化可能边际递减。
- **结论**：Rust 加速层的优先级已明确为 `PyO3 > 纯 Python > HTTP 微服务`（按热路径性能优先级），其中 `filter_boxes` 是最值得保留和继续优化的高频路径。
