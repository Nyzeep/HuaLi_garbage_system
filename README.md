# HuaLi_garbage_system

> 本项目为中国大学生计算机设计大赛作品，面向智慧社区场景，构建了一套集图像识别、视频分析、目标跟踪、告警留存与统计展示于一体的垃圾与火情智能预警系统。采用 **Python + Rust 混合技术栈**，在保持工程可维护性的同时对核心计算路径进行了性能加速。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Rust-1.75%2B-CE422B?logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Celery-5.4%2B-37814A?logo=celery&logoColor=white" alt="Celery">
  <img src="https://img.shields.io/badge/Redis-5.2%2B-DC382D?logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/ONNX_Runtime-1.20%2B-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/Ultralytics-YOLO-FF9F00" alt="Ultralytics">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white" alt="Pydantic">
  <img src="https://img.shields.io/badge/License-MIT-F7DF1E" alt="MIT">
</p>

---

## 目录

- [项目简介](#项目简介)
- [系统亮点](#系统亮点)
- [技术架构](#技术架构)
- [代码结构](#代码结构)
- [核心模块说明](#核心模块说明)
- [模型与推理策略](#模型与推理策略)
- [Rust 加速层](#rust-加速层)
- [安装与运行](#安装与运行)
- [环境变量](#环境变量)
- [启动方式](#启动方式)
- [常用接口](#常用接口)
- [数据存储](#数据存储)
- [测试](#测试)
- [优缺点分析与优化建议](#优缺点分析与优化建议)
- [许可证](#许可证)

---

## 项目简介

社区垃圾与火情识别预警系统面向智慧社区、园区巡检与安全治理场景，围绕"发现问题、生成预警、留存记录、辅助管理"这一闭环展开设计。系统通过 FastAPI 提供统一 Web 页面与 API 服务，结合 YOLO/ONNX 双后端推理能力，对上传图片、摄像头图像和视频内容进行识别分析，并将预警结果持久化到本地数据库，便于后续查询、统计与展示。

系统不仅支持垃圾检测（垃圾桶、垃圾溢出、散落垃圾）与火情检测（火焰、烟雾），还通过 ResNet18 颜色分类模型对检测到的垃圾桶进行四分类（可回收、有害、厨余、其他），为智慧社区的精细化垃圾分类管理提供数据支撑。

核心计算路径（IoU 计算、目标去重、告警冷却批量处理、坐标反算、感知哈希）由 **Rust** 实现，并通过 **PyO3** 绑定在进程内直接调用。Python 层保留完整回退逻辑，确保 Rust 不可用时系统仍可正常运行。

---

## 系统亮点

- **多入口检测**：支持图片上传、Base64 图像（摄像头抓拍）、视频文件三种检测入口
- **垃圾桶颜色分类**：基于 ResNet18 对检测到的垃圾桶进行可回收/有害/厨余/其他四分类，辅助精细化垃圾分类管理
- **Python + Rust 混合加速**：核心几何计算与告警去重迁移至 Rust，Python 保留完整回退路径
- **PyO3 进程内调用**：Rust 计算层通过 PyO3 扩展与 Python 直接交互，消除进程间通信开销
- **双后端推理**：优先加载 INT8 ONNX，其次普通 ONNX（优先尝试 GPU provider），再回退到 Ultralytics `.pt` 权重
- **IoU + Kalman 目标跟踪**：视频链路在 IoU 匹配基础上叠加 Kalman 预测，提升目标轻微位移或短暂丢失时的 ID 稳定性
- **分组告警冷却**：火情/烟雾（1s）与垃圾/溢出（3s）分组独立冷却；图片/Base64 路径支持本地持久化恢复，视频路径对 `alert_history` 做时间窗口裁剪与容量上限控制
- **自适应视频跳帧与微批处理**：视频处理根据上一关键帧的检测/告警密度动态调整有效跳帧与微批大小，兼顾静态场景吞吐与告警场景响应
- **跨模型去重**：多模型推理结果合并后优先使用 Rust scored NMS 做跨模型去重，减少重复框与重复告警
- **Rust 坐标反算**：ONNX letterbox 后处理优先通过 Rust 做坐标反算，统一热路径中的几何变换逻辑
- **感知哈希跳帧**：关键帧之间通过感知哈希 + 汉明距离评估画面变化幅度，辅助自适应跳帧决策
- **自适应阈值**：不同模型使用分模型阈值，并受统一上下限约束，便于在精度与召回之间平衡
- **异步视频处理**：Celery + Redis 异步任务调度，本地无 Worker 时自动回退到线程执行
- **实时流支持**：SSE 推送视频任务进度、告警列表和统计数据；WebSocket 支持摄像头实时检测
- **统一异常体系**：`AppError` 异常层次结构，API 响应 envelope 格式统一
- **集中输入校验**：上传大小、图片解码、分页边界等在入口集中校验，业务层无需处理脏输入
- **视频编码器自动选择**：优先尝试硬件编码器（NVENC/QSV/AMF），失败自动回退到 `libx264`
- **无模型安全模式**：所有模型不可用时检测接口返回空结果，不生成虚假告警，前端仍可正常访问与联调
- **可视化展示**：内置首页、综合检测页、视频页、预警页、统计页、数据集说明页

---

## 技术架构

### 后端与服务

| 组件 | 用途 | 版本要求 |
|---|---|---|
| FastAPI | Web 框架与 API 组织 | >=0.115.0 |
| Uvicorn | ASGI 服务启动 | >=0.32.0 |
| Pydantic v2 | 请求与响应模型校验 | >=2.9.0 |
| SQLAlchemy 2.0 | SQLite ORM 持久化 | >=2.0.36 |
| Celery + Redis | 视频异步任务调度 | >=5.4.0 / >=5.2.0 |
| pydantic-settings | `.env` 配置管理 | >=2.6.0 |

### 媒体处理

| 组件 | 用途 | 版本要求 |
|---|---|---|
| ONNX Runtime | 高性能模型推理（优先尝试 GPU provider） | >=1.20.0 |
| Ultralytics YOLO | `.pt` 权重加载（回退） | >=8.3.0 |
| OpenCV | 图像解码、绘框、视频帧处理 | >=4.8.0 |
| NumPy | 张量与数组运算 | >=1.24.0 |
| ImageIO + imageio-ffmpeg | 视频编码写出 | >=2.36.0 / >=0.6.0 |
| Pillow | 图像处理基础依赖 | >=10.0.0 |
| PyTorch / TorchVision | Ultralytics 与 ResNet18 运行依赖 | >=2.4.0 / >=0.19.0 |

### Rust 加速层

| 组件 | 用途 |
|---|---|
| `lib.rs` | BBox、scored NMS、批量 IoU 匹配、坐标反算、感知哈希、告警去重核心算法；PyO3 绑定 |
| `main.rs` | Axum REST 服务入口，暴露 `/health` 与 `/v1/*` 接口（备用） |
| `rust_bridge.py` | Python 侧统一桥接层，优先 PyO3，回退 HTTP |

> **性能说明**：基准测试表明，在轻量几何计算与事件去重场景中，PyO3 直连扩展在 batch 模式下显著优于纯 Python。HTTP 微服务虽然具备服务化隔离能力，但会引入序列化与进程间通信成本，当前仅作为兼容与调试备用方案。

### 当前加速策略

- **热路径优先 PyO3**：`non_max_suppression()`、`batch_iou_match()`、`invert_letterbox_bbox()`、`perceptual_hash()`、`dedupe_events()` 优先走 PyO3 进程内调用
- **HTTP 微服务保留兼容性**：`rust_bridge.py` 与 Rust REST 服务继续保留，用于调试或独立进程场景
- **tuple 轻量通道**：PyO3 接口优先接受 `(x1, y1, x2, y2)` 与 `(class_id, bbox, timestamp_ms)` 轻量元组

### 前端展示

- Jinja2 模板渲染
- Tailwind CSS（CDN）
- 原生 JavaScript（上传、SSE、WebSocket、结果渲染）

---

## 代码结构

```text
HuaLi_garbage_system/
├── app/                            # 主应用目录
│   ├── api/                        # 路由层
│   │   ├── pages.py                # 页面路由
│   │   └── routes.py               # API 路由（检测、预警、统计、任务、SSE、WebSocket）
│   ├── core/                       # 应用核心契约层
│   │   ├── exceptions.py           # AppError 异常层次结构
│   │   ├── geometry.py             # 几何计算工具
│   │   ├── responses.py            # 统一响应 envelope
│   │   └── validators.py           # 集中输入校验函数
│   ├── infrastructure/
│   │   └── ml/
│   │       ├── backends.py         # ONNX / Ultralytics 后端封装
│   │       ├── model_registry.py   # 模型注册表（loaded_map）
│   │       └── rust_bridge.py      # Rust 桥接层（PyO3 + HTTP）
│   ├── services/                   # 服务层
│   │   ├── alert_policy_service.py # 告警冷却策略
│   │   ├── bin_color_service.py    # 垃圾桶颜色分类（ResNet18）
│   │   ├── detection_service.py    # 检测主编排（组合各子服务）
│   │   ├── inference_service.py    # 多模型推理
│   │   ├── record_service.py       # 记录写入、查询、统计
│   │   ├── rendering_service.py    # 检测框绘制渲染
│   │   ├── scene_service.py        # 场景状态分析
│   │   └── video_service.py        # 视频逐帧处理与告警冷却
│   ├── upgrade/                    # 升级版时序处理流水线
│   │   ├── alarm.py                # AlarmEngine 连续帧告警
│   │   ├── detection.py            # DetectionEngine 适配器
│   │   ├── pipeline.py             # UpgradePipeline 组合
│   │   └── tracker.py              # TrackEngine IoU 追踪器
│   ├── models/                     # 模型权重文件（运行时加载）
│   ├── templates/                  # Jinja2 前端页面
│   ├── uploads/                    # 上传文件与结果存储（运行时创建）
│   ├── alert_cooldown.py           # 历史兼容层（委托 AlertPolicyService）
│   ├── bootstrap.py                # 启动初始化（建表、创建目录）
│   ├── celery_app.py               # Celery 应用
│   ├── config.py                   # pydantic-settings 配置
│   ├── constants.py                # 类别常量定义（含 BIN_TYPES）
│   ├── database.py                 # 数据库引擎与会话
│   ├── db_models.py                # ORM 模型
│   ├── dependencies.py             # FastAPI 依赖注入（lru_cache 单例）
│   ├── detector.py                 # 历史兼容层（无模型安全行为）
│   ├── main.py                     # FastAPI 入口与异常处理器
│   ├── schemas.py                  # Pydantic 响应 Schema
│   ├── services/inference.py       # 历史兼容层（委托 InferenceService / backends）
│   ├── tasks.py                    # 视频异步任务封装
│   └── utils.py                    # 通用工具函数
├── rust/                           # Rust 加速层
│   ├── Cargo.toml
│   ├── pyproject.toml              # Maturin PyO3 构建配置
│   ├── README.md
│   └── src/
│       ├── lib.rs                  # 核心算法库（BBox、scored NMS、批量 IoU、坐标反算、感知哈希、去重、PyO3 绑定）
│       └── main.rs                 # Rust REST 服务入口（备用）
├── scripts/                        # 辅助脚本
│   └── test_video.py               # 视频处理调试脚本
├── tests/                          # 测试目录（详见 test.md）
├── benchmarks/                     # 基准测试
│   └── rust_call_path_benchmark.py # Rust 调用路径性能对比
├── vendor/                         # 预编译 Rust wheel 本地缓存（可选）
├── logs/                           # 运行时日志（运行时创建）
├── start_queue.bat                 # Windows 一键启动脚本
├── requirements.txt                # Python 依赖
├── pyproject.toml                  # Python 项目配置（Ruff、MyPy、Pytest）
├── .pre-commit-config.yaml         # pre-commit 代码质量钩子（Ruff + MyPy + Cargo fmt/clippy）
├── Optimization.md                 # 变更与优化日志
├── test.md                         # 测试覆盖清单
├── README.md                       # 本文件
└── LICENSE                         # MIT 许可证
```

---

## 核心模块说明

### 契约层（`app/core/`）

- `exceptions.py`：`AppError` 基类及子类（`ValidationError`、`FileTypeError`、`FileParseError`、`ResourceNotFoundError`、`InferenceError`、`TaskDispatchError`、`ServiceUnavailableError`）
- `responses.py`：`success_response()` / `error_response()` 统一 JSON envelope
- `validators.py`：`validate_upload_size()`、`validate_image_bytes()`、`validate_skip_frames()`、`validate_pagination()` 集中校验函数
- `geometry.py`：几何计算工具函数

### 基础设施层（`app/infrastructure/ml/`）

- `backends.py`：INT8 ONNX / 普通 ONNX / Ultralytics 多级后端选择，ONNX 路径优先尝试 GPU provider（CUDA / DirectML / CPU），并在后处理阶段优先走 Rust 坐标反算；支持 IOBinding 加速与批量推理
- `model_registry.py`：`ModelRegistry.loaded_map()` 统一管理多模型实例，支持 `class_mapping` 重映射
- `rust_bridge.py`：`RustBridge` 统一桥接层，优先使用 PyO3 进程内调用，回退到 HTTP 服务

### 服务层（`app/services/`）

- `detection_service.py`：主编排入口，依赖 `InferenceService`、`SceneService`、`AlertPolicyService`、`RenderingService`，通过 `DetectionServiceDeps` 依赖容器注入
- `inference_service.py`：遍历 `model_registry`，执行并行推理、分模型自适应阈值，并对多模型结果做跨模型 scored NMS 去重
- `alert_policy_service.py`：图片/Base64 路径的告警冷却（本地状态文件持久化 + 过期裁剪）
- `scene_service.py`：分析当前帧目标构成，输出场景状态描述（含 `timestamp`）
- `rendering_service.py`：纯 cv2 绘框逻辑，与业务逻辑解耦
- `video_service.py`：视频逐帧处理，支持基于感知哈希 + 汉明距离的自适应跳帧与微批处理；告警冷却优先走 Rust 批量去重，回退纯 Python 路径；视频编码器自动选择并回退
- `bin_color_service.py`：基于 ResNet18 的垃圾桶颜色分类服务，对 `garbage_bin` 检测框裁剪后进行可回收/有害/厨余/其他四分类

### 升级流水线（`app/upgrade/`）

- `TrackEngine`：基于 Rust 批量 IoU 匹配 + Kalman 预测/校正的轻量追踪器，Rust 不可用时自动回退到 Python IoU 匹配；最多丢失 10 帧后轨迹退出
- `AlarmEngine`：连续帧计数器，`min_consecutive_frames` 判断基于真实 track_id，语义准确
- `UpgradePipeline`：以"非破坏性旁路"方式为现有检测结果附加 `track_id` 和时序告警元数据

### 历史兼容层

- `app/detector.py`：旧检测器兼容层，无模型时返回空结果，彻底避免虚假告警
- `app/alert_cooldown.py`：旧冷却逻辑兼容层，内部委托 `AlertPolicyService`
- `app/services/inference.py`：旧推理兼容层，内部委托 `InferenceService` / `backends.py`

---

## 模型与推理策略

模型配置以 `app/config.py` 为准：

| 配置键 | 默认路径 | 用途 |
|---|---|---|
| `garbage_int8_onnx_model` | `app/models/garbage.int8.onnx` | 垃圾类检测（最高优先） |
| `garbage_onnx_model` | `app/models/garbege.onnx` | 垃圾类检测（优先） |
| `garbage_pt_model` | `app/models/garbege.pt` | 垃圾类检测（回退） |
| `fire_int8_onnx_model` | `app/models/only_fire.int8.onnx` | 火情检测（最高优先） |
| `fire_onnx_model` | `app/models/only_fire.onnx` | 火情检测（优先） |
| `fire_pt_model` | `app/models/only_fire.pt` | 火情检测（回退） |
| `smoke_int8_onnx_model` | `app/models/fire_smoke.int8.onnx` | 烟雾检测（最高优先） |
| `smoke_onnx_model` | `app/models/fire_smoke.onnx` | 烟雾检测（优先） |
| `smoke_pt_model` | `app/models/fire_smoke.pt` | 烟雾检测（回退） |
| `bin_color_resnet18_model` | `app/models/bin_color_resnet18.pt` | 垃圾桶颜色分类 |

### 烟雾模型类别映射

若使用的烟雾检测模型同时包含火焰与烟雾两个输出头，可通过以下配置控制类别映射：

| 配置键 | 默认值 | 说明 |
|---|---|---|
| `smoke_model_include_fire` | `true` | 烟雾模型是否包含火焰类别 |
| `smoke_model_fire_class_id` | `1` | 火焰在烟雾模型中的类别索引 |
| `smoke_model_smoke_class_id` | `0` | 烟雾在烟雾模型中的类别索引 |

推理策略：

1. 优先加载对应的 INT8 ONNX 模型（如存在）
2. INT8 ONNX 不可用时加载普通 ONNX，并优先尝试 GPU provider（CUDA → DirectML → CPU）
3. ONNX 不可用时回退到 Ultralytics `.pt` 权重
4. 所有模型均不可用时进入无模型安全模式：检测接口返回空结果，前端页面仍可正常访问与联调

### 模型文件获取

模型文件（`.onnx` / `.pt`）体积较大，未纳入版本控制。获取方式：

1. **自行训练**：使用 Ultralytics YOLOv8 在自定义数据集上训练，导出 `.pt` 和 `.onnx` 格式；颜色分类模型使用 ResNet18 在垃圾桶图像上微调
2. **预训练模型**：联系项目维护者获取预训练权重文件
3. **无模型安全模式**：无模型时系统不生成随机检测结果，但可正常访问前端页面、接口与任务链路

---

## Rust 加速层

### 已实现算法

`rust/src/lib.rs` 中提供以下公开 API：

| 函数 | 说明 |
|---|---|
| `iou(a: BBox, b: BBox) -> f64` | 计算两框 IoU |
| `filter_overlapping_boxes(boxes, threshold)` | 基础重叠框过滤 |
| `non_max_suppression(scored_boxes, threshold)` | scored NMS 去重 |
| `batch_iou_match(left, right, threshold)` | 批量 IoU 匹配 |
| `invert_letterbox_bbox(bbox, transform)` | letterbox 坐标反算 |
| `perceptual_hash(grayscale_pixels, width, height)` | 感知哈希 |
| `hamming_distance(a, b)` | 哈希汉明距离 |
| `dedupe_track_events(events, cooldown_ms, iou_threshold)` | 批量去重告警事件（跨时间窗口） |

### Python 调用方式（PyO3 直连）

Python 通过 `huali_garbage_core` 扩展模块进行进程内调用：

```python
from app.infrastructure.ml.rust_bridge import RustBridge

bridge = RustBridge()

# scored NMS 去重
kept = bridge.non_max_suppression(
    boxes=[{"bbox": [10, 10, 50, 50], "score": 0.93}, ...],
    threshold=0.5,
)

# 批量 IoU 匹配
matches = bridge.batch_iou_match(left=[[0, 0, 10, 10]], right=[[1, 1, 11, 11]], threshold=0.3)

# 感知哈希 + 汉明距离
hash_a = bridge.perceptual_hash(grayscale_pixels=[...], width=64, height=64)
distance = bridge.hamming_distance(hash_a, hash_a)

# 批量告警去重（含历史窗口）
deduped = bridge.dedupe_events(events=[...], cooldown_ms=1000, iou_threshold=0.4)

# 健康检查
status = bridge.health_check()  # {'available': True, 'healthy': True, 'latency_ms': 0.5}
```

### 构建 PyO3 扩展

```bash
pip install maturin
maturin develop --manifest-path rust/Cargo.toml
```

### 预编译 wheel 分发

项目提供 GitHub Actions 工作流 [build-rust-wheel.yml](.github/workflows/build-rust-wheel.yml)，可在打 tag 或手动触发时构建 Windows 预编译 wheel。

- `push tag`：自动构建并上传到对应 Release
- `workflow_dispatch`：可只生成 artifact，或指定已有 tag 并发布到该 Release

> 当前仓库内这套自动分发链路以 Windows 为主；如需 Linux / macOS，可在同一工作流中继续扩展对应平台构建。

构建产物可：

- 上传到 GitHub Releases，供 `start_queue.bat` 自动下载并安装
- 或直接随发行包放入仓库 `vendor/` 目录，供 `start_queue.bat` 优先本地安装

> 若 `huali_garbage_core` 未安装或构建失败，应用会自动回退到纯 Python 实现，不会因 PyO3 缺失而不可用。

### 构建 Rust REST 服务（备用）

```bash
cargo build --release --manifest-path rust/Cargo.toml
```

生成的二进制文件位于 `rust/target/release/huali_garbage_server.exe`（Windows）或 `huali_garbage_server`（Linux/macOS）。

---

## 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/Nyzeep/HuaLi_garbage_system.git
cd HuaLi_garbage_system
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 Rust 加速层（默认无需本地 Rust）

普通用户**不需要安装 Rust 工具链**。

项目启动脚本会按以下顺序自动处理 Rust 加速：

1. 直接使用已安装的 `huali_garbage_core`
2. 从仓库 `vendor/` 目录安装预编译 wheel
3. 从 GitHub Releases 自动下载匹配的 Windows wheel
4. 如果本机已安装 Rust，再尝试本地构建 PyO3 扩展
5. 若以上都不可用，则自动回退到纯 Python 实现

如果你是项目维护者需要手动构建 Rust 层，请参考 [Rust 加速层 § 构建 PyO3 扩展](#构建-pyo3-扩展) 与 [§ 构建 Rust REST 服务](#构建-rust-rest-服务备用)。

### 5. 准备模型文件

将模型文件（`.onnx` 或 `.pt`）放置到 `app/models/` 目录下。模型文件获取方式见 [模型文件获取](#模型文件获取)。

### 6. （维护者）配置代码质量钩子

项目已内置 `.pre-commit-config.yaml`，包含 Ruff、MyPy 以及 Rust 的 `cargo fmt` / `clippy` 检查。首次配置只需执行：

```bash
pre-commit install
```

后续每次 `git commit` 前将自动对变更文件执行 lint 与类型检查，防止风格错误与类型问题进入仓库。

> 如需对已存在的全部文件跑一次检查，可执行 `pre-commit run --all-files`。

---

## 环境变量

项目默认读取根目录 `.env` 文件，无 `.env` 时按以下默认值运行：

```env
# 应用配置
APP_NAME=社区垃圾与火情识别预警系统
APP_VERSION=2.0.0
DEBUG=false

# 数据库与缓存
DATABASE_URL=sqlite:///garbage_system.db
REDIS_URL=redis://localhost:6379/0

# 任务配置
CELERY_TASK_ALWAYS_EAGER=false
VIDEO_DEFAULT_SKIP_FRAMES=1

# 上传限制
MAX_UPLOAD_SIZE_MB=200

# Rust 服务配置（HTTP 备用路径）
RUST_SERVICE_URL=http://127.0.0.1:3000

# 启动脚本自动下载 wheel 的配置
# 默认仓库：Nyzeep/HuaLi_garbage_system
# 默认 tag：latest（即最新 Release）
# 默认文件匹配：huali_garbage_core-*-win_amd64.whl
GITHUB_REPO=Nyzeep/HuaLi_garbage_system
GITHUB_RELEASE_TAG=latest
WHEEL_ASSET_GLOB=huali_garbage_core-*-win_amd64.whl

# ONNX / 推理优化
PREFER_ONNX_GPU=true
ONNX_GPU_DEVICE_ID=0
ADAPTIVE_CONF_FLOOR=0.35
ADAPTIVE_CONF_CEILING=0.7

# 检测阈值（可按模型独立配置）
DEFAULT_CONF_THRESHOLD=0.5
GARBAGE_BIN_CONF_THRESHOLD=0.4
FIRE_CONF_THRESHOLD=0.15
SMOKE_CONF_THRESHOLD=0.30
DEFAULT_IOU_THRESHOLD=0.3

# 烟雾模型类别映射
SMOKE_MODEL_INCLUDE_FIRE=true
SMOKE_MODEL_FIRE_CLASS_ID=1
SMOKE_MODEL_SMOKE_CLASS_ID=0

# 视频自适应跳帧与微批
ADAPTIVE_SKIP_MIN=1
ADAPTIVE_SKIP_MAX=12
VIDEO_MICRO_BATCH_SIZE=4
VIDEO_MICRO_BATCH_SIZE_MAX=16

# Kalman 滤波参数
KALMAN_PROCESS_NOISE=1e-2
KALMAN_MEASUREMENT_NOISE=1e-1
KALMAN_ERROR_COV_POST=1.0

# 垃圾桶颜色分类
BIN_COLOR_MIN_CONFIDENCE=0.4
```

---

## 启动方式

### 方式一：Windows 一键启动（推荐）

项目提供 `start_queue.bat` 作为 Windows 环境下的统一入口。脚本内置虚拟环境管理、依赖自动安装、Rust 加速层智能探测与多模式启动能力。

#### 使用语法

```bat
start_queue.bat [模式]
```

| 模式 | 说明 | 适用场景 |
|---|---|---|
| `all`（默认） | 启动 FastAPI + Celery Worker | 本地完整开发/演示环境，需 Redis |
| `lite` | 仅启动 FastAPI，不依赖 Redis | 普通用户快速体验，视频任务自动回退到本地线程 |
| `web` | 仅启动 FastAPI | 前端开发或 Web 服务单独部署 |
| `worker` | 仅启动 Celery Worker | 分布式 Worker 节点 |
| `setup` | 仅执行环境初始化与依赖安装 | 首次克隆或更新依赖后 |
| `check` | 运行运行时导入检查 | 排查环境或依赖问题 |
| `rust-http` | 构建并启动 Rust HTTP 备用服务 | 调试 HTTP 回退路径 |
| `help` | 显示帮助信息 | — |

#### 典型用法

```bat
:: 普通用户快速体验（无需 Redis）
start_queue.bat lite

:: 完整开发环境（需先启动 Redis）
start_queue.bat all

:: 首次初始化或更新依赖后
start_queue.bat setup
```

#### `start_queue.bat` 内部机制

脚本按以下流水线执行，任一环节失败均会输出日志到 `logs/` 并暂停等待：

1. **虚拟环境探测/创建**：检查 `.venv\Scripts\python.exe`，不存在时依次尝试 `py -m venv` 与 `python -m venv` 创建。
2. **Python 依赖懒安装**：通过快速导入探测判断依赖是否已就绪；仅在缺失时执行 `pip install -r requirements.txt`，并将输出写入 `logs/pip_install.log`。
3. **Rust 加速层智能获取**（四级降级策略）：
   - **Lv.1 已安装**：探测 `huali_garbage_core` 是否已存在于当前虚拟环境。
   - **Lv.2 本地 wheel**：遍历 `vendor/huali_garbage_core-*-win_amd64.whl` 并强制重装。
   - **Lv.3 GitHub Release**：若配置了 `GITHUB_REPO`，调用 `scripts/download_rust_wheel.py` 从对应 Release 下载匹配 wheel 到 `vendor/` 后安装。
   - **Lv.4 本地构建**：探测 `cargo` 与 `rustc`，若存在则安装 `maturin` 并执行 `maturin develop --release`。
   - **Fallback**：以上全部失败时，系统以纯 Python 模式运行，功能无损，仅性能下降。
4. **运行时校验**：执行核心导入探测（FastAPI、Celery、SQLAlchemy、OpenCV 等），确保环境完整。
5. **服务启动**：根据模式启动对应进程，每个进程在独立 `cmd` 窗口中运行，日志实时输出。

> **环境变量控制**：
> - `GITHUB_REPO`：指定 wheel 下载源仓库（默认从 `git remote.origin.url` 自动探测）。
> - `GITHUB_RELEASE_TAG`：指定 Release tag（默认 `latest`）。
> - `WHEEL_ASSET_GLOB`：wheel 文件名匹配模式（默认 `huali_garbage_core-*-win_amd64.whl`）。
> - `UPLOADS_DIR`：覆盖默认上传文件存储路径。

---

### 方式二：手动启动（跨平台 / 生产部署）

适用于 macOS / Linux 或需要精细控制启动参数的场景。

#### 1. 准备环境

```bash
# 创建并激活虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt
```

#### 2. （可选）安装 Rust 加速层

手动安装与脚本自动逻辑等价，优先级从上到下：

```bash
# 方法 A：直接安装已编译好的 wheel（如果有）
pip install vendor/huali_garbage_core-*.whl

# 方法 B：使用脚本自动从 GitHub Release 下载并安装
python scripts/download_rust_wheel.py --repo Nyzeep/HuaLi_garbage_system --tag latest

# 方法 C：本地构建 PyO3 扩展（需 Rust 工具链）
pip install maturin
maturin develop --manifest-path rust/Cargo.toml
```

> 若跳过此步骤，应用将以纯 Python 回退模式运行，不影响功能可用性。

#### 3. 启动服务

```bash
# 启动 Web 服务（开发模式，带热重载）
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 或显式工厂模式（推荐用于生产 ASGI 服务器 / systemd / Docker）
uvicorn app.main:create_asgi_app --factory --host 0.0.0.0 --port 8000
```

> `app.main:app` 为惰性 ASGI 包装器，兼容 `uvicorn` 直接启动；`create_asgi_app --factory` 确保每次 worker 进程都新建干净的应用实例，避免全局状态泄漏。

```bash
# （可选）启动 Celery Worker，处理视频异步任务
python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

> 无 Worker 时，视频任务将自动降级到本地线程执行，不影响前端任务提交与结果查询，但会失去异步队列的削峰能力。

#### 4. （仅维护者）启动 Rust HTTP 备用服务

当需要调试 HTTP 回退路径或在独立进程中运行 Rust 计算层时：

```bash
cargo build --release --manifest-path rust/Cargo.toml
./rust/target/release/huali_garbage_server      # Linux / macOS
# 或
rust\target\release\huali_garbage_server.exe    # Windows
```

默认监听 `http://127.0.0.1:50051`，可通过 `.env` 中的 `RUST_SERVICE_URL` 调整。

---

## 常用接口

### 检测接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/detect/image` | 图片上传检测（form: file） |
| POST | `/api/detect/base64` | Base64 图像检测（json: {image: "data:image/..."}) |
| POST | `/api/detect/video` | 视频任务提交（form: file, skip_frames） |
| GET | `/api/tasks/{task_id}` | 视频任务状态与进度查询 |

### 实时流接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/tasks/{task_id}/stream` | SSE 视频任务进度流 |
| WS | `/api/ws/camera` | WebSocket 摄像头实时检测 |
| GET | `/api/alerts/stream` | SSE 告警列表自动刷新 |
| GET | `/api/statistics/stream` | SSE 统计数据自动刷新 |

### 记录与统计接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/alerts` | 预警记录分页列表 |
| GET | `/api/alerts/{record_uid}/image` | 预警截图获取 |
| GET | `/api/statistics` | 检测量、预警量、类别分布统计 |
| GET | `/api/status` | 服务状态（含 Rust 状态） |
| GET | `/api/classes` | 支持的检测类别列表 |

### 页面路由

| 路径 | 说明 |
|---|---|
| `/` | 首页总览 |
| `/detection` | 综合检测页（图片/摄像头/视频） |
| `/video` | 重定向到 `/detection` 的视频检测入口 |
| `/alerts` | 预警记录页 |
| `/statistics` | 数据统计页 |
| `/dataset` | 数据集说明页 |
| `/docs` | FastAPI 在线接口文档（Swagger UI） |
| `/redoc` | FastAPI 在线接口文档（ReDoc） |

---

## 数据存储

| 数据 | 路径 | 说明 |
|---|---|---|
| SQLite 数据库 | `garbage_system.db` | 预警记录、视频任务状态 |
| 预警截图 | `{RUNTIME_DIR}/uploads/alerts/` | 检测告警截图（默认在项目同级 `HuaLi_garbage_runtime/uploads/` 下） |
| 视频上传与结果 | `{RUNTIME_DIR}/uploads/videos/` | 原始视频与处理后输出（运行时创建） |
| 静态访问前缀 | `/uploads/...` | 通过 Web 访问上传文件 |
| 日志 | `logs/` | 运行时日志（如已配置文件日志） |

> 说明：`uploads_dir` 默认位于项目目录之外的 `HuaLi_garbage_runtime/uploads/`，以避免 `uvicorn --reload` 在写入视频/截图时触发服务器重启。如需自定义，可在 `.env` 中设置 `UPLOADS_DIR`。

---

## 测试

项目包含完整的测试覆盖，详见 [test.md](test.md)。

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_detection_service.py

# 运行带覆盖率报告
pytest --cov=app --cov-report=html
```

### 测试分类

- **单元测试**：输入校验、几何计算、算法核心、常量验证
- **服务层测试**：检测编排、推理服务、告警策略、颜色分类
- **API 测试**：REST 接口、SSE 流、WebSocket
- **集成测试**：视频任务链路、数据库并发、Celery 任务注册
- **兼容性测试**：历史兼容层、运行时路径、模型配置
- **基准测试**：Rust 调用路径性能对比

---

## 优缺点分析与优化建议

### 优点

**1. 清晰的分层架构**
契约层（`core/`）、基础设施层（`infrastructure/`）、服务层（`services/`）职责边界清晰，各层依赖方向单一，便于单独测试和替换。

**2. 务实的 Python + Rust 混合策略**
Rust 仅覆盖核心计算热路径（scored NMS、批量 IoU、坐标反算、感知哈希、批量去重），Python 层保留完整回退逻辑。Rust 不可用时系统仍可正常运行，不强制所有部署都依赖 Rust。

**3. 双推理后端 + 自动回退**
INT8 ONNX → 普通 ONNX（GPU 优先） → Ultralytics `.pt` → 无模型安全模式四级降级，对运行环境的要求弹性较好，且不会因缺少模型而生成虚假告警。

**4. 统一异常与响应体系**
`AppError` 层次结构 + `error_response()` envelope 保证 API 错误响应格式一致，前端处理逻辑简单。

**5. 视频告警冷却语义正确**
按类别分组（火情/烟雾 1s、垃圾/溢出 3s）独立冷却，同一目标在冷却窗口内不重复告警，避免了朴素实现中跨类干扰的问题。

**6. 真实 IoU + Kalman 追踪器**
`TrackEngine` 实现贪心 IoU 匹配叠加 Kalman 预测/校正，跨帧目标 ID 稳定，使 `AlarmEngine` 的连续帧告警判断具有实际意义。

**7. 现代化实时通信**
SSE 替代轮询推送视频进度和统计数据，WebSocket 支持摄像头实时检测，降低延迟与服务器负载。

**8. 垃圾分类精细化能力**
通过 ResNet18 颜色分类模型对检测到的垃圾桶进行四分类，为智慧社区的垃圾分类管理提供数据支撑。

### 缺点

**1. 数据库为 SQLite，并发写入受限**
SQLite 写操作串行，在多 Celery Worker 并发写入预警记录时存在锁竞争，不适合高吞吐生产场景。

**2. 模型文件不在版本控制中**
`.onnx` / `.pt` 权重文件体积大，未进入仓库，新成员克隆后无法直接运行，缺乏明确的模型获取说明。

**3. 告警冷却目前为本地持久化状态，不是分布式共享**
`AlertPolicyService` 现在会将冷却状态持久化到本地状态文件，并在启动时加载与裁剪过期记录；这解决了单进程重启后的状态丢失问题，但多进程/多实例部署下仍不是分布式共享，可能出现跨进程重复告警。

**4. `TrackEngine` 为单帧贪心匹配，精度有限**
当前追踪器虽叠加 Kalman 滤波，但仍不保留更复杂的运动预测信息，目标快速移动或长时间遮挡时 ID 切换率较高，连续帧告警语义准确性受影响。

### 优化建议

#### 短期（工程质量）

| 建议 | 说明 |
|---|---|
| **补充单元测试** | 优先覆盖 `TrackEngine`、`dedupe_track_events`、`validate_*` 系列函数，可用 `pytest` + `hypothesis` 进行属性测试 |
| **完善模型获取说明** | 在 README 或 `app/models/` 下提供模型下载脚本或 DVC 配置，使新成员可一键获取权重 |
| ~~引入 `pre-commit` Hooks~~ ✅ | 已配置 `.pre-commit-config.yaml`（Ruff + MyPy + `cargo fmt/clippy`），执行 `pre-commit install` 即可启用 |
| **告警状态分布式共享** | 进一步将 `AlertPolicyService` 的冷却状态从本地状态文件迁移到 Redis（TTL key），解决多进程/多实例下的重复告警问题 |

#### 中期（性能与可靠性）

| 建议 | 说明 |
|---|---|
| **数据库迁移至 PostgreSQL** | 生产或高并发演示场景下替换 SQLite，解决写锁问题；使用 Alembic 管理 Schema 变更 |
| **视频处理引入帧缓冲队列** | 将解码、推理、编码解耦为三个并发步骤，充分利用多核，降低端到端延迟 |
| **跨模型 NMS 深化** | 在 `InferenceService.detect()` 中进一步利用 `RustBridge.non_max_suppression()` 对多模型重叠检测框进行 scored NMS 去重 |

#### 长期（架构演进）

| 建议 | 说明 |
|---|---|
| **补充 gRPC 接口** | 在保留现有 REST 兼容性的前提下，为高吞吐内部调用增加 gRPC 传输层 |
| **前端框架化** | 将检测页、统计页等交互复杂的页面迁移至 Vue 或 React，Jinja2 仅保留静态展示页 |
| **支持 RTSP/RTMP 实时流** | 将视频处理链路从"文件上传"扩展到"实时流拉取"，适配摄像头直接接入场景 |
| **引入指标监控** | 集成 Prometheus + Grafana，对推理延迟、告警量、队列积压等关键指标进行实时观测 |

---

## 许可证

本项目采用 [MIT License](LICENSE)。

## 支持项目

如果这个项目对你有帮助，欢迎给仓库点亮一个 Star，这对项目的持续完善非常有帮助。
