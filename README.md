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
- [优缺点分析与优化建议](#优缺点分析与优化建议)
- [许可证](#许可证)

---

## 项目简介

社区垃圾与火情识别预警系统面向智慧社区、园区巡检与安全治理场景，围绕"发现问题、生成预警、留存记录、辅助管理"这一闭环展开设计。系统通过 FastAPI 提供统一 Web 页面与 API 服务，结合 YOLO/ONNX 双后端推理能力，对上传图片、摄像头图像和视频内容进行识别分析，并将预警结果持久化到本地数据库，便于后续查询、统计与展示。

核心计算路径（IoU 计算、目标去重、告警冷却批量处理）由 **Rust** 实现，并通过独立 REST 微服务对外提供能力。Python 侧通过 HTTP bridge 调用 Rust 服务，在保留纯 Python 回退路径的前提下提升高频帧处理吞吐量。

---

## 系统亮点

- **多入口检测**：支持图片上传、Base64 图像（摄像头抓拍）、视频文件三种检测入口。
- **Python + Rust 混合加速**：核心几何计算与告警去重迁移至 Rust，Python 保留完整回退路径。
- **Rust 独立微服务**：Rust 计算层以独立 REST 服务运行，Python Web/Celery 通过网络调用，解耦部署与进程生命周期。
- **双后端推理**：优先使用 ONNX Runtime，自动回退到 Ultralytics `.pt` 权重，兼顾速度与兼容性。
- **IoU 目标跟踪**：视频链路集成贪心 IoU 追踪器，跨帧保持目标 ID，`AlarmEngine` 连续帧告警语义更准确。
- **分组告警冷却**：火情/烟雾（1s）与垃圾/溢出（3s）分组独立冷却，避免同一目标重复告警。
- **异步视频处理**：Celery + Redis 异步任务调度，本地无 Worker 时自动回退到线程执行。
- **统一异常体系**：`AppError` 异常层次结构，API 响应 envelope 格式统一。
- **集中输入校验**：上传大小、图片解码、分页边界等在入口集中校验，业务层无需处理脏输入。
- **可视化展示**：内置首页、综合检测页、视频页、预警页、统计页、数据集说明页。

---

## 技术架构

### 后端与服务

| 组件 | 用途 |
|---|---|
| FastAPI | Web 框架与 API 组织 |
| Uvicorn | ASGI 服务启动 |
| Pydantic v2 | 请求与响应模型校验 |
| SQLAlchemy 2.0 | SQLite ORM 持久化 |
| Celery + Redis | 视频异步任务调度 |
| pydantic-settings | `.env` 配置管理 |

### 媒体处理

| 组件 | 用途 |
|---|---|
| ONNX Runtime | 高性能模型推理（首选） |
| Ultralytics YOLO | `.pt` 权重加载（回退） |
| OpenCV | 图像解码、绘框、视频帧处理 |
| NumPy | 张量与数组运算 |
| ImageIO + imageio-ffmpeg | 视频编码写出 |
| Pillow | 图像处理基础依赖 |
| PyTorch / TorchVision | Ultralytics 运行依赖 |

### Rust 加速层

| 组件 | 用途 |
|---|---|
| `lib.rs` | BBox、IoU、NMS 过滤、告警去重核心算法；同时提供 PyO3 绑定 |
| `main.rs` | Axum REST 服务入口，暴露 `/health` 与 `/v1/*` 接口 |
| `rust_bridge.py` | Python 侧 HTTP bridge，含 `filter_boxes()`、`dedupe_events()` |

> 基准测试表明，在本项目的轻量几何计算与事件去重场景中，PyO3 直连扩展在 batch 模式下显著优于纯 Python，尤其在 `filter_boxes` 等高频热路径上收益明显；相比之下，Rust HTTP 微服务虽然具备服务化隔离能力，但会引入显著的序列化与进程间通信成本，不适合作为低延迟调用路径。因此，当前推荐将 Rust 能力以 PyO3 方式嵌入 Python 调用链，HTTP 微服务仅作为兼容与调试备用方案。

### 当前加速策略

- **热路径优先 PyO3**：`filter_boxes()`、`dedupe_events()` 优先走 Rust 直连扩展，减少 Python ↔ Rust 之间的对象转换与网络开销。
- **HTTP 微服务保留兼容性**：`rust_bridge.py` 与 Rust REST 服务继续保留，用于调试、隔离部署或需要独立进程时使用。
- **tuple 轻量通道**：PyO3 接口优先接受 `(x1, y1, x2, y2)` 与 `(class_id, bbox, timestamp_ms)` 这样的轻量元组输入，以降低绑定层开销。

### 前端展示

- Jinja2 模板渲染
- Tailwind CSS（CDN）
- 原生 JavaScript（上传、轮询、结果渲染）

---

## 代码结构

```text
HuaLi_garbage_system/
├── app/
│   ├── api/                        # 路由层
│   │   ├── pages.py                # 页面路由
│   │   └── routes.py               # API 路由（检测、预警、统计、任务）
│   ├── core/                       # 应用核心契约层
│   │   ├── constants.py            # 常量 re-export（兼容导入路径）
│   │   ├── exceptions.py           # AppError 异常层次结构
│   │   ├── responses.py            # 统一响应 envelope
│   │   └── validators.py           # 集中输入校验函数
│   ├── infrastructure/
│   │   └── ml/
│   │       ├── backends.py         # ONNX / Ultralytics 后端封装
│   │       ├── model_registry.py   # 模型注册表（loaded_map）
│   │       └── rust_bridge.py      # Rust REST 服务 HTTP bridge
│   ├── services/                   # 服务层
│   │   ├── alert_policy_service.py # 告警冷却策略
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
│   ├── models/                     # 模型权重文件
│   ├── templates/                  # Jinja2 前端页面
│   ├── uploads/                    # 上传文件与结果存储
│   ├── bootstrap.py                # 启动初始化（建表、创建目录）
│   ├── celery_app.py               # Celery 应用
│   ├── config.py                   # pydantic-settings 配置
│   ├── constants.py                # 类别常量定义
│   ├── database.py                 # 数据库引擎与会话
│   ├── db_models.py                # ORM 模型
│   ├── dependencies.py             # FastAPI 依赖注入（lru_cache 单例）
│   ├── main.py                     # FastAPI 入口与异常处理器
│   ├── schemas.py                  # Pydantic 响应 Schema
│   └── tasks.py                    # 视频异步任务封装
├── rust/                           # Rust 加速层
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                  # 核心算法库（BBox、IoU、NMS、去重）
│       └── main.rs                 # Rust REST 服务入口
├── start_queue.bat                 # Windows 一键启动脚本
├── requirements.txt
├── Optimization.md                 # 变更与优化日志
└── README.md
```

---

## 核心模块说明

### 契约层（`app/core/`）

- `exceptions.py`：`AppError` 基类及子类（`ValidationError`、`FileTypeError`、`FileParseError`、`ResourceNotFoundError`、`InferenceError`、`TaskDispatchError`、`ServiceUnavailableError`）。
- `responses.py`：`success_response()` / `error_response()` 统一 JSON envelope。
- `validators.py`：`validate_upload_size()`、`validate_image_bytes()`、`validate_skip_frames()`、`validate_pagination()` 集中校验函数。

### 基础设施层（`app/infrastructure/ml/`）

- `backends.py`：ONNX Runtime 与 Ultralytics 双后端，带优先级选择与加载失败回退。
- `model_registry.py`：`ModelRegistry.loaded_map()` 统一管理多模型实例，支持 `class_mapping` 重映射。
- `rust_bridge.py`：`RustBridge` 封装 HTTP 调用，`filter_boxes()` / `dedupe_events()` 提供类型化高层接口，`health_check()` 暴露 Rust 微服务可用性与延迟。

### 服务层（`app/services/`）

- `detection_service.py`：主编排入口，依赖 `InferenceService`、`SceneService`、`AlertPolicyService`、`RenderingService`，通过 `DetectionServiceDeps` 依赖容器注入。
- `inference_service.py`：遍历 `model_registry`，合并多模型检测结果。
- `alert_policy_service.py`：图片/Base64 路径的告警冷却（内存 TTL）。
- `scene_service.py`：分析当前帧目标构成，输出场景状态描述（含 `timestamp`）。
- `rendering_service.py`：纯 cv2 绘框逻辑，与业务逻辑解耦。
- `video_service.py`：视频逐帧处理，告警冷却优先走 Rust 批量去重，回退纯 Python 路径。

### 升级流水线（`app/upgrade/`）

- `TrackEngine`：贪心 IoU 匹配追踪器（阈值 0.3），跨帧持久化目标 ID，最多丢失 10 帧后退出。
- `AlarmEngine`：连续帧计数器，`min_consecutive_frames` 判断基于真实 track_id，语义准确。
- `UpgradePipeline`：以"非破坏性旁路"方式为现有检测结果附加 `track_id` 和时序告警元数据。

---

## 模型与推理策略

模型配置以 `app/config.py` 为准：

| 配置键 | 默认路径 | 用途 |
|---|---|---|
| `garbage_onnx_model` | `app/models/garbege.onnx` | 垃圾类检测（优先） |
| `garbage_pt_model` | `app/models/garbege.pt` | 垃圾类检测（回退） |
| `fire_onnx_model` | `app/models/only_fire.onnx` | 火情检测（优先） |
| `fire_pt_model` | `app/models/only_fire.pt` | 火情检测（回退） |
| `smoke_onnx_model` | `app/models/fire_smoke.onnx` | 烟雾检测（优先） |
| `smoke_pt_model` | `app/models/fire_smoke.pt` | 烟雾检测（回退） |

推理策略：

1. 优先加载 ONNX Runtime 运行对应 `.onnx` 模型。
2. ONNX 不可用时回退到 Ultralytics `.pt` 权重。
3. 所有模型均不可用时保留演示模式，前端页面可正常联调。

---

## Rust 加速层

### 已实现算法

`rust/src/lib.rs` 中提供以下公开 API：

| 函数 | 说明 |
|---|---|
| `iou(a: BBox, b: BBox) -> f64` | 计算两框 IoU |
| `filter_overlapping_boxes(boxes, threshold)` | NMS 风格过滤重叠框 |
| `dedupe_track_events(events, cooldown_ms, iou_threshold)` | 批量去重告警事件（跨时间窗口） |

### REST 接口

Rust 微服务当前暴露以下端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/iou` | 计算两框 IoU |
| `POST` | `/v1/filter-boxes` | 过滤重叠框 |
| `POST` | `/v1/dedupe-events` | 批量去重事件 |

### Python 调用方式

Python 通过 `RustBridge.call(payload)` 以 JSON HTTP 请求调用 Rust REST 服务。高层接口：

```python
bridge = RustBridge(base_url="http://127.0.0.1:50051")

# NMS 过滤
filtered = bridge.filter_boxes(boxes=[[10,10,50,50], ...], threshold=0.5)

# 批量告警去重（含历史窗口）
deduped = bridge.dedupe_events(events=[...], cooldown_ms=1000, iou_threshold=0.4)

# 健康检查
status = bridge.health_check()
```

### 启动 Rust 微服务

```bash
cargo build --release --manifest-path rust/Cargo.toml
./rust/target/release/huali_garbage_core
```

Windows 下可执行文件为 `rust/target/release/huali_garbage_core.exe`。

> Rust 微服务不可用时，视频告警冷却等路径会自动回退到 Python 实现，系统仍可继续运行。

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

### 4. 编译 Rust 微服务（可选但推荐）

需要安装 [Rust 工具链](https://rustup.rs/)：

```bash
cargo build --release --manifest-path rust/Cargo.toml
```

---

## 环境变量

项目默认读取根目录 `.env` 文件，无 `.env` 时按以下默认值运行：

```env
APP_NAME=社区垃圾与火情识别预警系统
APP_VERSION=2.0.0
DEBUG=false
DATABASE_URL=sqlite:///garbage_system.db
REDIS_URL=redis://localhost:6379/0
VIDEO_DEFAULT_SKIP_FRAMES=1
CELERY_TASK_ALWAYS_EAGER=false
MAX_UPLOAD_SIZE_MB=200
RUST_SERVICE_URL=http://127.0.0.1:50051
RUST_SERVICE_TIMEOUT_SECONDS=2.0
```

---

## 启动方式

### 方式一：Windows 一键启动

```bat
start_queue.bat
```

脚本自动完成：检查/创建虚拟环境 → 安装依赖 → 启动 Rust REST 服务 → 启动 Celery Worker → 启动 FastAPI 并打开浏览器。

### 方式二：手动启动

```bash
# 启动 Rust REST 服务
cargo build --release --manifest-path rust/Cargo.toml
./rust/target/release/huali_garbage_core

# 启动 Web 服务
RUST_SERVICE_URL=http://127.0.0.1:50051 uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# （可选）启动 Celery Worker
RUST_SERVICE_URL=http://127.0.0.1:50051 python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

> 说明：视频任务优先通过 Celery 分发。本地无可用 Worker 时，系统自动回退到本地线程执行，兼顾演示易用性与正式链路扩展性。

---

## 常用接口

### 检测

```http
POST /api/detect/image          # 图片上传检测（form: file）
POST /api/detect/base64         # Base64 图像检测（json: {image: "data:image/..."} ）
POST /api/detect/video          # 视频任务提交（form: file, skip_frames）
GET  /api/tasks/{task_id}       # 视频任务状态与进度查询
```

### 记录与统计

```http
GET /api/alerts                         # 预警记录分页列表
GET /api/alerts/{record_uid}/image      # 预警截图获取
GET /api/statistics                     # 检测量、预警量、类别分布统计
GET /api/status                         # 服务状态（含 Rust 微服务状态）
GET /api/classes                        # 支持的检测类别列表
```

### 页面

| 路径 | 说明 |
|---|---|
| `/` | 首页总览 |
| `/detection` | 综合检测页（图片/摄像头/视频） |
| `/video` | 独立视频检测页 |
| `/alerts` | 预警记录页 |
| `/statistics` | 数据统计页 |
| `/dataset` | 数据集说明页 |
| `/docs` | FastAPI 在线接口文档 |

---

## 数据存储

| 数据 | 路径 |
|---|---|
| SQLite 数据库 | `garbage_system.db` |
| 预警截图 | `app/uploads/alerts/` |
| 视频上传与结果 | `app/uploads/videos/` |
| 静态访问前缀 | `/uploads/...` |

---

## 优缺点分析与优化建议

### 优点

**1. 清晰的分层架构**
契约层（`core/`）、基础设施层（`infrastructure/`）、服务层（`services/`）职责边界清晰，各层依赖方向单一，便于单独测试和替换。

**2. 务实的 Python + Rust 混合策略**
Rust 仅覆盖核心计算热路径（IoU、NMS、批量去重），Python 层保留完整回退逻辑。Rust 服务不可用时系统仍可正常运行，不强制所有部署都依赖 Rust。

**3. 双推理后端 + 自动回退**
ONNX Runtime → Ultralytics `.pt` → 演示模式三级降级，对运行环境的要求弹性较好，适合多种部署场景。

**4. 统一异常与响应体系**
`AppError` 层次结构 + `error_response()` envelope 保证 API 错误响应格式一致，前端处理逻辑简单。

**5. 视频告警冷却语义正确**
按类别分组（火情/烟雾 1s、垃圾/溢出 3s）独立冷却，同一目标在冷却窗口内不重复告警，避免了朴素实现中跨类干扰的问题。

**6. 真实 IoU 追踪器**
`TrackEngine` 实现贪心 IoU 匹配，跨帧目标 ID 稳定，使 `AlarmEngine` 的连续帧告警判断具有实际意义。

---

### 缺点

**1. 数据库为 SQLite，并发写入受限**
SQLite 写操作串行，在多 Celery Worker 并发写入预警记录时存在锁竞争，不适合高吞吐生产场景。

**2. Rust 微服务引入额外运维面**
相比进程内调用，独立 REST 服务需要额外的启动、探活与配置管理；如果未启动 Rust 服务，性能优化路径不会生效。

**3. 缺乏自动化测试**
核心服务层（`InferenceService`、`AlertPolicyService`、`TrackEngine`）均无单元测试或集成测试，重构时的回归风险无法量化。

**4. 模型文件不在版本控制中**
`.onnx` / `.pt` 权重文件体积大，未进入仓库，新成员克隆后无法直接运行，缺乏明确的模型获取说明。

**5. 前端为服务端渲染 + 原生 JS，扩展性有限**
Jinja2 + 原生 JavaScript 的组合在功能简单时够用，但随着页面交互复杂度增长，状态管理和组件复用将变得困难。

**6. 告警冷却历史存储在内存中**
`AlertPolicyService` 的冷却历史随进程重启丢失，多进程部署（多 Uvicorn Worker）下各进程历史独立，可能出现同一目标在不同进程上重复告警。

**7. `TrackEngine` 为单帧贪心匹配，精度有限**
当前追踪器不保留运动预测信息（如卡尔曼滤波），目标快速移动或短暂遮挡时 ID 切换率较高，连续帧告警语义准确性受影响。

---

### 优化建议

#### 短期（工程质量）

| 建议 | 说明 |
|---|---|
| **补充单元测试** | 优先覆盖 `TrackEngine`、`dedupe_track_events`、`validate_*` 系列函数，可用 `pytest` + `hypothesis` 进行属性测试 |
| **完善模型获取说明** | 在 README 或 `app/models/` 下提供模型下载脚本或 DVC 配置，使新成员可一键获取权重 |
| **引入 `pre-commit` Hooks** | 配置 `ruff` + `mypy` + `cargo fmt/clippy`，保持代码风格统一，防止类型错误上线 |
| **告警历史持久化** | 将 `AlertPolicyService` 的冷却历史从内存字典迁移到 Redis（TTL key），解决多进程/重启后状态丢失问题 |

#### 中期（性能与可靠性）

| 建议 | 说明 |
|---|---|
| **为 Rust 微服务增加自动拉起或容器编排** | 通过 Docker Compose、Supervisor 或系统服务管理 Rust REST 进程，减少手动启动成本 |
| **数据库迁移至 PostgreSQL** | 生产或高并发演示场景下替换 SQLite，解决写锁问题；使用 Alembic 管理 Schema 变更 |
| **视频处理引入帧缓冲队列** | 将解码、推理、编码解耦为三个并发步骤，充分利用多核，降低端到端延迟 |
| **跨模型 NMS** | 在 `InferenceService.detect()` 中调用 `RustBridge.filter_boxes()` 对多模型重叠检测框进行去重，减少重复告警 |

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
