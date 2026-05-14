
# Ecoscout

> 面向社区环境的智能巡检系统，聚焦垃圾治理与火情风险预警，提供图像识别、视频分析、告警留存及统计展示的全链路解决方案。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Uvicorn-ASGI-4051B5" alt="Uvicorn">
  <img src="https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white" alt="Pydantic">
  <img src="https://img.shields.io/badge/Jinja2-Templates-B41717?logo=jinja&logoColor=white" alt="Jinja2">
  <img src="https://img.shields.io/badge/TailwindCSS-CDN-06B6D4?logo=tailwindcss&logoColor=white" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/Celery-5.4%2B-37814A?logo=celery&logoColor=white" alt="Celery">
  <img src="https://img.shields.io/badge/Redis-5.2%2B-DC382D?logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0%2B-D71F00?logo=sqlalchemy&logoColor=white" alt="SQLAlchemy">
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-Array-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pillow-Image-8C52FF" alt="Pillow">
  <img src="https://img.shields.io/badge/ImageIO-Video-4B5563" alt="ImageIO">
  <img src="https://img.shields.io/badge/ONNX_Runtime-1.20%2B-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/Ultralytics-YOLO-FF9F00" alt="Ultralytics">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/TorchVision-0.19%2B-EE4C2C" alt="TorchVision">
  <img src="https://img.shields.io/badge/License-MIT-F7DF1E" alt="MIT">
</p>

## 项目简介

Ecoscout 面向智慧社区、园区巡检及环境安全治理场景，围绕“发现问题 → 生成预警 → 留存记录 → 辅助管理”的闭环进行设计。  
系统通过 FastAPI 提供统一的 Web 页面与接口服务，结合 YOLO / ONNX 推理能力，对上传图片、摄像头画面及视频内容进行智能分析，并将预警结果持久化到本地数据库，便于查询、统计与展示。

目前已完成从前端页面、后端接口、异步视频处理到记录分析的全链路整合，适合本地部署、功能演示及进一步扩展。

## 核心特性

- **多入口检测**：支持图片上传、Base64 图像、视频上传等多种检测方式。
- **风险场景覆盖**：涵盖社区垃圾桶满溢、散落垃圾、火情等典型巡检需求。
- **双引擎推理**：优先使用 ONNX Runtime，必要时自动回退到 Ultralytics 权重推理。
- **细粒度识别**：对垃圾桶目标补充颜色分类结果，便于后续桶型关联与展示。
- **异步视频处理**：基于 Celery 的任务队列，同时支持本地线程兜底执行。
- **结果可追踪**：预警截图、检测记录、视频任务状态全部入库保存。
- **可视化面板**：内置首页、检测页、视频页、预警页、统计页及数据集说明页。
- **升级流水线**：视频链路集成目标跟踪与时序告警，输出 `track_id` 及连续帧告警信息。

## 技术栈

### 后端与服务

- `FastAPI`：Web 框架与 API 组织
- `Uvicorn`：ASGI 服务启动
- `Pydantic v2`：请求与响应模型校验
- `SQLAlchemy`：SQLite ORM 持久化
- `Celery + Redis`：视频异步任务调度

### 媒体与推理

- `OpenCV`：图像解码、绘制检测框与视频帧处理
- `NumPy`：张量与数组运算
- `ONNX Runtime`：ONNX 模型推理
- `Ultralytics YOLO`：`.pt` 权重加载与推理
- `PyTorch / TorchVision`：YOLO 运行依赖
- `Pillow`：图像处理基础库
- `ImageIO / imageio-ffmpeg`：视频编码与输出

### 前端展示

- `Jinja2`：模板渲染
- `Tailwind CSS CDN`：页面样式
- 原生 `JavaScript`：交互、上传、轮询与结果渲染

## 功能概览

### 检测接口

- `POST /api/detect/image`：上传图片进行检测
- `POST /api/detect/base64`：摄像头或前端抓拍图像检测
- `POST /api/detect/video`：提交视频检测任务
- `GET /api/tasks/{task_id}`：轮询视频任务处理状态

### 数据接口

- 预警记录存储与分页查询
- 检测结果截图保存与回显
- 任务进度、结果视频、统计数据统一管理
- 服务启动时自动建表并初始化上传目录

### 页面路由

- `/`：首页总览
- `/detection`：综合检测页
- `/video`：视频检测页
- `/alerts`：预警记录页
- `/statistics`：数据统计页
- `/dataset`：数据集说明页
- `/docs`：FastAPI 自动生成的接口文档

## 目录结构

```text
ecoscout/
├── app/
│   ├── api/                # 页面路由与检测/统计/任务接口
│   ├── models/             # 检测模型、分类模型与训练产物
│   ├── services/           # 检测、视频、记录、桶体颜色服务
│   ├── templates/          # Jinja2 前端页面
│   ├── upgrade/            # 跟踪与时序告警升级流水线
│   ├── bootstrap.py        # 启动初始化
│   ├── celery_app.py       # Celery 应用
│   ├── config.py           # 项目配置
│   ├── constants.py        # 类别常量
│   ├── database.py         # 数据库连接
│   ├── dependencies.py     # 依赖注入与服务装配
│   ├── db_models.py        # ORM 模型
│   ├── main.py             # FastAPI 入口
│   ├── schemas.py          # Pydantic 响应模型
│   ├── tasks.py            # 视频异步任务
│   └── utils.py            # 图像/Base64/路径等通用工具
├── tools/                  # 数据预标注与人工复标辅助脚本
├── .env.example            # 环境变量示例
├── ecoscout.db             # 本地 SQLite 数据库
├── start_queue.bat         # Windows 一键启动脚本
├── requirements.txt
└── README.md
```

## 模块详解

### Web 与接口

- `app/main.py`：FastAPI 应用入口
- `app/api/pages.py`：页面路由注册
- `app/api/routes.py`：检测、预警、统计、任务相关接口

### 检测与视频处理

- `app/services/inference.py`：封装 ONNX Runtime 与 Ultralytics 双推理后端
- `app/services/detection_service.py`：检测主逻辑、场景分析、绘框渲染
- `app/services/bin_color_service.py`：基于 ResNet18 的垃圾桶颜色分类服务
- `app/services/video_service.py`：逐帧处理、告警冷却、视频统计
- `app/tasks.py`：视频任务异步执行封装

### 数据与统计

- `app/database.py`：数据库引擎与会话管理
- `app/db_models.py`：预警记录、检测记录、视频任务模型
- `app/services/record_service.py`：记录写入、查询与统计构建
- `app/bootstrap.py`：启动时自动建表、创建上传目录

### 升级时序处理

- `app/upgrade/pipeline.py`：检测、跟踪、时序告警组合流程
- `app/upgrade/tracker.py`：目标跟踪占位实现
- `app/upgrade/alarm.py`：连续帧告警规则
- `app/upgrade/detection.py`：检测结果适配器

### 辅助工具

- `tools/prelabel_garbage_bin.py`：使用检测模型批量预标注垃圾桶框，生成 YOLO 标签
- `tools/quick_relabel_tool.py`：基于 OpenCV 的快速人工复标工具，适用于颜色分类数据清洗

## 模型与推理策略

项目配置位于 `app/config.py`，当前采用多模型组合方式：

- 垃圾检测模型：`app/models/garbege.onnx` / `app/models/garbege.pt`
- 火情检测模型：`app/models/only_fire.onnx` / `app/models/only_fire.pt`
- 烟雾检测模型：`app/models/fire_smoke.onnx` / `app/models/fire_smoke.pt`
- 桶体颜色分类模型：`app/models/bin_color_resnet18.pt`

推理策略：

1. 优先尝试 ONNX Runtime 加载模型；
2. 若 ONNX 不可用，则回退到 Ultralytics `.pt` 权重；
3. 若环境中没有任何可用模型，系统进入演示模式，便于前端联调。

## 安装与运行

### 离线交付说明

本系统为方便现场演示，建议直接使用包含完整虚拟环境（如 `.venv311`、`.venv` 或 `venv`）的项目副本。依赖库已预先安装在虚拟环境中，只需双击 `start_queue.bat` 即可启动，无需联网或重新安装依赖。

若仅拷贝源码文件（不含虚拟环境），启动脚本会提示错误并中止，请使用完整的离线包。

### 开发环境安装

如需从源码重新配置环境，请按以下步骤操作：

```bash
git clone https://github.com/Nyzeep/Ecoscout.git
cd Ecoscout
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 环境变量配置

项目默认读取根目录下的 `.env` 文件，若不存在则使用默认配置。

```env
APP_NAME=Ecoscout
APP_VERSION=2.0.0
DEBUG=false
DATABASE_URL=sqlite:///ecoscout.db
REDIS_URL=redis://localhost:6379/0
VIDEO_DEFAULT_SKIP_FRAMES=1
CELERY_TASK_ALWAYS_EAGER=false
```

## 启动方式

### 方式一：Windows 一键启动

双击项目根目录下的 `start_queue.bat` 或在命令行中执行：

```bat
start_queue.bat
```

脚本会自动完成：

1. 检测可用的虚拟环境；
2. 验证关键依赖是否已安装；
3. 启动 Celery Worker；
4. 启动 FastAPI 服务并自动打开浏览器（默认地址 `http://127.0.0.1:8010`）。

> 注：一键启动脚本面向离线演示场景，不会自动创建虚拟环境或联网安装依赖。请确保项目目录下已包含 `.venv311`、`.venv` 或 `venv`。

### 方式二：手动启动

启动 Web 服务：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

如需启用异步队列（可选），另开终端启动 Worker：

```bash
python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

> 说明：视频任务优先通过 Celery 分发执行；若没有可用 Worker，系统会自动回退到本地线程处理，兼顾演示的便利性与正式链路的扩展性。

## 常用接口

### 图片检测

```http
POST /api/detect/image
```
表单字段：`file`（图片文件）

### Base64 图像检测

```http
POST /api/detect/base64
```
请求体示例：
```json
{ "image": "data:image/jpeg;base64,..." }
```

### 视频检测

```http
POST /api/detect/video
```
表单字段：`file`（视频文件），`skip_frames`（跳帧数，默认为 `VIDEO_DEFAULT_SKIP_FRAMES`）

任务状态查询：
```http
GET /api/tasks/{task_id}
```

### 记录与状态

```http
GET /api/alerts
GET /api/alerts/{record_uid}/image
GET /api/alerts/{record_uid}/detail
GET /api/statistics
GET /api/status
GET /api/classes
```

## 数据存储

- SQLite 数据库：`ecoscout.db`
- 预警截图目录：`app/uploads/alerts/`
- 视频上传与结果目录：`app/uploads/videos/`
- 静态资源访问前缀：`/uploads/...`

## 界面预览

系统默认提供清晰的功能页面：

- **首页**：能力概览与功能入口
- **综合检测页**：图片、摄像头、视频统一检测
- **视频页**：独立视频处理与进度轮询
- **预警页**：历史记录、状态筛选与图片查看
- **统计页**：检测量、预警量、类别分布
- **数据集页**：类别说明、模型信息与数据展示

## 适用场景

- 智慧社区巡检与风险识别演示
- 课程设计 / 毕业设计原型系统
- 目标检测、视频任务处理及可视化展示的综合实践项目
- 本地部署的环境监控解决方案

## 许可证

本项目采用 [MIT License](LICENSE)

## 支持项目

如果这个项目对你有帮助，欢迎点亮 Star。你的支持是项目持续改进的重要动力。
```
