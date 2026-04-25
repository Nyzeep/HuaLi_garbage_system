# HuaLi_garbage_system

> 本项目为中国大学生计算机设计大赛作品，聚焦社区场景下的垃圾治理与火情风险预警，构建了一套集图像识别、视频分析、告警留存、统计展示于一体的智能巡检系统。

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

社区垃圾与火情识别预警系统面向智慧社区、园区巡检与安全治理场景，围绕“发现问题、生成预警、留存记录、辅助管理”这一闭环展开设计。项目通过 FastAPI 提供统一 Web 页面与接口服务，结合 YOLO / ONNX 推理能力，对上传图片、摄像头图像和视频内容进行识别分析，并将预警结果沉淀到本地数据库中，便于后续查询、统计与展示。

系统当前已经形成从前端页面、后端接口、异步视频处理到记录分析的完整链路，适合课程设计、竞赛展示、功能拓展与本地部署演示。

## 系统亮点

- 多入口检测：支持图片上传检测、Base64 图像检测、视频上传检测。
- 风险场景覆盖：围绕社区垃圾桶、满溢、散落垃圾、火情等巡检场景构建识别能力。
- 双后端推理：优先使用 ONNX Runtime，必要时自动回退到 Ultralytics 权重推理。
- 桶体细粒度识别：支持对垃圾桶目标补充颜色分类结果，便于后续做桶型关联与展示。
- 视频任务编排：支持 Celery 异步处理，也支持在本地线程中自动兜底执行。
- 结果可追踪：预警截图、检测记录、视频任务状态均可落库保存。
- 可视化展示：内置首页、综合检测页、视频页、预警页、统计页、数据集说明页。
- 升级流水线：视频链路集成检测、跟踪、时序告警的升级版处理流程，可为结果附加 `track_id` 和时序告警信息。

## 技术架构

### 后端与服务

- `FastAPI`：Web 框架与 API 组织
- `Uvicorn`：ASGI 服务启动
- `Pydantic v2`：请求与响应模型校验
- `SQLAlchemy`：SQLite ORM 持久化
- `Celery + Redis`：视频异步任务调度

### 媒体处理

- `OpenCV`：图像解码、绘框与视频帧处理
- `NumPy`：张量与数组运算
- `ONNX Runtime`：ONNX 模型推理
- `Ultralytics YOLO`：`.pt` 权重加载与推理
- `PyTorch / TorchVision`：YOLO 运行依赖
- `Pillow`：图像处理基础依赖
- `ImageIO / imageio-ffmpeg`：视频写出与编码支持

### 前端展示

- `Jinja2`：模板渲染
- `Tailwind CSS CDN`：页面样式系统
- 原生 `JavaScript`：前端交互、上传、轮询、结果渲染

## 功能概览

### 检测能力

- `POST /api/detect/image`：图片上传检测
- `POST /api/detect/base64`：摄像头或前端抓拍图像检测
- `POST /api/detect/video`：视频检测任务提交
- `GET /api/tasks/{task_id}`：视频任务状态轮询

### 数据能力

- 预警记录保存与分页查询
- 检测结果截图保存与回显
- 任务进度、结果视频、统计数据统一管理
- 启动后自动建表与初始化上传目录

### 页面能力

- `/`：首页总览
- `/detection`：综合检测页
- `/video`：独立视频检测页
- `/alerts`：预警记录页
- `/statistics`：数据统计页
- `/dataset`：数据集说明页
- `/docs`：FastAPI 在线接口文档

## 主要代码结构

```text
garbage_system/
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
├── garbage_system.db       # 本地 SQLite 数据库
├── start_queue.bat         # Windows 一键启动脚本
├── requirements.txt
└── README.md
```

## 核心模块说明

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

### 升级版时序处理

- `app/upgrade/pipeline.py`：检测、跟踪、时序告警组合流程
- `app/upgrade/tracker.py`：目标跟踪占位实现
- `app/upgrade/alarm.py`：连续帧告警规则
- `app/upgrade/detection.py`：检测结果适配器

### 数据辅助工具

- `tools/prelabel_garbage_bin.py`：使用检测模型批量预标注垃圾桶框，生成 YOLO 标签
- `tools/quick_relabel_tool.py`：基于 OpenCV 的快速人工复标工具，适合颜色分类数据清洗

## 模型与推理策略

项目配置以 `app/config.py` 为准，当前主流程采用多模型组合方式工作：

- 垃圾相关模型：`app/models/garbege.onnx` / `app/models/garbege.pt`
- 火情相关模型：`app/models/only_fire.onnx` / `app/models/only_fire.pt`
- 烟雾相关模型：`app/models/fire_smoke.onnx` / `app/models/fire_smoke.pt`
- 桶体颜色分类模型：`app/models/bin_color_resnet18.pt`
- 其余模型资源与数据展示内容保留在仓库中，便于后续扩展、展示与实验

推理策略如下：

1. 优先尝试 ONNX Runtime 加载模型。
2. 若 ONNX 不可用，则回退到 Ultralytics `.pt` 权重。
3. 若运行环境中没有可用模型，也保留演示模式能力，便于前端页面联调与功能展示。

## 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/Nyzeep/HuaLi_garbage_system.git
cd HuaLi_garbage_system
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows 激活方式：

```bash
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量示例

项目默认会读取根目录 `.env` 文件，没有 `.env` 时也可按默认配置直接运行。

```env
APP_NAME=社区垃圾与火情识别预警系统
APP_VERSION=2.0.0
DEBUG=false
DATABASE_URL=sqlite:///garbage_system.db
REDIS_URL=redis://localhost:6379/0
VIDEO_DEFAULT_SKIP_FRAMES=1
CELERY_TASK_ALWAYS_EAGER=false
```

## 启动方式

### 方式一：Windows 一键启动

双击目录下start_queue.bat,或者通过以下命令行执行

```bat
start_queue.bat
```

脚本会自动完成以下流程：

1. 检查当前环境或项目目录下可用的虚拟环境
2. 自动创建 `.venv`（如不存在）
3. 自动安装缺失依赖
4. 启动 Celery Worker
5. 启动 FastAPI Web 服务并打开浏览器（默认访问 `http://127.0.0.1:8010`）

### 方式二：手动启动

启动 Web：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

如需启用异步队列，再启动 Worker：

```bash
python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

说明：

- 视频任务优先通过 Celery 分发执行。
- 当本地没有可用 worker 时，系统仍可自动回退到本地线程完成视频处理。
- 这种设计兼顾了演示环境的易用性与正式链路的扩展性。

## 常用接口

### 图片检测

```http
POST /api/detect/image
```

表单字段：

- `file`：图片文件

### Base64 图像检测

```http
POST /api/detect/base64
```

请求体示例：

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

### 视频检测

```http
POST /api/detect/video
```

表单字段：

- `file`：视频文件
- `skip_frames`：跳帧数，默认读取 `VIDEO_DEFAULT_SKIP_FRAMES`

任务查询：

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

- SQLite 数据库：`garbage_system.db`
- 预警截图目录：`app/uploads/alerts/`
- 视频上传与结果目录：`app/uploads/videos/`
- 静态访问路径前缀：`/uploads/...`

## 页面展示

系统默认提供统一的竞赛展示风格页面：

- 首页：能力概览与功能入口
- 综合检测页：图片、摄像头、视频统一检测入口
- 视频页：独立视频处理与进度轮询
- 预警页：历史记录、状态筛选与图片查看
- 统计页：检测量、预警量、类别分布
- 数据集页：类别说明、模型信息与数据展示

## 适用场景

- 中国大学生计算机设计大赛项目展示
- 智慧社区巡检与风险识别演示
- 课程设计与毕业设计原型系统
- 目标检测、视频任务处理、可视化展示的综合练习项目

## 许可证

本项目采用 [MIT License](LICENSE)。

## 支持项目

如果这个项目对你有帮助，欢迎给仓库点亮一个 Star。这会是对项目持续完善与维护非常大的支持。
