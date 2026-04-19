# HuaLi_garbage_system

> 本项目为大学生计算机设计大赛参赛项目，面向智慧校园与智慧社区场景，构建了一个集目标检测、视频异步处理、告警留存与统计分析于一体的智能巡检系统。项目目前采用 `FastAPI + Celery + SQLite + YOLO/ONNX` 架构，并提供 Windows 一键启动脚本 `start_queue.bat`。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-4051B5)](https://www.uvicorn.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063)](https://docs.pydantic.dev/)
[![Jinja2](https://img.shields.io/badge/Jinja2-Templates-B41717)](https://jinja.palletsprojects.com/)
[![Celery](https://img.shields.io/badge/Celery-5.4%2B-37814A?logo=celery)](https://docs.celeryq.dev/)
[![Redis](https://img.shields.io/badge/Redis-5.2%2B-DC382D?logo=redis)](https://redis.io/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)](https://www.sqlite.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0%2B-D71F00?logo=sqlalchemy)](https://www.sqlalchemy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array-013243?logo=numpy)](https://numpy.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-FF9F00)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.20%2B-grey?logo=onnx)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目概览

这是一个面向智慧校园与社区场景的智能巡检系统，当前仓库已经具备下面这些核心能力：

- 图片上传检测
- Base64 图像检测
- 视频异步检测与结果视频回放
- 告警记录落库、截图留存、统计分析
- ONNX Runtime 优先推理，失败时回退到 Ultralytics `.pt`
- Celery + Redis 视频任务队列；如果没有可用 worker，会自动回退到本地线程处理
- FastAPI 页面路由与接口统一管理

当前主入口是 [app/main.py](app/main.py)。仓库里仍保留了历史 Flask 原型 [app/app.py](app/app.py)，但它已经不是当前推荐入口。

## 当前仓库状态

- Web 主线：`FastAPI`
- 异步任务：`Celery`
- 数据库：`SQLite`
- 页面：`Jinja2` 模板
- 模型推理：`ONNX Runtime` / `Ultralytics`
- 启动脚本：`start_queue.bat`

还有两点需要特别说明：

1. `package.json` 里只保留了 `docx` 依赖，当前 Web 页面并不是 Node.js 构建产物，主运行链路不依赖前端构建工具。
2. 仓库里保留了烟雾相关模型、数据集和模板展示资源，但当前 FastAPI 主检测链路在 [app/services/detection_service.py](app/services/detection_service.py) 中实际接入的是“垃圾相关模型”和“火焰模型”。

## 功能对应代码

### Web 与 API

- [app/main.py](app/main.py): FastAPI 应用入口
- [app/api/pages.py](app/api/pages.py): 页面路由
- [app/api/routes.py](app/api/routes.py): 检测、任务、告警、统计接口
- [app/templates/index.html](app/templates/index.html): 首页
- [app/templates/detection.html](app/templates/detection.html): 检测页
- [app/templates/video.html](app/templates/video.html): 视频检测页
- [app/templates/alerts.html](app/templates/alerts.html): 告警页
- [app/templates/statistics.html](app/templates/statistics.html): 统计页
- [app/templates/dataset.html](app/templates/dataset.html): 数据集展示页

### 推理与视频处理

- [app/services/inference.py](app/services/inference.py): ONNX / Ultralytics 双后端
- [app/services/detection_service.py](app/services/detection_service.py): 检测主逻辑、场景分析、绘框
- [app/services/video_service.py](app/services/video_service.py): 视频逐帧处理、视频内告警去重
- [app/tasks.py](app/tasks.py): Celery 视频任务

### 数据落库与状态记录

- [app/database.py](app/database.py): 数据库引擎与会话
- [app/db_models.py](app/db_models.py): 告警记录、检测明细、视频任务表
- [app/services/record_service.py](app/services/record_service.py): 告警与任务记录写入/查询
- [app/bootstrap.py](app/bootstrap.py): 启动时建表与目录初始化

### 训练与数据处理脚本

- [train_garbage.py](train_garbage.py): 垃圾检测训练脚本
- [train_yolo.py](train_yolo.py): 垃圾检测训练脚本，包含部分绝对路径
- [train_fire_smoke.py](train_fire_smoke.py): 火焰/烟雾训练脚本，包含 Linux 风格绝对路径
- [export_onnx.py](export_onnx.py): 权重导出 ONNX
- [convert_coco.py](convert_coco.py), [convert_coco2yolo_separate.py](convert_coco2yolo_separate.py): COCO 转 YOLO
- [merge_datasets.py](merge_datasets.py), [create_smoke_dataset.py](create_smoke_dataset.py): 数据集整理
- [analyze_dataset.py](analyze_dataset.py): 数据集标注统计
- [detect_video.py](detect_video.py), [detect_fire_smoke.py](detect_fire_smoke.py): 历史/独立测试脚本

## 项目结构

```text
garbage_system/
├─ app/
│  ├─ api/                 # 页面与接口路由
│  ├─ models/              # 推理权重与 ONNX 文件
│  ├─ services/            # 检测、视频、记录服务
│  ├─ templates/           # Jinja2 页面模板
│  ├─ uploads/             # 告警截图、视频输出
│  ├─ bootstrap.py         # 启动初始化
│  ├─ celery_app.py        # Celery 应用
│  ├─ config.py            # 配置
│  ├─ database.py          # 数据库连接
│  ├─ db_models.py         # ORM 模型
│  └─ main.py              # FastAPI 入口
├─ dataset/                # 垃圾相关数据集
├─ dataset_fire/           # 火焰数据集
├─ dataset_smoke_5images_new/  # 烟雾样例数据集
├─ runs/                   # 训练/检测输出
├─ start_queue.bat         # Windows 一键启动脚本
├─ requirements.txt
├─ garbage_system.db
└─ README.md
```

## 安装依赖

### 1. 克隆项目

```bash
git clone https://github.com/Nyzeep/HuaLi_garbage_system.git
cd HuaLi_garbage_system
```

### 2. 手动创建虚拟环境

如果你准备手动启动项目，推荐这样创建虚拟环境：

```bash
python -m venv .venv
```

如果你已经有自己的虚拟环境目录名，也可以继续沿用，`start_queue.bat` 会自动尝试检测项目根目录下可用的虚拟环境。

### 3. 激活虚拟环境

Windows:

```bash
.venv\Scripts\activate
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 可选环境变量

项目会自动读取根目录 `.env`。没有 `.env` 时也能按默认配置启动。

示例：

```env
APP_NAME=Garbage Detection System
APP_VERSION=2.0.0
DEBUG=false
DATABASE_URL=sqlite:///garbage_system.db
REDIS_URL=redis://localhost:6379/0
VIDEO_DEFAULT_SKIP_FRAMES=1
CELERY_TASK_ALWAYS_EAGER=false
```

## 模型加载规则

当前主应用以 [app/config.py](app/config.py) 为准，自动加载优先级如下：

- 垃圾模型：`app/models/garbege.onnx`，找不到时回退到 `app/models/garbege.pt`
- 火焰模型：`app/models/only_fire.onnx`，找不到时回退到 `app/models/only_fire.pt`

仓库里虽然还存在这些模型文件：

- `app/models/fire_smoke.onnx`
- `app/models/fire_smoke.pt`
- `app/models/smoke_yolov8.pt`

但它们目前没有接入 FastAPI 主检测链路。

另外，`garbege` 这个文件名是历史命名，当前代码就是按这个名字读取的，不要直接重命名，除非同时修改配置。

## 启动方式

### 方式一：使用 `start_queue.bat`

适合 Windows 本机开发与演示。

```bat
start_queue.bat
```

这个脚本会按下面的流程自动处理：

1. 检测当前激活的虚拟环境，或扫描项目根目录下是否存在可用虚拟环境
2. 如果没有找到可用虚拟环境，则自动创建 `.venv`
3. 如果检测到依赖缺失，则自动执行 `pip install -r requirements.txt`
4. 新开一个窗口启动 Celery Worker
5. 新开一个窗口启动 FastAPI，并自动打开浏览器

脚本里实际执行的核心命令是：

```bash
python -m celery -A app.celery_app worker --loglevel=info --pool=solo
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

脚本当前不再强依赖 Redis 服务是否已安装或已启动；即使没有 Redis，项目中的视频任务也会在检测不到可用队列时自动回退到本地线程处理。

提示：Redis 不是项目运行的必要条件，但推荐在正式部署时配合 Celery 一起使用。这样视频任务可以通过 `Celery + Redis` 异步队列更稳定地执行；若未检测到可用 worker，系统会自动回退到本地线程处理模式。

### 方式二：手动启动

先启动 Web：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

再在另一个终端启动 Celery Worker：

```bash
python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

如果不启动 Celery Worker，视频任务仍然可以运行；[app/api/routes.py](app/api/routes.py) 会检测不到 worker 时自动回退到本地线程处理。这样适合轻量测试，但不适合长期高并发。

## 页面入口

服务启动后可访问：

- `http://127.0.0.1:8000/`: 首页，用于查看项目简介、能力概览和快速导航
- `http://127.0.0.1:8000/detection`: 检测页，用于上传图片或提交图像数据进行识别
- `http://127.0.0.1:8000/video`: 视频检测页，用于上传视频、提交异步任务并查看处理结果
- `http://127.0.0.1:8000/alerts`: 告警页，用于查看历史告警记录和告警截图
- `http://127.0.0.1:8000/statistics`: 统计页，用于查看检测数量、告警数量和类别统计
- `http://127.0.0.1:8000/dataset`: 数据集展示页，用于展示项目涉及的数据集与类别信息
- `http://127.0.0.1:8000/docs`: FastAPI 接口文档页，用于接口调试和开发联调

## 常用接口

### 图片检测

```http
POST /api/detect/image
```

表单字段：

- `file`: 图片文件

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

- `file`: 视频文件
- `skip_frames`: 跳帧数，当前默认值为 `1`

查询任务状态：

```http
GET /api/tasks/{task_id}
```

### 告警与统计

```http
GET /api/alerts
GET /api/alerts/{record_uid}/image
GET /api/statistics
GET /api/status
GET /api/classes
```

## 数据存储位置

- SQLite 数据库：`garbage_system.db`
- 告警截图：`app/uploads/alerts/`
- 视频上传与结果视频：`app/uploads/videos/`
- 上传静态访问前缀：`/uploads/...`

[app/bootstrap.py](app/bootstrap.py) 会在启动时自动创建数据库表和上传目录。

## 当前检测逻辑说明

主应用里定义的类别常量仍然包含 5 类：

- 垃圾桶
- 垃圾满溢
- 散落垃圾
- 火焰
- 烟雾

但当前 FastAPI 主检测链路的实际接入状态是：

- 已接入：垃圾相关检测、火焰检测
- 未接入主链路：烟雾检测

如果你后续准备继续完善烟雾检测，需要优先检查：

- [app/services/detection_service.py](app/services/detection_service.py)
- [app/config.py](app/config.py)
- [app/services/inference.py](app/services/inference.py)

## 训练与导出

常用脚本如下：

```bash
python train_garbage.py --mode train
python train_garbage.py --mode val
python train_yolo.py --mode train
python train_fire_smoke.py
python export_onnx.py
```

需要注意：

- `train_yolo.py` 中存在 `D:/garbage_system/...` 形式的绝对路径
- `train_fire_smoke.py` 中存在 `/root/workspace/...` 形式的绝对路径
- 一些数据处理脚本中也写入了本机路径或下载目录
- `export_onnx.py` 会导出若干历史模型对，但主应用实际自动加载的文件名仍然以 [app/config.py](app/config.py) 为准

因此，这些训练和数据脚本在换机器前通常都需要先改路径。

## 已知注意事项

- `app/app.py` 是旧版 Flask 原型，不建议作为主入口继续维护
- `requirements.txt` 中仍保留了 Flask 相关依赖，这是为了兼容历史文件
- 当前视频任务优先走 Celery，检测不到 worker 时自动回退到本地线程
- README 已根据项目现阶段实现重新整理，重点覆盖现在可直接运行和复现的功能链路

## 许可证

本项目使用 [MIT License](LICENSE)。

## 支持项目

如果这个项目对你有帮助，欢迎给仓库点一个 Star，这会是对项目维护和后续完善很大的支持。
