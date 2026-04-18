# 🗑️🔥 HuaLi_garbage_system

<div align="center">

**广州华立学院 · 大学生计算机设计大赛作品**

*垃圾满溢智能检测 · 火情实时预警系统*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-FF9F00?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.18+-grey?logo=onnx)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green?logo=opencv)](https://opencv.org/)
[![Celery](https://img.shields.io/badge/Celery-5.3+-37814A?logo=celery)](https://docs.celeryq.dev/)
[![Redis](https://img.shields.io/badge/Redis-7.2+-DC382D?logo=redis)](https://redis.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

<br>

<p align="center">
  <img src="docs/demo-banner.png" alt="系统演示" width="80%">
</p>

## 📖 项目简介

**HuaLi_garbage_system** 是一套面向智慧校园/智慧社区的 **垃圾站智能监控解决方案**。  
系统通过摄像头实时采集垃圾桶区域图像，利用深度学习模型 **同时检测** 垃圾桶满溢状态与火情隐患，并在异常发生时通过可视化界面、语音播报、短信/邮件等方式**即时预警**，有效提升环卫管理效率与公共安全水平。

本项目为 **大学生计算机设计大赛** 参赛作品，体现了计算机视觉、边缘计算与物联网技术的综合应用。

---

## ✨ 核心功能

| 🗑️ 垃圾满溢检测 | 🔥 火情预警 | 📊 数据可视化 |
| :---: | :---: | :---: |
| 实时分析每个垃圾桶的 **填充率**，超过阈值自动标记为满溢状态 | 基于火焰与烟雾特征识别早期火情，响应速度 **< 2 秒** | Web 仪表盘展示历史满溢记录、火情事件统计与趋势图表 |

| 📷 多路视频流支持 | 🔊 多端告警通知 | 📅 巡检报告导出 |
| :---: | :---: | :---: |
| 同时接入 4 路以上摄像头，覆盖多个垃圾投放点 | 现场语音提示、后台弹窗、短信/邮件推送 | 自动生成每日/每周检测报告（PDF/Excel） |

---

## 🧠 技术架构

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│   摄像头阵列    │ ──> │   边缘推理端    │ ──> │   FastAPI 后端服务      │
│  (RTSP / USB)   │     │ ONNX Runtime +  │     │ (Uvicorn 异步处理)      │
└─────────────────┘     │    OpenCV       │     └───────────┬─────────────┘
                        └─────────────────┘                 │
                                                              ▼
                                                    ┌─────────────────┐     ┌─────────────────┐
                                                    │ Celery 异步队列 │ <─> │   Redis 缓存    │
                                                    │ (定时/告警任务) │     │ (状态/限流)     │
                                                    └─────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │   SQLAlchemy    │
                                                    │ (PostgreSQL /   │
                                                    │   SQLite)       │
                                                    └─────────────────┘
```

### 技术栈明细

| 层级         | 技术选型                                                                       | 作用说明                                     |
| ------------ | ------------------------------------------------------------------------------ | -------------------------------------------- |
| **Web 框架** | FastAPI + Uvicorn + Pydantic                                                   | 高性能异步 API、请求验证与文档自动生成       |
| **模型训练** | Ultralytics YOLOv8                                                             | 训练垃圾满溢与火焰烟雾检测模型               |
| **推理部署** | ONNX Runtime (CPU / CUDA)                                                      | 高效跨平台模型推理，脱离 PyTorch 依赖        |
| **图像处理** | OpenCV                                                                         | 视频流读取、图像预处理与 ROI 提取            |
| **异步任务** | Celery + Redis                                                                 | 告警推送、定时巡检、报告生成等后台任务       |
| **数据库**   | SQLAlchemy (支持 SQLite / PostgreSQL) + Alembic 迁移                            | ORM 操作，事件记录、用户管理、配置存储       |
| **前端**     | Jinja2 模板 / Vue3 (可选) + Bootstrap 5 + ECharts                               | 响应式仪表盘与数据可视化                     |

---

## 🚀 快速开始

### 环境要求
- Python **3.10** 或更高版本
- CUDA 11.8+ (如需 GPU 加速推理，可选)
- Redis 服务 (用于 Celery)
- 至少 4GB RAM，推荐 8GB+

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Nyzeep/HuaLi_garbage_system.git
cd HuaLi_garbage_system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量 (复制示例文件并修改)
cp .env.example .env
# 编辑 .env 设置 Redis 地址、摄像头 RTSP 流、告警邮箱等

# 5. 导出 ONNX 模型 (从训练好的 PyTorch 权重)
python scripts/export_onnx.py --weights models/best.pt --output models/best.onnx

# 6. 初始化数据库 (SQLite 默认)
alembic upgrade head

# 7. 启动 Redis (若未运行)
redis-server

# 8. 启动 Celery Worker (新终端)
celery -A app.celery_app worker --loglevel=info

# 9. 启动 FastAPI 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问以下地址查看效果：
- API 交互文档: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 监控仪表盘: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

---

## 🖼️ 界面预览

| 实时监控界面 | 历史数据看板 |
| :---: | :---: |
| ![实时监控](docs/live-demo.png) | ![数据看板](docs/dashboard.png) |

| 告警记录列表 | 满溢检测特写 |
| :---: | :---: |
| ![告警列表](docs/alert-list.png) | ![满溢检测](docs/overflow-demo.jpg) |


---

---

## 🧪 模型训练与导出

### 训练命令 (使用 Ultralytics)

```bash
# 垃圾满溢检测 (需准备 COCO 格式或 YOLO 格式数据集)
yolo train model=yolov8n.pt data=configs/overflow.yaml epochs=100 imgsz=640

# 火焰烟雾检测
yolo train model=yolov8s.pt data=configs/fire_smoke.yaml epochs=120 imgsz=640
```

### 导出为 ONNX 格式 (用于部署推理)

```bash
yolo export model=runs/train/exp/weights/best.pt format=onnx opset=12 simplify
```

导出的 `best.onnx` 放置于 `models/` 目录，供推理引擎加载。

---

## 📊 数据集与模型精度

- **垃圾满溢检测**: 自建数据集，包含 5000+ 张不同光照、角度下的垃圾桶图像，标注满溢/未满溢二分类及分割掩码。
- **火焰烟雾检测**: 公开数据集  增强后共 12000+ 张。
- **模型精度 (ONNX 推理验证)**:
  - 满溢检测 mAP@0.5: **92.7%**
  - 火焰检测 mAP@0.5: **95.3%**
  - 烟雾检测 mAP@0.5: **89.1%**
- **推理速度**: ONNX Runtime CPU 约 25ms/帧，GPU (CUDA) 约 6ms/帧。


---

## 📝 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎用于学习、研究及非商业用途。  

---

## 📮 联系我们

- 📧 Email: `2411919245@QQ.COM`
- 🐙 GitHub Issues: [提交反馈](https://github.com/Nyzeep/HuaLi_garbage_system/issues)

<div align="center">
  <br>
  <strong>🌟 如果这个项目对您有帮助，请给一个 Star 支持我们！ 🌟</strong>
</div>
```

---
