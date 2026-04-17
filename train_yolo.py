"""
YOLOv8 训练脚本 - 社区垃圾分类检测系统
基于真实数据集（3349张图，3类目标）训练

使用前安装:
  pip install ultralytics

运行:
  py -3 D:\garbage_system\train_yolo.py
"""
import os
import subprocess
import sys

# ===================== 训练配置 =====================
DATASET_YAML = r'D:/garbage_system/dataset/dataset.yaml'
MODEL_OUT    = r'D:/garbage_system/app/models'
RUN_NAME     = 'garbage_yolov8'

# YOLOv8 训练超参数（根据算力调整）
TRAIN_CONFIG = {
    'model':    'yolov8n.pt',   # nano 模型（轻量，适合入门）；也可换 yolov8s.pt
    'data':     DATASET_YAML,
    'epochs':   100,
    'imgsz':    640,
    'batch':    16,             # 显存不足时改为 8
    'device':   0,              # 0=GPU，'cpu'=CPU（无显卡时用 cpu）
    'workers':  4,
    'project':  MODEL_OUT,
    'name':     RUN_NAME,
    'exist_ok': True,
    'patience': 20,             # 20 轮无提升则早停
    'save':     True,
    'plots':    True,
    'cache':    False,          # 内存充足时改 True 加速
    'augment':  True,           # 数据增强
    'degrees':  10,             # 旋转增强
    'flipud':   0.3,            # 上下翻转概率
    'fliplr':   0.5,            # 左右翻转概率
    'mosaic':   0.8,            # mosaic 增强
    'mixup':    0.1,            # mixup 增强
}

# ===================== 验证环境 =====================

def check_env():
    """检查环境是否满足训练条件"""
    print("=" * 55)
    print("  YOLOv8 训练环境检测")
    print("=" * 55)

    # 检查 ultralytics
    try:
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[ERROR] ultralytics 未安装，正在安装...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', '-q'])
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__} 安装完成")

    # 检查 torch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"[OK] PyTorch {torch.__version__}, CUDA={cuda_ok}")
        if not cuda_ok:
            print("[提示] 未检测到 GPU，将使用 CPU 训练（速度较慢，建议改 device='cpu'）")
            TRAIN_CONFIG['device'] = 'cpu'
            TRAIN_CONFIG['batch'] = 8
    except ImportError:
        print("[提示] PyTorch 未安装，ultralytics 会自动处理")

    # 检查数据集
    if not os.path.exists(DATASET_YAML):
        print(f"[ERROR] 数据集不存在: {DATASET_YAML}")
        print("请先运行: py -3 D:\\garbage_system\\convert_voc2yolo.py")
        sys.exit(1)
    else:
        train_count = len(os.listdir(os.path.join(os.path.dirname(DATASET_YAML), 'train', 'images')))
        val_count = len(os.listdir(os.path.join(os.path.dirname(DATASET_YAML), 'val', 'images')))
        print(f"[OK] 数据集: train={train_count}, val={val_count}")

    os.makedirs(MODEL_OUT, exist_ok=True)
    print(f"[OK] 模型输出目录: {MODEL_OUT}")
    return True


# ===================== 开始训练 =====================

def train():
    """执行 YOLOv8 训练"""
    from ultralytics import YOLO

    print("\n" + "=" * 55)
    print("  开始训练 - 社区垃圾分类检测模型")
    print("=" * 55)

    model = YOLO(TRAIN_CONFIG.pop('model'))

    print(f"配置参数: {TRAIN_CONFIG}")
    print()

    results = model.train(**TRAIN_CONFIG)

    # 最佳模型路径
    best_model = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')
    system_model = os.path.join(MODEL_OUT, 'garbage_yolov8.pt')

    if os.path.exists(best_model):
        import shutil
        shutil.copy2(best_model, system_model)
        print(f"\n[完成] 最佳模型已复制至系统路径: {system_model}")
        print(f"[提示] 重启系统后将自动加载真实模型")
    else:
        print(f"[警告] 最佳模型未找到: {best_model}")

    return results


# ===================== 验证 =====================

def validate():
    """验证已训练模型的性能"""
    from ultralytics import YOLO

    model_path = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print("[ERROR] 模型不存在，请先训练")
        return

    print("\n" + "=" * 55)
    print("  模型验证")
    print("=" * 55)
    model = YOLO(model_path)
    metrics = model.val(data=DATASET_YAML)
    print(f"\nmAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv8 垃圾检测训练')
    parser.add_argument('--mode', choices=['train', 'val'], default='train')
    args = parser.parse_args()

    check_env()
    if args.mode == 'train':
        train()
    else:
        validate()
