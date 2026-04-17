"""
YOLOv8 烟火检测训练脚本（云 GPU 版本）
同时训练火检测模型和烟检测模型

使用方式:
  py -3 train_fire_smoke_cloud.py
"""
import os
import shutil

# ===================== 训练配置 =====================
WORKSPACE_DIR = '/root/workspace'

# 火检测数据集
FIRE_DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset_fire')
FIRE_YAML_PATH = os.path.join(FIRE_DATASET_DIR, 'dataset_fire.yaml')
FIRE_MODEL_OUTPUT = os.path.join(WORKSPACE_DIR, 'fire_yolov8.pt')

# 烟检测数据集
SMOKE_DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset_smoke_5images_new')
SMOKE_YAML_PATH = os.path.join(SMOKE_DATASET_DIR, 'dataset_smoke_5images.yaml')
SMOKE_MODEL_OUTPUT = os.path.join(WORKSPACE_DIR, 'smoke_yolov8.pt')

# 训练输出目录
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'runs', 'detect')


def check_env():
    """检查环境"""
    import torch
    print("=" * 50)
    print("  环境检测")
    print("=" * 50)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return True


def check_and_fix_yaml(yaml_path, dataset_dir):
    """检查并修正YAML路径"""
    import re
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML不存在: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_text = f.read()
    
    yaml_text = re.sub(r'^path:.*$', f'path: {dataset_dir}', yaml_text, flags=re.MULTILINE)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_text)
    
    print(f"  ✅ YAML已修正: {yaml_path}")


def train_model(data_yaml, model_name, output_path):
    """训练单个模型"""
    from ultralytics import YOLO
    import torch
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n训练设备: {'GPU' if device == 0 else 'CPU'}")
    
    print(f"\n开始训练 {model_name}...")
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=768,
        batch=16,
        device=device,
        workers=2,
        project=OUTPUT_DIR,
        name=model_name,
        exist_ok=True,
        patience=40,
        save=True,
        plots=True,
        augment=True,
        degrees=10,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        lr0=0.0005,
        lrf=0.01
    )
    
    # 复制到固定路径
    best_model = os.path.join(OUTPUT_DIR, model_name, 'weights', 'best.pt')
    if os.path.exists(best_model):
        shutil.copy2(best_model, output_path)
        print(f"✅ 模型已保存: {output_path}")
    
    return results


def main():
    print("=" * 50)
    print("  YOLOv8 烟火检测训练（云 GPU 版）")
    print("=" * 50)
    
    # 检查环境
    check_env()
    
    # 检查数据集
    print("\n检查数据集...")
    check_and_fix_yaml(FIRE_YAML_PATH, FIRE_DATASET_DIR)
    check_and_fix_yaml(SMOKE_YAML_PATH, SMOKE_DATASET_DIR)
    
    # 1. 训练火检测模型
    print("\n" + "=" * 50)
    print("  训练火检测模型")
    print("=" * 50)
    train_model(FIRE_YAML_PATH, 'fire_train', FIRE_MODEL_OUTPUT)
    
    # 2. 训练烟检测模型
    print("\n" + "=" * 50)
    print("  训练烟检测模型")
    print("=" * 50)
    train_model(SMOKE_YAML_PATH, 'smoke_train', SMOKE_MODEL_OUTPUT)
    
    # 汇总
    print("\n" + "=" * 50)
    print("  训练完成！")
    print("=" * 50)
    print(f"🔥 火模型: {FIRE_MODEL_OUTPUT}")
    print(f"💨 烟模型: {SMOKE_MODEL_OUTPUT}")
    print("\n请下载以上两个模型文件！")


if __name__ == '__main__':
    main()
