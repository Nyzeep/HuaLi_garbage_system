#!/usr/bin/env python3
"""
合并数据集：
1. 将新5张图片的fire标注合并到原dataset_fire（100张 -> 105张）
2. 创建独立的烟雾数据集（5张图片，只有smoke标注）
"""

import json
import os
import shutil

# 配置
original_fire_dir = "d:/garbage_system/dataset_fire"
new_images_folder = "C:/Users/24039/Downloads"
json_files = [
    "C:/Users/24039/Downloads/111.json",
    "C:/Users/24039/Downloads/222.json",
    "C:/Users/24039/Downloads/333.json",
    "C:/Users/24039/Downloads/444.json",
    "C:/Users/24039/Downloads/555.json"
]

print("="*60)
print("第一步：将新5张图片的fire标注合并到原dataset_fire")
print("="*60)

# 读取一个JSON文件（5个内容相同）
with open(json_files[0], 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

print(f"找到类别: {[cat['name'] for cat in coco_data['categories']]}")

# 复制新5张图片和fire标注到原dataset_fire
train_images_dir = os.path.join(original_fire_dir, "train/images")
train_labels_dir = os.path.join(original_fire_dir, "train/labels")

fire_count = 0
for img in coco_data['images']:
    img_id = img['id']
    img_name = img['file_name']
    img_width = img['width']
    img_height = img['height']
    
    src_img_path = os.path.join(new_images_folder, img_name)
    
    if not os.path.exists(src_img_path):
        print(f"跳过: 图片不存在 {img_name}")
        continue
    
    # 收集fire标注
    fire_bboxes = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            # 找到类别名
            cat_name = None
            for cat in coco_data['categories']:
                if cat['id'] == ann['category_id']:
                    cat_name = cat['name']
                    break
            
            if cat_name == 'fire':
                fire_bboxes.append(ann['bbox'])
    
    if fire_bboxes:
        # 复制图片
        dst_img_path = os.path.join(train_images_dir, img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 生成YOLO标注（类别0 = fire）
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_name)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for bbox in fire_bboxes:
                x_min, y_min, w, h = bbox
                x_center = (x_min + w / 2) / img_width
                y_center = (y_min + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        fire_count += 1
        print(f"已添加: {img_name} ({len(fire_bboxes)}个fire标注)")

print(f"\n火数据集更新完成！新增 {fire_count} 张图片")

# 统计新总数
total_images = len([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"火数据集总图片数: {total_images} 张 (原100张 + 新{fire_count}张)")

print("\n" + "="*60)
print("第二步：创建独立的烟雾数据集（5张图片）")
print("="*60)

# 创建烟雾数据集目录
smoke_dir = "d:/garbage_system/dataset_smoke_only"
smoke_images_dir = os.path.join(smoke_dir, "images")
smoke_labels_dir = os.path.join(smoke_dir, "labels")
os.makedirs(smoke_images_dir, exist_ok=True)
os.makedirs(smoke_labels_dir, exist_ok=True)

smoke_count = 0
for img in coco_data['images']:
    img_id = img['id']
    img_name = img['file_name']
    img_width = img['width']
    img_height = img['height']
    
    src_img_path = os.path.join(new_images_folder, img_name)
    
    if not os.path.exists(src_img_path):
        print(f"跳过: 图片不存在 {img_name}")
        continue
    
    # 收集smoke标注
    smoke_bboxes = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            cat_name = None
            for cat in coco_data['categories']:
                if cat['id'] == ann['category_id']:
                    cat_name = cat['name']
                    break
            
            if cat_name == 'smoke':
                smoke_bboxes.append(ann['bbox'])
    
    if smoke_bboxes:
        # 复制图片
        dst_img_path = os.path.join(smoke_images_dir, img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 生成YOLO标注（类别0 = smoke，单类别数据集）
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(smoke_labels_dir, label_name)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for bbox in smoke_bboxes:
                x_min, y_min, w, h = bbox
                x_center = (x_min + w / 2) / img_width
                y_center = (y_min + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        smoke_count += 1
        print(f"已添加: {img_name} ({len(smoke_bboxes)}个smoke标注)")

print(f"\n烟雾数据集创建完成！共 {smoke_count} 张图片")

# 创建烟雾数据集YAML配置文件
yaml_content = f"""# YOLOv8 烟雾检测数据集（5张图片）
# 由 merge_datasets.py 自动生成

path: d:/garbage_system/dataset_smoke_only
train: images

nc: 1
names: ['smoke']
"""

yaml_path = os.path.join(smoke_dir, "dataset_smoke_only.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"配置文件: {yaml_path}")

print("\n" + "="*60)
print("完成！")
print("="*60)
print(f"1. 火数据集: {original_fire_dir}")
print(f"   - 总图片数: {total_images} 张")
print(f"   - 类别: fire (0)")
print(f"   - 配置文件: {original_fire_dir}/dataset_fire.yaml")
print()
print(f"2. 烟雾数据集: {smoke_dir}")
print(f"   - 总图片数: {smoke_count} 张")
print(f"   - 类别: smoke (0)")
print(f"   - 配置文件: {yaml_path}")
print("="*60)