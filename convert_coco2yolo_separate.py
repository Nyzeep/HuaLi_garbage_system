#!/usr/bin/env python3
"""
将 COCO JSON 标注转换成 YOLO 格式，并分离火和烟雾为两个独立数据集
"""

import json
import os
import glob
import shutil
from pathlib import Path

def convert_coco_to_yolo(json_path, image_folder, output_base, target_category=None):
    """
    将单个 COCO JSON 文件转换为 YOLO 格式，可选择只提取特定类别
    
    Args:
        json_path: COCO JSON 文件路径
        image_folder: 图片文件夹路径
        output_base: 输出数据集根目录
        target_category: 只提取的类别名称（如 'fire' 或 'smoke'），None 表示全部
    """
    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建类别映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_name_to_id = {cat['name']: i for i, cat in enumerate(data['categories'])}
    
    print(f"找到 {len(data['categories'])} 个类别: {list(cat_id_to_name.values())}")
    print(f"共 {len(data['images'])} 张图片")
    
    # 创建输出目录结构
    output_images_dir = os.path.join(output_base, 'images')
    output_labels_dir = os.path.join(output_base, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 创建数据集配置文件
    config_path = os.path.join(output_base, f"{os.path.basename(output_base)}.yaml")
    
    # 收集所有图片的标注
    image_map = {img['id']: img for img in data['images']}
    
    for img_info in data['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 查找这张图的所有标注
        img_annots = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        # 如果是单类别提取，只保留目标类别的标注
        if target_category is not None:
            img_annots = [
                ann for ann in img_annots 
                if cat_id_to_name[ann['category_id']] == target_category
            ]
        
        # 如果没有符合条件的标注，跳过
        if not img_annots:
            continue
        
        # 复制图片
        src_img_path = os.path.join(image_folder, img_filename)
        dst_img_path = os.path.join(output_images_dir, img_filename)
        
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"警告: 图片不存在 {src_img_path}")
            continue
        
        # 生成 YOLO 标注文件
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_filename)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in img_annots:
                cat_id = ann['category_id']
                cat_name = cat_id_to_name[cat_id]
                
                # YOLO 类别 ID（从0开始）
                # 如果是单类别数据集，类别 ID 总是 0
                if target_category is not None:
                    yolo_class_id = 0  # 单类别数据集
                else:
                    yolo_class_id = list(cat_name_to_id.keys()).index(cat_name)
                
                # COCO bbox 格式: [x_min, y_min, width, height]
                bbox = ann['bbox']
                x_min, y_min, w, h = bbox
                
                # 转 YOLO 格式: [x_center, y_center, width, height] (归一化 0-1)
                x_center = (x_min + w / 2) / img_width
                y_center = (y_min + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                # 写入 YOLO 行
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # 创建 YAML 配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        if target_category is not None:
            # 单类别数据集
            f.write(f"# YOLOv8 单类别数据集 - {target_category}\n")
            f.write(f"# 由 convert_coco2yolo_separate.py 自动生成\n\n")
            f.write(f"path: {os.path.abspath(output_base)}\n")
            f.write(f"train: images\n\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['{target_category}']\n")
        else:
            # 多类别数据集
            f.write(f"# YOLOv8 多类别数据集\n")
            f.write(f"# 由 convert_coco2yolo_separate.py 自动生成\n\n")
            f.write(f"path: {os.path.abspath(output_base)}\n")
            f.write(f"train: images\n\n")
            f.write(f"nc: {len(data['categories'])}\n")
            f.write(f"names: {[cat['name'] for cat in data['categories']]}\n")
    
    print(f"完成! 数据集保存到: {output_base}")
    print(f"配置文件: {config_path}")
    print(f"图片数: {len(os.listdir(output_images_dir))}")
    print(f"标注数: {len(os.listdir(output_labels_dir))}")

def main():
    # 配置参数
    json_files = [
        "C:/Users/24039/Downloads/111.json",
        "C:/Users/24039/Downloads/222.json",
        "C:/Users/24039/Downloads/333.json",
        "C:/Users/24039/Downloads/444.json",
        "C:/Users/24039/Downloads/555.json"
    ]
    
    # 假设图片在下载文件夹
    image_folder = "C:/Users/24039/Downloads"
    
    # 创建独立数据集
    # 1. 火数据集
    print("="*50)
    print("正在创建火数据集...")
    print("="*50)
    fire_output = "d:/garbage_system/dataset_fire_new"
    
    # 合并所有 JSON 文件（只提取 fire 类别）
    combined_data = {
        "info": {"description": "fire-only dataset"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fire"}]
    }
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 添加图片信息
        for img in data['images']:
            # 检查这张图是否有 fire 标注
            img_annots = [ann for ann in data['annotations'] 
                         if ann['image_id'] == img['id'] 
                         and data['categories'][ann['category_id']-1]['name'] == 'fire']
            
            if img_annots:
                # 为新数据集分配新 ID
                new_img_id = len(combined_data['images']) + 1
                img_copy = img.copy()
                img_copy['id'] = new_img_id
                combined_data['images'].append(img_copy)
                
                # 添加标注
                for ann in img_annots:
                    new_ann = ann.copy()
                    new_ann['id'] = len(combined_data['annotations']) + 1
                    new_ann['image_id'] = new_img_id
                    new_ann['category_id'] = 1  # 在 fire-only 数据集中 fire 始终是类别 1
                    combined_data['annotations'].append(new_ann)
    
    # 保存合并后的 JSON
    temp_json = "d:/garbage_system/temp_fire.json"
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    # 转换
    convert_coco_to_yolo(temp_json, image_folder, fire_output, target_category='fire')
    
    # 2. 烟雾数据集
    print("\n" + "="*50)
    print("正在创建烟雾数据集...")
    print("="*50)
    smoke_output = "d:/garbage_system/dataset_smoke"
    
    combined_data = {
        "info": {"description": "smoke-only dataset"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "smoke"}]
    }
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for img in data['images']:
            img_annots = [ann for ann in data['annotations'] 
                         if ann['image_id'] == img['id'] 
                         and data['categories'][ann['category_id']-1]['name'] == 'smoke']
            
            if img_annots:
                new_img_id = len(combined_data['images']) + 1
                img_copy = img.copy()
                img_copy['id'] = new_img_id
                combined_data['images'].append(img_copy)
                
                for ann in img_annots:
                    new_ann = ann.copy()
                    new_ann['id'] = len(combined_data['annotations']) + 1
                    new_ann['image_id'] = new_img_id
                    new_ann['category_id'] = 1  # smoke 始终是类别 1
                    combined_data['annotations'].append(new_ann)
    
    temp_json = "d:/garbage_system/temp_smoke.json"
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    convert_coco_to_yolo(temp_json, image_folder, smoke_output, target_category='smoke')
    
    # 清理临时文件
    os.remove("d:/garbage_system/temp_fire.json")
    os.remove("d:/garbage_system/temp_smoke.json")
    
    print("\n" + "="*50)
    print("完成！")
    print("- 火数据集: d:/garbage_system/dataset_fire_new")
    print("- 烟雾数据集: d:/garbage_system/dataset_smoke")
    print("="*50)

if __name__ == "__main__":
    main()