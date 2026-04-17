#!/usr/bin/env python3
"""
从COCO格式标注中提取smoke类别，创建只包含smoke标注的YOLO格式数据集
"""
import json
import os
import shutil
from pathlib import Path

# COCO JSON文件
coco_json_path = r"C:/Users/24039/Downloads/111.json"
# 图片源目录
images_source = r"C:/Users/24039/Downloads"
# 输出目录
output_dir = r"d:/garbage_system/dataset_smoke_5images_new"

# 图片文件映射 (COCO中的filename -> 实际文件名)
image_mapping = {
    "111.png": "111.png",
    "222.jpg": "222.jpg",
    "333.jpg": "333.jpg",
    "444.jpg": "444.jpg",
    "555.jpg": "555.jpg"
}

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """COCO bbox [x, y, w, h] -> YOLO格式 [x_center, y_center, w, h] (归一化)"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm

def main():
    # 读取COCO标注
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 创建输出目录
    img_out_dir = Path(output_dir) / "images"
    label_out_dir = Path(output_dir) / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # 建立image_id到图片信息的映射
    image_info = {img['id']: img for img in coco_data['images']}

    # 按图片分组标注 (只取smoke类别 category_id=2)
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if ann['image_id'] not in annotations_by_image:
            annotations_by_image[ann['image_id']] = []
        # 只保留smoke标注
        if ann['category_id'] == 2:  # smoke
            annotations_by_image[ann['image_id']].append(ann)

    # 处理每张图片
    processed_count = 0
    for img_id, img_data in image_info.items():
        file_name = img_data['file_name']
        width = img_data['width']
        height = img_data['height']

        if file_name not in image_mapping:
            continue

        # 复制图片
        src_img = Path(images_source) / file_name
        dst_img = img_out_dir / file_name
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            print(f"Copied image: {file_name}")

        # 写入标注 (只包含smoke类别，转为class 0)
        anns = annotations_by_image.get(img_id, [])
        label_file = label_out_dir / (Path(file_name).stem + ".txt")

        with open(label_file, 'w') as f:
            for ann in anns:
                bbox = ann['bbox']
                x_c, y_c, w, h = convert_bbox_coco_to_yolo(bbox, width, height)
                # smoke在COCO中是category_id=2，转换为YOLO的class 0
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        if anns:
            print(f"Created label for {file_name} with {len(anns)} smoke annotations")
            processed_count += 1
        else:
            print(f"Warning: No smoke annotations for {file_name}")

    print(f"\n处理完成! 共处理 {processed_count} 张图片")

    # 创建YAML配置文件
    yaml_content = f"""# YOLOv8 烟雾检测数据集（5张新图片）
# 只包含smoke类别的标注

path: {output_dir.replace(chr(92), '/')}
train: images

nc: 1
names: ['smoke']
"""
    yaml_path = Path(output_dir) / "dataset_smoke_5images.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n创建YAML配置: {yaml_path}")

if __name__ == "__main__":
    main()
