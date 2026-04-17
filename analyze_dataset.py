"""分析 VOC XML 数据集标签分布"""
import os
import xml.etree.ElementTree as ET
from collections import Counter

xml_dir = r'D:\garbage_system\dataset\annotations'

counter = Counter()
multi_label = 0
total_imgs = 0

for fn in os.listdir(xml_dir):
    if not fn.endswith('.xml'):
        continue
    try:
        tree = ET.parse(os.path.join(xml_dir, fn))
        objs = tree.findall('object')
        total_imgs += 1
        if len(objs) > 1:
            multi_label += 1
        for obj in objs:
            name = obj.find('name').text
            counter[name] += 1
    except Exception as e:
        print(f"Error: {fn} - {e}")

print('=== 标签类别统计 ===')
for k, v in counter.most_common():
    print(f'  {k}: {v}')
print(f'总标注框数: {sum(counter.values())}')
print(f'多目标图片数: {multi_label}')
print(f'总图片数: {total_imgs}')
