"""
同时使用火和烟雾检测模型进行推理
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class FireSmokeDetector:
    def __init__(self, fire_model_path=None, smoke_model_path=None):
        """
        初始化火和烟雾检测器
        
        Args:
            fire_model_path: 火检测模型路径 (如果不提供则使用默认模型)
            smoke_model_path: 烟雾检测模型路径 (如果不提供则使用默认模型)
        """
        self.fire_model = None
        self.smoke_model = None
        
        # 尝试加载预训练模型
        try:
            if fire_model_path and os.path.exists(fire_model_path):
                print(f"加载火检测模型: {fire_model_path}")
                self.fire_model = YOLO(fire_model_path)
            else:
                print("使用默认YOLOv8模型进行火检测")
                self.fire_model = YOLO('yolov8n.pt')
                
            if smoke_model_path and os.path.exists(smoke_model_path):
                print(f"加载烟雾检测模型: {smoke_model_path}")
                self.smoke_model = YOLO(smoke_model_path)
            else:
                print("使用默认YOLOv8模型进行烟雾检测")
                self.smoke_model = YOLO('yolov8n.pt')
                
        except Exception as e:
            print(f"模型加载失败: {e}")
    
    def detect(self, image_path, conf_threshold=0.3):
        """
        检测图片中的火和烟雾
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            result_image: 带标注的结果图片
            fire_results: 火焰检测结果列表
            smoke_results: 烟雾检测结果列表
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return None, [], []
        
        # 检测火焰
        fire_results = []
        if self.fire_model:
            fire_detections = self.fire_model(img, conf=conf_threshold)[0]
            for box in fire_detections.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # YOLO类别名
                cls_name = "fire"
                
                fire_results.append({
                    'bbox': xyxy,
                    'confidence': conf,
                    'class': cls,
                    'class_name': cls_name
                })
                
                # 在图片上绘制火焰框（红色）
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Fire: {conf:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 检测烟雾
        smoke_results = []
        if self.smoke_model:
            smoke_detections = self.smoke_model(img, conf=conf_threshold)[0]
            for box in smoke_detections.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # YOLO类别名
                cls_name = "smoke"
                
                smoke_results.append({
                    'bbox': xyxy,
                    'confidence': conf,
                    'class': cls,
                    'class_name': cls_name
                })
                
                # 在图片上绘制烟雾框（蓝色）
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"Smoke: {conf:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img, fire_results, smoke_results
    
    def process_folder(self, folder_path, output_folder=None, conf_threshold=0.3):
        """
        处理文件夹中的所有图片
        
        Args:
            folder_path: 输入文件夹路径
            output_folder: 输出文件夹路径
            conf_threshold: 置信度阈值
            
        Returns:
            处理统计信息
        """
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return
        
        if output_folder is None:
            output_folder = os.path.join(folder_path, "detections")
        os.makedirs(output_folder, exist_ok=True)
        
        # 支持的图片格式
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        stats = {
            'total_images': 0,
            'images_with_fire': 0,
            'images_with_smoke': 0,
            'total_fire': 0,
            'total_smoke': 0
        }
        
        for filename in os.listdir(folder_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in image_exts:
                continue
            
            image_path = os.path.join(folder_path, filename)
            print(f"处理图片: {filename}")
            
            result_img, fire_results, smoke_results = self.detect(
                image_path, conf_threshold
            )
            
            if result_img is not None:
                # 保存结果
                output_path = os.path.join(output_folder, f"detected_{filename}")
                cv2.imwrite(output_path, result_img)
                
                # 更新统计
                stats['total_images'] += 1
                if fire_results:
                    stats['images_with_fire'] += 1
                    stats['total_fire'] += len(fire_results)
                if smoke_results:
                    stats['images_with_smoke'] += 1
                    stats['total_smoke'] += len(smoke_results)
                
                print(f"  - 火焰: {len(fire_results)}个, 烟雾: {len(smoke_results)}个")
        
        print("\n处理完成!")
        print(f"总共处理图片: {stats['total_images']}")
        print(f"有火焰的图片: {stats['images_with_fire']}")
        print(f"有烟雾的图片: {stats['images_with_smoke']}")
        print(f"检测到的火焰总数: {stats['total_fire']}")
        print(f"检测到的烟雾总数: {stats['total_smoke']}")
        
        return stats

def main():
    # 创建检测器（训练后可以指定模型路径）
    detector = FireSmokeDetector()
    
    # 测试用单张图片
    test_image = "d:/garbage_system/dataset_fire_5images/images/111.png"
    
    if os.path.exists(test_image):
        print(f"检测图片: {test_image}")
        result_img, fire_results, smoke_results = detector.detect(test_image)
        
        if result_img is not None:
            # 保存结果
            output_path = "d:/garbage_system/fire_smoke_detection_result.jpg"
            cv2.imwrite(output_path, result_img)
            print(f"结果保存到: {output_path}")
            print(f"检测到火焰: {len(fire_results)}个")
            print(f"检测到烟雾: {len(smoke_results)}个")
    else:
        print(f"测试图片不存在: {test_image}")

if __name__ == "__main__":
    main()