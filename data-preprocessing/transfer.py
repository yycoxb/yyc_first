import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class YOLOMosaicGenerator:
    """
    YOLO格式数据集的Mosaic拼接生成器
    简单按顺序将4张图拼接成2x2网格,不重复使用图片
    """
    
    def __init__(self, 
                 source_images_dir,
                 source_labels_dir,
                 output_images_dir,
                 output_labels_dir):
        """
        参数:
            source_images_dir: 原始图像目录
            source_labels_dir: 原始标签目录(YOLO格式txt)
            output_images_dir: 输出图像目录
            output_labels_dir: 输出标签目录
        """
        self.source_images_dir = Path(source_images_dir)
        self.source_labels_dir = Path(source_labels_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # 创建输出目录
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        self.image_files = self._get_image_files()
        
    def _get_image_files(self):
        """获取所有图像文件"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        for ext in extensions:
            image_files.extend(list(self.source_images_dir.glob(f'*{ext}')))
        
        # 去重(防止同一文件被多次添加)
        image_files = list(set(image_files))
        
        # 排序并打印统计信息
        image_files = sorted(image_files)
        print(f"\n[文件统计]")
        print(f"找到图像文件: {len(image_files)} 张")
        
        # 按扩展名统计
        from collections import Counter
        ext_count = Counter([f.suffix.lower() for f in image_files])
        for ext, count in sorted(ext_count.items()):
            print(f"  {ext}: {count} 张")
        
        return image_files
    
    def _read_yolo_label(self, label_path):
        """
        读取YOLO格式标签
        返回: list of [class_id, x_center, y_center, width, height] (归一化坐标)
        """
        if not label_path.exists():
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
        return labels
    
    def _write_yolo_label(self, label_path, labels):
        """写入YOLO格式标签"""
        with open(label_path, 'w') as f:
            for label in labels:
                class_id, x_center, y_center, width, height = label
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def _convert_label_to_mosaic(self, labels, img_h, img_w, position, mosaic_h, mosaic_w):
        """
        将单张图的YOLO标签转换到mosaic图中的坐标
        
        参数:
            labels: 原始标签 [[class_id, x_center, y_center, width, height], ...]
            img_h, img_w: 原始图像尺寸
            position: 图像在mosaic中的位置 (row, col)
                     (0,0)=左上, (0,1)=右上, (1,0)=左下, (1,1)=右下
            mosaic_h, mosaic_w: mosaic图像总尺寸
        
        返回:
            转换后的标签(归一化到mosaic尺寸)
        """
        if not labels:
            return []
        
        row, col = position
        converted_labels = []
        
        for label in labels:
            class_id, x_center, y_center, width, height = label
            
            # 将归一化坐标转换为原图像素坐标
            x_pixel = x_center * img_w
            y_pixel = y_center * img_h
            w_pixel = width * img_w
            h_pixel = height * img_h
            
            # 计算在mosaic中的偏移
            offset_x = col * img_w
            offset_y = row * img_h
            
            # 新的像素坐标(在mosaic中)
            new_x_pixel = x_pixel + offset_x
            new_y_pixel = y_pixel + offset_y
            
            # 归一化到mosaic尺寸
            new_x_center = new_x_pixel / mosaic_w
            new_y_center = new_y_pixel / mosaic_h
            new_width = w_pixel / mosaic_w
            new_height = h_pixel / mosaic_h
            
            # 边界检查:确保在[0,1]范围内
            if (0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and
                new_width > 0 and new_height > 0):
                converted_labels.append([
                    class_id, 
                    new_x_center, 
                    new_y_center, 
                    new_width, 
                    new_height
                ])
        
        return converted_labels
    
    def create_mosaic(self, image_paths):
        """
        创建一个2x2 mosaic图像
        
        拼接布局:
        [img1] [img2]
        [img3] [img4]
        
        参数:
            image_paths: 4张图像的路径列表
        
        返回:
            mosaic_image: 拼接后的图像
            mosaic_labels: 拼接后的标签
        """
        if len(image_paths) != 4:
            raise ValueError(f"需要4张图像,但提供了{len(image_paths)}张")
        
        # 读取第一张图确定尺寸
        first_img = cv2.imread(str(image_paths[0]))
        if first_img is None:
            raise ValueError(f"无法读取图像: {image_paths[0]}")
        
        img_h, img_w = first_img.shape[:2]
        mosaic_h = img_h * 2
        mosaic_w = img_w * 2
        
        # 创建mosaic画布
        mosaic_image = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_labels = []
        
        # 拼接位置: (row, col)
        positions = [
            (0, 0),  # 左上
            (0, 1),  # 右上
            (1, 0),  # 左下
            (1, 1)   # 右下
        ]
        
        # 拼接图像和标签
        for idx, (img_path, (row, col)) in enumerate(zip(image_paths, positions)):
            # 读取图像
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"警告: 无法读取图像 {img_path}, 使用黑色填充")
                img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            
            # 调整图像尺寸(如果不一致)
            if img.shape[:2] != (img_h, img_w):
                img = cv2.resize(img, (img_w, img_h))
            
            # 放置图像到对应位置
            y_start = row * img_h
            y_end = (row + 1) * img_h
            x_start = col * img_w
            x_end = (col + 1) * img_w
            mosaic_image[y_start:y_end, x_start:x_end] = img
            
            # 读取并转换标签
            label_path = self.source_labels_dir / (img_path.stem + '.txt')
            labels = self._read_yolo_label(label_path)
            converted = self._convert_label_to_mosaic(
                labels, img_h, img_w, (row, col), mosaic_h, mosaic_w
            )
            mosaic_labels.extend(converted)
        
        return mosaic_image, mosaic_labels
    
    def generate_dataset(self):
        """
        生成mosaic数据集
        按顺序每4张图拼接成1张mosaic,不重复使用
        """
        total_images = len(self.image_files)
        num_mosaics = total_images // 4
        remaining = total_images % 4
        
        print("\n" + "=" * 60)
        print(f"原始数据集: {total_images} 张图像")
        print(f"将生成: {num_mosaics} 张 mosaic 图像 (2x2拼接)")
        print(f"每张mosaic使用: 4 张原始图像")
        if remaining > 0:
            print(f"⚠️  剩余: {remaining} 张图像无法凑成4张,将被跳过")
        print("\n拼接布局:")
        print("  [图1] [图2]")
        print("  [图3] [图4]")
        print("=" * 60)
        
        # 确认是否继续
        if num_mosaics == 0:
            print("\n❌ 错误: 图像数量少于4张,无法生成mosaic!")
            return
        
        # 生成mosaic
        success_count = 0
        for i in tqdm(range(num_mosaics), desc="生成Mosaic数据集"):
            # 按顺序取4张图
            start_idx = i * 4
            selected_images = self.image_files[start_idx:start_idx + 4]
            
            try:
                # 创建mosaic
                mosaic_img, mosaic_labels = self.create_mosaic(selected_images)
                
                # 保存图像
                output_img_name = f"mosaic_{i:06d}.jpg"
                output_img_path = self.output_images_dir / output_img_name
                cv2.imwrite(str(output_img_path), mosaic_img)
                
                # 保存标签
                output_label_name = f"mosaic_{i:06d}.txt"
                output_label_path = self.output_labels_dir / output_label_name
                self._write_yolo_label(output_label_path, mosaic_labels)
                
                success_count += 1
                
            except Exception as e:
                print(f"\n❌ 错误: 生成第{i}个mosaic时失败: {e}")
                # 打印出错的图像文件名
                print(f"   涉及的图像: {[img.name for img in selected_images]}")
                continue
        
        print("\n" + "=" * 60)
        print(f"✅ 完成! 成功生成 {success_count} 张mosaic图像")
        print(f"📁 图像保存在: {self.output_images_dir}")
        print(f"📁 标签保存在: {self.output_labels_dir}")
        print(f"\n验证:")
        print(f"  原始图像使用: {success_count * 4} / {total_images} 张")
        print(f"  利用率: {success_count * 4 / total_images * 100:.1f}%")
        print("=" * 60)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    使用方法:
    
    1. 确保你的YOLO数据集结构如下:
       dataset/
       ├── images/
       │   ├── img1.jpg
       │   ├── img2.jpg
       │   ├── img3.jpg
       │   ├── img4.jpg
       │   └── ...
       └── labels/
           ├── img1.txt
           ├── img2.txt
           ├── img3.txt
           ├── img4.txt
           └── ...
    
    2. 修改下面的路径
    3. 运行脚本
    
    拼接规则:
    - 按文件名顺序,每4张图拼接成1张
    - 布局: [图1][图2]
            [图3][图4]
    - 不重复使用任何图片
    - 如果图片数量不是4的倍数,剩余的会被跳过
    """
    
    # ========== 配置参数 ==========
    
    # 原始数据集路径
    SOURCE_IMAGES_DIR = "data/val/images"  # 修改为你的路径
    SOURCE_LABELS_DIR = "data/val/labels"  # 修改为你的路径
    
    # 输出路径
    OUTPUT_IMAGES_DIR = "dataset_mosaic/val/images"
    OUTPUT_LABELS_DIR = "dataset_mosaic/val/labels"
    
    # ========== 生成Mosaic数据集 ==========
    
    generator = YOLOMosaicGenerator(
        source_images_dir=SOURCE_IMAGES_DIR,
        source_labels_dir=SOURCE_LABELS_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        output_labels_dir=OUTPUT_LABELS_DIR
    )
    
    # 生成mosaic数据集
    generator.generate_dataset()
    
    print("\n使用说明:")
    print("1. 如果要用于训练YOLO,需要创建对应的yaml配置文件")
    print("2. 如果要用于评估,可以直接使用生成的数据集")
    print("3. 原始图片顺序: img1, img2, img3, img4 → mosaic_000000.jpg")
    print("                 img5, img6, img7, img8 → mosaic_000001.jpg")
    print("                 ...")