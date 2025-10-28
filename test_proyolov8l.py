# Ultralytics YOLO 🚀, AGPL-3.0 license


from ultralytics import RTDETR, YOLO
import torch
CFG = 'ultralytics/cfg/models/11/yolo11-bifpn_small.yaml'	#使用l模型加一个l字母
SOURCE = r"F:\test\test1_yolov8\data\yunda\images\train\00127.jpg"




def test_model_forward():
    """Test the forward pass of the YOLO model."""


    # 创建模型
    model = YOLO(CFG)



