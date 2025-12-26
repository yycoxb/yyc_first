import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 然后再 import numpy, torch, onnxruntime ...
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r"E:\wajueji_4k_results\yolo11s\weights\best.pt") # 选择训练好的权重路径
    model.val(data='excavator-test.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=2,
              # iou=0.7,
              #rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              #name='excavator-mosaic3_yolo11s_asff_c3k2_mseis',
              )