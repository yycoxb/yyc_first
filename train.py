import os
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 然后再 import numpy, torch, onnxruntime ...
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 指定显卡和多卡训练问题 统一都在<YOLOV11配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。

if __name__ == '__main__':
    model = YOLO('yolo11n-pose.pt')
    #model = YOLO('weights/last.pt')
    #model.load('weights/yolo11s.pt') # loading pretrain weights
    model.train(data='crowd_pose.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                close_mosaic=0,
                workers=8,
                # device='0',
                optimizer='SGD', # using SGD,Adam
                # patience=0, # close earlystop
                #resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='crowd_pose_yolo11n',
               #freeze=10
                )