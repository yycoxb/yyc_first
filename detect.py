
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r"E:\wajueji_4k_results\yolo11s\weights\best.pt") # select your model.pt path
    model.predict(source=r"D:\BaiduNetdiskDownload\excavator.v1i.yolov11\data_mosaic3\val\images",
                  imgsz=640,
                  project=r"E:\CSMEF-YOLO-test\yyc",
                  name='dbb',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )