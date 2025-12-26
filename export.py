import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('best.pt')
    model.export(format='onnx', simplify=True, opset=13)

    onnx_model = YOLO("best.onnx")




    results = onnx_model("E:\exacvator_nostrengthen_little\images/train\exacvator_2023-11-24-0001.jpg")