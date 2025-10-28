import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('best.pt')
    model.export(format='onnx', simplify=True, opset=13)

    onnx_model = YOLO("best.onnx")



    # # Load the exported ONNX model
    # onnx_model = YOLO("best.onnx")
    # #
    # # Run inference
    results = onnx_model("E:\exacvator_nostrengthen_little\images/train\exacvator_2023-11-24-0001.jpg")