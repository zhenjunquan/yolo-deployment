import onnx
from ultralytics import YOLO

# jeston 部署需要 先导出opset=11
model = YOLO("yolo11nt.pt")
model.export(format="onnx", opset=11)

# 然后将 ir_version 修改为 7
model_path = 'yolo11n.onnx'
model = onnx.load(model_path)
print(f"Current IR version: {model.ir_version}")

new_ir_version = 7
model.ir_version = new_ir_version

updated_model_path = 'yolo11n.onnx'
onnx.save(model, updated_model_path)

print(f"IR version updated to: {new_ir_version}")
