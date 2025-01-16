import onnx
from onnxsim import simplify

# 加载动态ONNX模型
model = onnx.load("rope_batch01_fp16.onnx")

# 简化并固定动态维度
model_simp, check = simplify(model, overwrite_input_shapes={'images_float32':[1, 3, 544, 960]})

# 保存静态ONNX模型
onnx.save(model_simp, "rope_static_fp16.onnx")