import onnx_tensorrt.backend as backend
import onnx
import numpy as np

model = onnx.load("./temp.onnx")
onnx.checker.check_model(model)
engine = backend.prepare(model, device='CUDA:0')
input_data = np.random.random(size=(1, 3, 800, 800)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)