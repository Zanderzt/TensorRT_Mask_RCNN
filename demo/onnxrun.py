'''

import numpy
import onnxruntime as rt

sess = rt.InferenceSession("mask_backbone_fpn.onnx")
input_name = sess.get_inputs()[0].name
print(sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
X = numpy.random.random((1, 3, 448, 768)).astype(numpy.float32)
pred_onnx = sess.run(None, {input_name: X})
print(len(pred_onnx))
for i in range(len(pred_onnx)):
    print(pred_onnx[i].shape)

'''

import numpy as np
x1 = np.random.random((1, 3, 448, 768))
