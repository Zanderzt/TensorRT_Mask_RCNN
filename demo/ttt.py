import torch
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os, sys
import cv2
import datetime
import json
from tqdm import tqdm
import glob


config_file = '../configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
opts = [# "MODEL.WEIGHT",
        #  "./model_0.pth",
         "MODEL.DEVICE", "cpu"]
# load config from file and command-line arguments
cfg.merge_from_file(config_file)
cfg.merge_from_list(opts)
cfg.freeze()

model = COCODemo(cfg,
                 min_image_size=800,
                 # confidence_threshold=0.7,
                 show_mask_heatmaps=False)  # 0-30
model.model.backbone.eval()

trace_backbone = torch.jit.trace(model.model.backbone, (torch.rand([2, 3, 800, 800])))
output = model.model.backbone(torch.rand([4, 3, 800, 800]))
import torch.onnx as onnx
onnx._export(trace_backbone, torch.randn(4, 3, 800, 800), './temp.pb',
             verbose=False, export_params=True, training=False,
            example_outputs=output)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network() as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open('./mask_backbone_fpn.onnx', 'rb') as model:
        a = parser.parse(model.read())
        print(a)
    print('Building an engine from file temp.pb; this may take a while...')
    engine = builder.build_cuda_engine(network)
    print("Completed creating Engine")

inputs = []
outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    print(binding)
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))

    # 分配host和device端的buffer
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    # 将device端的buffer追加到device的bindings.
    bindings.append(int(device_mem))

    # Append to the appropriate list.
    if engine.binding_is_input(binding):
        inputs.append((host_mem, device_mem))
    else:
        outputs.append((host_mem, device_mem))

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    '''
    :param context:
    :param bindings:
    :param inputs:
    :param outputs:
    :param stream:
    :param batch_size:
    :return: out from TensorRT engine inference
    '''
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1],stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]

from predictor import COCODemo
# Test
from tqdm import tqdm_notebook as tqdm
import numpy as np
for i in tqdm(range(10000)):
    with engine.create_execution_context() as context:
        np.copyto(inputs[0][0], np.ones([448, 768, 3]).reshape([1032192]))
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = np.array(output)


