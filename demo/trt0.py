import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
from timeit import default_timer as timer
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network() as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = (1 << 30) * 4
    with open('./mask_backbone_fpn.onnx', 'rb') as model:
        a = parser.parse(model.read())
        print(a)
    # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
    print('Building an engine from file mask_backbone_fpn.onnx; this may take a while...')
    engine = builder.build_cuda_engine(network)
    print("Completed creating Engine")
    for binding in engine:
        print('binding:', binding)
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print("size", size)



inputs = []
outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
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
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]

'''
#调用代码
from tqdm import tqdm_notebook as tqdm
import numpy as np
import cv2
image = cv2.imread("./a1.jpg")
image = cv2.resize(image, (448, 768))
for i in (range(10000)):
    with engine.create_execution_context() as context:
        # np.copyto(inputs[0][0], np.ones([128, 128, 3]).reshape([49152]))
        np.copyto(inputs[0][0], image.reshape([1032192]))
        # np.copyto(inputs[0][0], image.reshape(1200000))
        start = timer()
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = np.array(output)
        xcv4 = pred[0].reshape([1, 256, 14, 24])
        xcv4 = torch.tensor(xcv4).cuda()
        xcv3 = pred[1].reshape([1, 256, 28, 48])
        xcv3 = torch.tensor(xcv3).cuda()
        xcv2 = pred[2].reshape([1, 256, 56, 96])
        xcv2 = torch.tensor(xcv2).cuda()
        xcv1 = pred[3].reshape([1, 256, 112, 192])
        xcv1 = torch.tensor(xcv1).cuda()
        xcv5 = pred[4].reshape([1, 256, 7, 12])
        xcv5 = torch.tensor(xcv5).cuda()
        torch.cuda.empty_cache()
        ppp = []
        ppp.append(xcv1)
        ppp.append(xcv2)
        ppp.append(xcv3)
        ppp.append(xcv4)
        ppp.append(xcv5)
        for i in range(len(ppp)):
            print(ppp[i].shape)
        # pred = pred[0].reshape([1, 1024, 32, 50])
        # pred = pred[np.newaxis, :]
        pred = ppp
        del xcv1, xcv2, xcv3, xcv4, xcv5, pred, ppp, output
'''

import cv2
import numpy as np
video_path = "./out.mp4"
cam = cv2.VideoCapture(video_path)
k=0
accum_time = 0
curr_fps1 = 0
fps = "FPS: ??"
prev_time = timer()
with engine.create_execution_context() as context:
    while True:
        _, img = cam.read()
        image1 = cv2.resize(img, (448, 768))
        np.copyto(inputs[0][0], image1.reshape(1032192))
        # start_time = time.time()
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
