import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import torch
import os
from torchvision import transforms as T
from torchvision.transforms import functional as F
import  cv2
import time
from PIL import Image
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument(
    "--config-file",
    default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.7,
    help="Minimum score for the prediction to be shown",
)
parser.add_argument(
    "--min-image-size",
    type=int,
    default=500,
    help="Smallest size of the image to feed to the model. "
        "Model was trained with 800, which gives best results",
)
parser.add_argument(
    "--show-mask-heatmaps",
    dest="show_mask_heatmaps",
    help="Show a heatmap probability for the top masks-per-dim masks",
    action="store_true",
)
parser.add_argument(
    "--masks-per-dim",
    type=int,
    default=2,
    help="Number of heatmaps per dimension to show",
)
parser.add_argument(
    "opts",
    help="Modify model config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

# load config from file and command-line arguments
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

class Resize(object):
    '''
    Resize the pic
    '''
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        '''

        :param image_size:
        :return: Scaled image_size
        '''
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        # size = self.get_size(image.size)
        size = (448, 768)
        image = F.resize(image, size)
        return image

def transform():
    """
    Creates a basic transformation that was used to train the models
    """

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = 800
    # max_size = self.min_image_size
    min_size = 500
    transform = T.Compose(
        [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

# prepare object that handles inference plus adds predictions on top of image
coco_demo = COCODemo(
    cfg,
    confidence_threshold=args.confidence_threshold,
    show_mask_heatmaps=args.show_mask_heatmaps,
    masks_per_dim=args.masks_per_dim,
    min_image_size=args.min_image_size,
)

from timeit import default_timer as timer
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network() as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = (1 << 30) * 4
    with open('./mask_backbone_fpn.onnx', 'rb') as model:
        a = parser.parse(model.read())  # parser the model
        print(a)
    # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
    print('Building an engine from file mask_backbone.onnx; this may take a while...')
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
    # Assign host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    # 将device端的buffer追加到device的bindings.
    # Append the buffer on the device side to the bindings of the device
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
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]


'''
#调用代码
from tqdm import tqdm_notebook as tqdm
import numpy as np
import cv2
# Batch processing image test
image = cv2.imread("./a1.jpg")
img = cv2.resize(image, (448, 768))
with engine.create_execution_context() as context:
    for i in (range(10000000)):
        image = cv2.imread("./a1.jpg")
        img = cv2.resize(image, (448, 768))
        image1 = img
        np.copyto(inputs[0][0], image1.reshape(1032192))
        # np.copyto(inputs[0][0], image1.reshape(1200000))
        start = timer()
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end = timer()
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
        # pred = pred[0].reshape([1, 1024, 32, 50])
        # pred = pred[np.newaxis, :]
        pred = ppp
        result, ttt = coco_demo.run_on_opencv_image(img, pred)
        del xcv1, xcv2, xcv3, xcv4, xcv5, pred, ppp, output, img, image1

        print('time_cost', (end - start)*1000, "ms")
        print("time", ttt)

'''

import numpy as np
total = 0
i = 0
with engine.create_execution_context() as context:
    # read the all the pic in folder /image/
    for (root, dirs, files) in os.walk('../20191012_0/'):
        if files:
            for f in files:
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image1 = transform()(image)
                i = i + 1
                # np.copyto(inputs[0][0], np.ones([800, 800, 3]).reshape([1920000]))
                np.copyto(inputs[0][0], image1.reshape(1032192))
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
                pred = ppp
                result, ttt = coco_demo.run_on_opencv_image(image, pred)
                total = total + ttt
                cv2.putText(result, text="xyz", org=(3, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.50, color=(255, 255, 255), thickness=3)
                # cv2.imshow("pic", result)
                print("path:", path[14:])
                result = cv2.resize(result, (result.shape[1] // 2, result.shape[0] // 2))
                cv2.imwrite("./ce/" + path[14:], result)


print("avg time:", total/i)

'''
# Read Camare
video_path = "./out.mp4"
cam = cv2.VideoCapture(video_path)
k=0
accum_time = 0
curr_fps1 = 0
fps = "FPS: ??"
prev_time = timer()

with engine.create_execution_context() as context:
    while True:
        start_time = time.time()
        _, img = cam.read()
        image1 = transform()(img)
        np.copyto(inputs[0][0], image1.reshape(1032192))
        # start_time = time.time()
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
        # pred = pred[0].reshape([1, 1024, 32, 50])
        # pred = pred[np.newaxis, :]
        pred = ppp
        result, ttt = coco_demo.run_on_opencv_image(img, pred)
        del xcv1, xcv2, xcv3, xcv4, xcv5, pred, ppp, output, img, image1
        fps = "FPS: " + str(int(1000/int(ttt)))
        cv2.putText(result, text=fps, org=(3, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.50, color=(255, 255, 255), thickness=3)
        cv2.namedWindow("result", 0)
        cv2.imshow("result", result)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
    
'''