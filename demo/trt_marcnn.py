import torch
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os, sys
import cv2
import datetime
import json
from tqdm import tqdm
import glob
from torch.autograd import Variable

# device = torch.device("cuda")
# device = torch.device('cpu')
config_file = '../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
opts = [#"MODEL.WEIGHT",
        # "./model_0.pth",
         "MODEL.DEVICE", "cuda"]
# load config from file and command-line arguments
cfg.merge_from_file(config_file)
cfg.merge_from_list(opts)
cfg.freeze()

model = COCODemo(cfg,
                 min_image_size=448,
                 # confidence_threshold=0.7,
                 show_mask_heatmaps=False)  # 0-30

model.model.backbone.eval()
trace_backbone = torch.jit.trace(model.model.backbone, (torch.rand([1, 3, 448, 768]).cuda()))
rand = torch.rand([1, 3, 448, 768]).cuda()
output = model.model.backbone(rand)
import torch.onnx as onnx
onnx._export(trace_backbone, torch.randn(1, 3, 448, 768), './mask_backbone_fpn.onnx',
             verbose=True, export_params=True, training=False,
             example_outputs=output, opset_version=9)
