#!/usr/bin/env python3

import numpy as np
from torchvision.transforms import ToTensor
import cv2
import torch
import logging

import sys
import os
from typing import Tuple

# add yolov7 submodule to path
# ABS_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT = os.path.join(ABS_DIR, 'yolov7')
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
from visualizer import draw_detections
from utils.general import non_max_suppression
from models.experimental import attempt_load

def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape
    copied from https://github.com/meituan/YOLOv6/blob/main/yolov6/core/inferer.py
    '''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (
        ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes


class YoloV7:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 img_size: Tuple[int, int] = (640, 480), device: str = "cuda",
                 visualize: bool = True):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = device
        self.weights = weights
        self.model = attempt_load(self.weights, map_location=device)
        self.names = self.model.names
        self.visualize = visualize

    def model_info(self, verbose=False, img_size=640):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel()
                  for x in self.model.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.model.parameters()
                  if x.requires_grad)  # number gradients
        if verbose:
            logging.info('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name',
                  'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(self.model.named_parameters()):
                name = name.replace('module_list.', '')
                logging.info('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                      (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

        try:  # FLOPS
            from thop import profile
            stride = max(int(self.model.stride.max()), 32) if hasattr(
                self.model, 'stride') else 32
            img = torch.zeros((1, self.model.yaml.get('ch', 3), stride, stride), device=next(
                self.model.parameters()).device)  # input
            flops = profile(deepcopy(self.model), inputs=(img,), verbose=False)[
                0] / 1E9 * 2  # stride GFLOPS
            img_size = img_size if isinstance(img_size, list) else [
                img_size, img_size]  # expand if int/float
            fs = ', %.1f GFLOPS' % (
                flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        except (ImportError, Exception):
            fs = ''
        summary = f"\N{rocket}\N{rocket}\N{rocket} Yolov7 Detector summary:\n" \
            + f"Weights: {self.weights}\n" \
            + f"Confidence Threshold: {self.conf_thresh}\n" \
            + f"IOU Threshold: {self.iou_thresh}\n"\
            + f"{len(list(self.model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}"
        logging.info(summary)

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections

    def process_img(self, img_input):
        # automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(img_input, (w_scaled, h_scaled))
        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose(
            (2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)
        return img

    def run_webcam_demo(self):
        self.model_info()
        w_scaled, h_scaled = self.img_size
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if ret:
                # inference & rescaling the output to original img size
                h_orig, w_orig, c = frame.shape
                img_input = self.process_img(frame)
                detections = self.inference(img_input)
                detections[:, :4] = rescale(
                    [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
                detections[:, :4] = detections[:, :4].round()

                if self.visualize:
                    bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                              for x1, y1, x2, y2 in detections[:, :4].tolist()]
                    classes = [int(c) for c in detections[:, 5].tolist()]
                    conf = [float(c) for c in detections[:, 4].tolist()]
                    vis_img = draw_detections(
                        frame, bboxes, classes, self.names, conf)
                    cv2.imshow("yolov7", vis_img)
                    if cv2.waitKey(1) == ord('q'):
                        break


def main():
    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
    weights_path = "./weights/yolov7-tiny.pt"
    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError("Weights not found.")
    detector = YoloV7(weights=weights_path)
    detector.run_webcam_demo()


if __name__ == "__main__":
    main()
