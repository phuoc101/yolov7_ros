from torchvision.transforms import ToTensor
import torch
from typing import Tuple
import cv2
import numpy as np
import random
import sys
import os
import time
import logging
import argparse

# add yolov7 submodule to path
FILE_ABS_DIR = os.path.dirname(os.path.abspath(__file__)) #nopep8
YOLOV7_ROOT = os.path.abspath(os.path.join(FILE_ABS_DIR, '../src/yolov7')) #nopep8
if str(YOLOV7_ROOT) not in sys.path: #nopep8
    sys.path.append(str(YOLOV7_ROOT)) #nopep8
from utils.ros import create_detection_msg #nopep8
from models.experimental import attempt_load #nopep8
from utils.general import non_max_suppression #nopep8
from visualizer import draw_detections #nopep8

def rescale(ori_shape, boxes, target_shape):
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
    def __init__(self, args):
        self.__conf_thres = args.conf_thres
        self.__iou_thres = args.iou_thres
        self.__img_size = args.img_size
        self.__device = args.device
        self.__visualize = args.visualize
        self.__weights = os.path.join(YOLOV7_ROOT, args.weights)
        self.__model = attempt_load(self.__weights, map_location=self.__device)
        self.__names = self.__model.names
        self.__colors = [[random.randint(0, 255)
                          for _ in range(3)] for _ in self.__names]

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.__model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.__conf_thres, iou_thres=self.__iou_thres
        )
        if detections:
            detections = detections[0]
        return detections

    def process_img(self, img_input):
        # automatically resize the image to the next smaller possible size
        h_ori, w_ori, _ = img_input.shape
        w_scaled = self.__img_size
        h_scaled = int(w_scaled * h_ori/w_ori)

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(img_input, (w_scaled, h_scaled))
        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose(
            (2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.__device)
        return img

    def run(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if ret:
                start = time.perf_counter()
                h_ori, w_ori, c = frame.shape
                w_scaled = self.__img_size
                h_scaled = int(w_scaled * h_ori/w_ori)
                img_input = self.process_img(frame)
                detections = self.inference(img_input)
                detections[:, :4] = rescale(
                    [h_scaled, w_scaled], detections[:, :4], [h_ori, w_ori])
                detections[:, :4] = detections[:, :4].round()
                end = time.perf_counter()
                elapsed = end - start
                logging.info(f"Inference time: {elapsed*1000:.2f} ms")
                logging.info(f"FPS: {1/elapsed:.2f} ms")

                if self.__visualize:
                    bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                              for x1, y1, x2, y2 in detections[:, :4].tolist()]
                    classes = [int(c) for c in detections[:, 5].tolist()]
                    conf = [float(c) for c in detections[:, 4].tolist()]
                    vis_img = draw_detections(
                        frame, bboxes, classes, self.__names, conf, self.__colors)
                    cv2.imshow("yolov7", vis_img)
                    if cv2.waitKey(1) == ord('q'):
                        break

def main(args):
    detector = YoloV7(args)
    detector.run()

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',  type=str, default='weights/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--visualize', action='store_true', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    args = parser.parse_args()
    logging.info(args)

    main(args)
