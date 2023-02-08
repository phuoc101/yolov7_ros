#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import random
import sys
import os
import time
import logging
import argparse

# add yolov7 submodule to path
FILE_ABS_DIR = os.path.dirname(os.path.abspath(__file__))  # noqa
YOLOV7_ROOT = os.path.abspath(os.path.join(FILE_ABS_DIR, "../src/yolov7"))  # noqa
if str(YOLOV7_ROOT) not in sys.path:  # noqa
    sys.path.append(str(YOLOV7_ROOT))  # noqa
from models.experimental import attempt_load  # noqa
from utils.general import non_max_suppression  # noqa
from visualizer import draw_detections  # noqa


def rescale(ori_shape, boxes, target_shape):
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (
        ori_shape[0] - target_shape[0] * ratio
    ) / 2

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
        self.__source = args.source
        self.__conf_thres = args.conf_thres
        self.__iou_thres = args.iou_thres
        self.__img_size = args.img_size
        self.__device = args.device
        self.__visualize = args.visualize
        self.__show_perf = args.show_perf
        self.__weights = os.path.join(YOLOV7_ROOT, args.weights)
        self.__model = attempt_load(self.__weights, map_location=self.__device)
        self.__names = self.__model.names
        self.__colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in self.__names
        ]
        self.model_info()

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

    def model_info(self, verbose=False):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in self.__model.parameters())  # number parameters
        n_g = sum(
            x.numel() for x in self.__model.parameters() if x.requires_grad
        )  # number gradients
        if verbose:
            rospy.loginfo(
                "%5s %40s %9s %12s %20s %10s %10s"
                % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
            )
            for i, (name, p) in enumerate(self.__model.named_parameters()):
                name = name.replace("module_list.", "")
                rospy.loginfo(
                    "%5g %40s %9s %12g %20s %10.3g %10.3g"
                    % (
                        i,
                        name,
                        p.requires_grad,
                        p.numel(),
                        list(p.shape),
                        p.mean(),
                        p.std(),
                    )
                )

        try:  # FLOPS
            from thop import profile

            stride = (
                max(int(self.__model.stride.max()), 32)
                if hasattr(self.__model, "stride")
                else 32
            )
            img = torch.zeros(
                (1, self.__model.yaml.get("ch", 3), stride, stride),
                device=next(self.__model.parameters()).device,
            )  # input
            flops = (
                profile(deepcopy(self.__model), inputs=(img,), verbose=False)[0]
                / 1e9
                * 2
            )  # stride GFLOPS
            img_size = (
                img_size if isinstance(img_size, list) else [img_size, img_size]
            )  # expand if int/float
            fs = ", %.1f GFLOPS" % (
                flops * img_size[0] / stride * img_size[1] / stride
            )  # 640x640 GFLOPS
        except (ImportError, Exception):
            fs = ""
        summary = (
            f"\N{rocket}\N{rocket}\N{rocket} Yolov7 Detector summary:\n"
            + f"Weights: {self.__weights}\n"
            + f"{len(list(self.__model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}\n"
            + f"Confidence Threshold: {self.__conf_thres}\n"
            + f"IOU Threshold: {self.__iou_thres}\n"
            + f"Device: {self.__device}\n"
        )
        logging.info(summary)

    def process_img(self, img_input):
        # automatically resize the image to the next smaller possible size
        h_ori, w_ori, _ = img_input.shape
        w_scaled = self.__img_size
        h_scaled = int(w_scaled * h_ori / w_ori)

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(img_input, (w_scaled, h_scaled))
        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.__device)
        return img

    def run(self):
        cam = cv2.VideoCapture(self.__source)
        while True:
            ret, frame = cam.read()
            if ret:
                start = time.perf_counter()
                h_ori, w_ori, c = frame.shape
                w_scaled = self.__img_size
                h_scaled = int(w_scaled * h_ori / w_ori)
                img_input = self.process_img(frame)
                detections = self.inference(img_input)
                detections[:, :4] = rescale(
                    [h_scaled, w_scaled], detections[:, :4], [h_ori, w_ori]
                )
                detections[:, :4] = detections[:, :4].round()
                end = time.perf_counter()
                if self.__show_perf:
                    elapsed = end - start
                    logging.info(f"Inference time: {elapsed*1000:.2f} ms")
                    logging.info(f"FPS: {1/elapsed:.2f}")

                if self.__visualize:
                    bboxes = [
                        [int(x1), int(y1), int(x2), int(y2)]
                        for x1, y1, x2, y2 in detections[:, :4].tolist()
                    ]
                    classes = [int(c) for c in detections[:, 5].tolist()]
                    conf = [float(c) for c in detections[:, 4].tolist()]
                    vis_img = draw_detections(
                        frame, bboxes, classes, self.__names, conf, self.__colors
                    )
                    cv2.imshow("yolov7", vis_img)
                    if cv2.waitKey(1) == ord("q"):
                        break
            else:
                logging.error(f"cannot open {self.__source}")
                break


def main(args):
    detector = YoloV7(args)
    detector.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="weights/yolov7-tiny.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="/dev/video0", help="video source"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou_thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--img_size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument("--visualize", action="store_true", help="Show detection image")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--show_perf", action="store_true", help="log fps and inference time"
    )
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
    main(args)
