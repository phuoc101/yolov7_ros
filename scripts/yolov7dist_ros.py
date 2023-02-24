from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBoxesDist

# from torchvision.transforms import ToTensor
import torch

# from typing import Tuple
import rospy
import cv2
import numpy as np
import random
import sys
import os
from xgboost import XGBRegressor
from hummingbird.ml import convert

# add yolov7 submodule to path
FILE_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV7_ROOT = os.path.abspath(os.path.join(FILE_ABS_DIR, "../src/yolov7"))
if str(YOLOV7_ROOT) not in sys.path:
    sys.path.append(str(YOLOV7_ROOT))
from visualizer import draw_detections  # noqa
from utils.general import non_max_suppression  # noqa
from models.experimental import attempt_load  # noqa
from utils.ros import create_detection_with_dist_msg  # noqa


def rescale(ori_shape, boxes, target_shape):
    """Rescale the output to the original image shape
    copied from https://github.com/meituan/YOLOv6/blob/main/yolov6/core/inferer.py
    """
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class YoloV7Dist_ROS:
    def __init__(
        self,
        weights,
        xgb_weights,
        img_size: int,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45,
        device: str = "cuda",
        visualize: bool = True,
        input_img_topic: str = "/image_raw",
        pub_topic: str = "yolov7_detections_dist",
        # output_img_topic: str = "yolov7/image_raw",
    ):
        rospy.loginfo("Starting Yolov7Dist_ROS node")
        self.__conf_thresh = conf_thresh
        self.__iou_thresh = iou_thresh
        self.__img_size = img_size
        self.__device = device
        self.__weights = weights
        self.__xgb_weights = xgb_weights
        self.__model = attempt_load(self.__weights, map_location=device)
        self.__names = self.__model.names
        self.__colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.__names]
        self.__visualize = visualize
        self.__input_img_topic = input_img_topic
        self.__output_topic = pub_topic
        # self.__output_img_topic = output_img_topic
        # ROS
        # self.visualization_publisher = (
        #     rospy.Publisher(self.__output_img_topic, Image, queue_size=10) if visualize else None
        # )
        self.xgb_dist = XGBRegressor()
        self.xgb_dist.load_model(self.__xgb_weights)
        self.xgb_dist_gpu = convert(self.xgb_dist, "pytorch")
        self.xgb_dist_gpu.to("cuda")
        rospy.loginfo("XGB CUDA init")
        self.model_info()
        self.img_subscriber = rospy.Subscriber(self.__input_img_topic, Image, self.__img_cb)
        self.detection_publisher = rospy.Publisher(self.__output_topic, BoundingBoxesDist, queue_size=10)
        self.bridge = CvBridge()

    def model_info(self, verbose=False, img_size=640):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in self.__model.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.__model.parameters() if x.requires_grad)  # number gradients
        if verbose:
            rospy.loginfo(
                "%5s %40s %9s %12s %20s %10s %10s" % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
            )
            for i, (name, p) in enumerate(self.__model.named_parameters()):
                name = name.replace("module_list.", "")
                rospy.loginfo(
                    "%5g %40s %9s %12g %20s %10.3g %10.3g"
                    % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
                )

        try:  # FLOPS
            from thop import profile

            stride = max(int(self.__model.stride.max()), 32) if hasattr(self.__model, "stride") else 32
            img = torch.zeros(
                (1, self.__model.yaml.get("ch", 3), stride, stride), device=next(self.__model.parameters()).device
            )  # input
            flops = profile(deepcopy(self.__model), inputs=(img,), verbose=False)[0] / 1e9 * 2  # stride GFLOPS
            img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
            fs = ", %.1f GFLOPS" % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        except (ImportError, Exception):
            fs = ""
        summary = (
            "\N{rocket}\N{rocket}\N{rocket} Yolov7 Detector summary:\n"
            + f"Weights: {self.__weights}\n"
            + f"Confidence Threshold: {self.__conf_thresh}\n"
            + f"IOU Threshold: {self.__iou_thresh}\n"
            + f"Image size: [{self.__img_size}, {self.__img_size}]\n"
            + f"{len(list(self.__model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}\n"
            + f"Input topic: {self.__input_img_topic}\n"
            + f"Output topic: {self.__output_topic}\n"
            # + f"Output image topic: {self.__output_img_topic}"
        )
        rospy.loginfo(summary)

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.__model(img)[0]
        detections = non_max_suppression(pred_results, conf_thres=self.__conf_thresh, iou_thres=self.__iou_thresh)
        if detections:
            detections = detections[0]
        return detections

    def process_img(self, img_input):
        # automatically resize the image to the next smaller possible size
        h_ori, w_ori, _ = img_input.shape
        w_scaled = self.__img_size
        h_scaled = int(w_scaled * h_ori / w_ori)
        if self.__img_size == 1280:
            # (TODO: implemnent letterbox to get standard image size, or just crop img properly so that it's square)
            h_scaled = 1280  # hot fix for 1280

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(img_input, (w_scaled, h_scaled))
        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.__device)
        return img

    def __img_cb(self, img_msg):
        """callback function for publisher"""
        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        h_ori, w_ori, c = frame.shape
        w_scaled = self.__img_size
        h_scaled = int(w_scaled * h_ori / w_ori)
        img_input = self.process_img(frame)
        detections = self.inference(img_input)
        detections[:, :4] = rescale([h_scaled, w_scaled], detections[:, :4], [h_ori, w_ori])
        detections[:, :4] = detections[:, :4].round()
        gn = torch.tensor([[w_ori, h_ori, w_ori, h_ori]])  # normalization gain whwh
        dists = []
        for *xyxy, conf, cls in detections:
            dist = -1
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            # detect distance for person
            if self.__names[int(cls)] == "person":
                xywh_np = np.array(xywh)[np.newaxis, :]
                dist = self.xgb_dist_gpu.predict(xywh_np)[0]
            dists.append(dist)
        # publishing
        detection_msg = create_detection_with_dist_msg(
            img_msg=img_msg, detections=detections, dists=dists, names=self.__names
        )
        self.detection_publisher.publish(detection_msg)

        if self.__visualize:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            conf = [float(c) for c in detections[:, 4].tolist()]
            vis_img = draw_detections(frame, bboxes, classes, self.__names, conf, self.__colors)
            cv2.imshow("yolov7", vis_img)
            # vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            # vis_msg.header.stamp = detection_msg.header.stamp
            # self.visualization_publisher.publish(vis_msg)
            cv2.waitKey(1)
