#!/usr/bin/env python3

import rospy

import sys
import os
FILE_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV7_ROOT = os.path.abspath(os.path.join(FILE_ABS_DIR, '../src/yolov7'))
if str(FILE_ABS_DIR) not in sys.path:
    sys.path.append(str(FILE_ABS_DIR))
from yolov7_ros import YoloV7_ROS


if __name__ == "__main__":
    rospy.init_node("yolov7_node", log_level=rospy.INFO)

    ns = rospy.get_name() + "/"

    weights_path = os.path.join(YOLOV7_ROOT, rospy.get_param(
        ns + "weights_path", 'weights/yolov7-tiny.pt'))
    img_topic = rospy.get_param(ns + "img_topic", "/usb_cam/image_raw")
    out_topic = rospy.get_param(ns + "out_topic", "yolo/detections")
    conf_thresh = rospy.get_param(ns + "conf_thresh", 0.25)
    iou_thresh = rospy.get_param(ns + "iou_thresh", 0.45)
    img_size = rospy.get_param(ns + "img_size", 640)
    visualize = rospy.get_param(ns + "visualize", True)
    device = rospy.get_param(ns + "device", "cuda")

    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError("Weights not found.")

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")

    detector = YoloV7_ROS(
        weights=weights_path,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=img_size,
        device="cuda",
        visualize=visualize,
        img_topic=img_topic,
        pub_topic=out_topic)

    rospy.spin()
