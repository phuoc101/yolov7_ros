import rospy
import torch
from typing import List
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBox, BoundingBoxes


def create_header():
    h = Header()
    h.stamp = rospy.Time.now()
    return h


def create_detection_msg(img_msg: Image, detections: torch.Tensor, names: List[str]) -> BoundingBoxes:
    """
    :param img_msg: original ros image message
    :param detections: torch tensor of shape [num_boxes, 6] where each element is
        [x1, y1, x2, y2, confidence, class_id]
    :returns: detections as a ros message of type Detection2DArray
    """
    bounding_boxes_msg = BoundingBoxes()

    # header
    header = create_header()
    bounding_boxes_msg.header = header
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        # bbox
        bbox = BoundingBox()
        bbox.xmin = int(x1)
        bbox.ymin = int(y1)
        bbox.xmax = int(x2)
        bbox.ymax = int(y2)
        bbox.Class = names[int(cls)]
        bbox.probability = conf
        bounding_boxes_msg.bounding_boxes.append(bbox)
    return bounding_boxes_msg


