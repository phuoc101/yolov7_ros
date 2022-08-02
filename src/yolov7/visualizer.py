import numpy as np
import cv2
from typing import List


# def get_random_color(seed):
#     gen = np.random.default_rng(seed)
#     color = tuple(gen.choice(range(256), size=3))
#     color = tuple([int(c) for c in color])
#     return color


def draw_detections(img: np.array, bboxes: List[List[int]], classes: List[int], names: List[str], conf: List[float], colors):
    line_thickness = 3
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    for bbox, cls, prob in zip(bboxes, classes, conf):
        x1, y1, x2, y2 = bbox

        color = colors[int(cls)]
        img = cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), color, tl)
        x_text = int(x1)
        y_text = max(15, int(y1 - 10))
        label = f"{names[cls]}: {prob:.2f}"
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        p2 = x_text + t_size[0], y_text - t_size[1] - 10
        img = cv2.rectangle(img, (x_text, y_text + 10), p2, color, -
                            1, cv2.LINE_AA)  # filled
        img = cv2.putText(
            img, label, (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            tl / 3, (255,255,255), tf, cv2.LINE_AA)

    return img
