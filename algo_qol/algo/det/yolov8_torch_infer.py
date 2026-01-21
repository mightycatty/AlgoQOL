# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/14 16:39
@Auth ： heshuai.sec@gmail.com
@File ：yolov8_torch_infer.py
"""
import os
import numpy as np

from ultralytics import YOLO
from PIL import Image
import torch


def nms(bounding_boxes, confidence_score, labels, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_label.append(labels[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_label


class Yolov8DetTorch:
    def __init__(self,
                 model_dir,
                 device='cuda',
                 post_processing_fn=None,
                 iou=0.5,
                 conf=0.1,
                 *args, **kwargs
                 ):

        # if (not torch.cuda.is_available()) and device == 'cuda':
        #     print('gpu not found, fall back to cpu')
        #     device = 'cpu'

        self.model = YOLO(model_dir)  # pretrained YOLOv8n model
        self._device = device
        self.post_processing_fn = post_processing_fn
        self.iou = iou
        self.conf = conf

    def __call__(self, image_rgb: np.array, **kwargs):

        if 'verbose' not in kwargs: kwargs['verbose'] = False

        image_rgb = Image.fromarray(image_rgb)
        results = self.model.predict(source=image_rgb, device=self._device, nms=True, iou=self.iou, conf=self.conf,
                                     **kwargs)

        # assert len(results) == 1, 'only support 1 image input'

        results = results[0].cpu().numpy()
        scores = np.float32(results.boxes.conf)
        bboxes = np.int32(results.boxes.xyxy)[scores > self.conf].tolist()
        labels = np.int32(results.boxes.cls)[scores > self.conf].tolist()
        scores = scores.tolist()

        classes_list = results.names
        labels = [classes_list[item] for item in labels]
        ret = (bboxes, scores, labels)
        ret = nms(*ret, self.iou)

        if self.post_processing_fn is not None:
            ret = self.post_processing_fn(ret)

        labels = ret[2]
        scores = [round(s, 3) for s in ret[1]]
        bboxes = ret[0]
        result = dict(label=labels, score=scores, bboxes=bboxes)
        return result


