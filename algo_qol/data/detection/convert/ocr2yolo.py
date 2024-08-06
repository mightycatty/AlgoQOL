# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/29 17:26
@Auth ： heshuai.sec@gmail.com
@File ：netease2yolo.py
"""
import json
import os

import numpy as np
import cv2

# write a function to split text
def _bbox_2_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    centerx = bbox[0] + w / 2
    centery = bbox[1] + h / 2
    dw = 1 / img_w
    dh = 1 / img_h
    centerx *= dw
    w *= dw
    centery *= dh
    h *= dh
    return centerx, centery, w, h

def bbox_from_points(points):
    points = np.array([points]).reshape(-1, 2)

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return min_x, min_y, max_x, max_y


result_path = r'/logs/heshuai03/projects/detection/projects/gaming_text_yolov8/paddle_ocr_results.txt'
save_f = r'paddle_ocr'
# img_f = r'C:\Users\N28828\Downloads\train_full_images_0\train_full_images_0'
os.makedirs(save_f, exist_ok=True)
with open(result_path, 'r') as f:
    data = f.readlines()
    data = [item.strip() for item in data]
for item in data:
    img_path, points = item.split('\t')
    img_name = os.path.basename(img_path)
    points = eval(points)
    if len(points) < 1:continue
    txt_name = os.path.splitext(img_name)[0] + '.txt'
    sub_f = os.path.join(save_f, img_path.split('/')[-2])
    os.makedirs(sub_f, exist_ok=True)
    txt_name = os.path.join(sub_f, txt_name)
    # img_name = os.path.join(img_f, img_name) + '.jpg'
    # if not os.path.exists(img_name): continue

    with open(txt_name, 'w') as f:
        for point in points:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                x0, y0, x1, y1 = bbox_from_points(point)
                centerx, centery, w, h = _bbox_2_yolo([x0, y0, x1-x0, y1-y0], w, h)
                line = f'4 {centerx} {centery} {w} {h}\n'
                f.write(line)
            # print(line)
        # print(bbox_from_points(points))
    # print(line)
