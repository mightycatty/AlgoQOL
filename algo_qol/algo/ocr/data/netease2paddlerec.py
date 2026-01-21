# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/29 17:26
@Auth ： heshuai.sec@gmail.com
@File ：

crop
"""
import json
import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


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
    points = np.array([points]).reshape(-1, 2).astype(np.int32)

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return min_x, min_y, max_x, max_y


def main(result_json=r'E:\data\game_character\ocr\all-全产品通用-G112OCR识别模型0513（1）-20240517.json',
         img_f=r'E:\data\game_character\ocr\paddle_training\20240420_g112'):
    with open(result_json, 'r') as f:
        data = f.readlines()
    sub_f = os.path.basename(img_f)
    save_f = os.path.join('paddle_ocr_image', sub_f)
    os.makedirs(save_f, exist_ok=True)
    with open(f'paddle_ocr_result_{sub_f}.txt', 'w', encoding="utf-8") as f:
        for line in data:
            try:
                line = json.loads(line)
                img_name = line['url'].split('?')[0].split('/')[-1]
                img_name = os.path.join(img_f, img_name)
                anns = line['annotated_zones']
                if len(anns) < 1: continue
                img = cv2.imread(img_name)

                # print(img_name)
                for ann in anns:
                    label = ann['tag']
                    angle = ''
                    # try:
                    #     if '，' in label:
                    #         result = label.split('，')
                    #         label = result[0]
                    #         angle = result[-1]
                    #     if ',' in label:
                    #         result = label.split(',')
                    #         label = result[0]
                    #         angle = result[-1]
                    # except:
                    #     print(label)
                    #     exit()

                    if label == '*': continue
                    points = ann['points']

                    x0, y0, x1, y1 = bbox_from_points(points)
                    crop = img[y0:y1, x0:x1]
                    if angle != '':
                        angle = int(angle)
                        crop = Image.fromarray(crop)
                        crop = crop.rotate(-angle)
                        crop = np.array(crop)
                        print(label)
                        plt.imshow(crop)
                        plt.show()
                    dst_name = str(uuid.uuid4()) + '.jpg'
                    dst = os.path.join(save_f, dst_name)
                    line = f'{os.path.basename(img_f)}/{dst_name}\t{label}\n'
                    f.write(line)
                    f.flush()
                    # print(label)
                    # cv2.imshow('test', crop)
                    # cv2.waitKey(0)
                    cv2.imwrite(dst, img)
            except Exception as e:
                print(line, e)
                # cv2.imshow('test', img)
                # cv2.waitKey(0)
                # cv2.imshow('test', crop)
                # cv2.waitKey(0)
                # h, w = img.shape[:2]
                # pts = np.array(points, np.int32).reshape((-1, 1, 2))
                # img = cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=1)
                # # for idx, point in enumerate(points):
                # #     point = (int(point[0]), int(point[1]))
                # #     img = cv2.circle(img, tuple(point), 2, (0, 0, 255), -1)
                # #     img = cv2.putText(img, str(idx), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # img = cv2.putText(img, str(label), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # x0, y0, x1, y1 = bbox_from_points(points)
                # centerx, centery, w, h = _bbox_2_yolo([x0, y0, x1 - x0, y1 - y0], w, h)
                # line = f'{label} {centerx} {centery} {w} {h}\n'
                # f.write(line)
            # cv2.imshow('test', crop)
            # cv2.waitKey(0)
            # print(line)
            # print(bbox_from_points(points))
        # print(line)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
