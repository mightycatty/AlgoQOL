# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/25 14:20
@Auth ： heshuai.sec@gmail.com
@File ：make_dataset_from_yolo.py
"""
import os
import cv2

import numpy as np


def get_bounding_box(det_boxes, w, h, expand_ratio=0., min_expand_ratio=0, mini_pixels=10):
    try:
        x0 = int(np.min(det_boxes[:, 0]))
        y0 = int(np.min(det_boxes[:, 1]))
        x1 = int(np.max(det_boxes[:, 2]))
        y1 = int(np.max(det_boxes[:, 3]))
        w_expand = int((x1 - x0) * expand_ratio) // 2
        h_expand = int((y1 - y0) * expand_ratio) // 2
        w_expand = int(max(w_expand, w * min_expand_ratio))
        h_expand = int(max(h_expand, h * min_expand_ratio))
        w_expand = int(max(w_expand, mini_pixels))
        h_expand = int(max(h_expand, mini_pixels))
        x0 = max(0, x0 - w_expand)
        y0 = max(0, y0 - h_expand)
        x1 = min(w, x1 + w_expand)
        y1 = min(h, y1 + h_expand)
        return x0, y0, x1, y1
    except Exception as e:
        print(e)
        return 0, 0, w, h


from algo_qol.utils.file_utils import get_all_images


def read_yolo_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split() for line in lines]
        lines = [[float(i) for i in line] for line in lines]
    return lines


class YoloDataset:
    def __init__(self, root_dir, image_dir=None, label_dir=None, class_names=None):
        self.root = root_dir
        if image_dir is None:
            self.images_f = os.path.join(root_dir, 'images')
        else:
            self.images_f = image_dir
        if label_dir is None:
            self.label_f = os.path.join(root_dir, 'labels')
        else:
            self.label_f = label_dir
        self.img_list = get_all_images(self.images_f)
        self.id2label = {}
        if class_names is None:
            class_txt = os.path.join(root_dir, 'classes.txt')
            with open(class_txt, 'r') as f:
                self.class_name = f.readlines()
                self.class_name = [i.strip() for i in self.class_name]
        else:
            self.class_name = class_names.split(',')
        for i in range(len(self.class_name)):
            self.id2label[i] = self.class_name[i]

    def __iter__(self):
        for img_path in self.img_list:
            img_relative_path = os.path.relpath(img_path, self.images_f)
            txt_path = os.path.join(self.label_f, img_relative_path)
            img_format = os.path.splitext(img_path)[-1]
            txt_path = txt_path[:-len(img_format)] + '.txt'
            if not os.path.exists(txt_path):
                continue
            annotations = read_yolo_txt(txt_path)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            crop_result = []
            for annotation in annotations:
                if len(annotation) == 4:
                    label_id, x_center, y_center, w, h = annotation
                else:
                    label_id, x_center, y_center, w, h, score = annotation
                x_center = x_center * img_w
                y_center = y_center * img_h
                w = w * img_w
                h = h * img_h
                x1 = round(x_center - w / 2)
                y1 = round(y_center - h / 2)
                x2 = round(x_center + w / 2)
                y2 = round(y_center + h / 2)
                bboxes = np.array([[x1, y1, x2, y2]]).astype(np.int32)
                x0, y0, x1, y1 = get_bounding_box(bboxes, img_w, img_h)
                crop_result.append((img[y0:y1, x0:x1, :], self.id2label[int(label_id)]))
            yield crop_result, img_path


def main():
    dataset = YoloDataset('E:\data\politics\detection\merge',
                          image_dir='/mnt/juicefs/cv_tikv/heshuai/projects/server_test/dev_20240429_game_character/U5_results/character_exist',
                          label_dir=r'/mnt/juicefs/cv_tikv/heshuai/projects/algo_detection/projects/gaming_character/runs/detect/predict/labels',
                          class_names='harmful,curve_text,gaming_logo,gaming_text,normal_text')
    save_f = r'character_crop/U5_results'
    os.makedirs(save_f, exist_ok=True)
    for img_crop_result, img_path in dataset:
        try:
            for idx, data in enumerate(img_crop_result):
                img, label = data
                dst_f = os.path.join(save_f, label)
                name = os.path.splitext(os.path.basename(img_path))[0] + f'_{idx}.jpg'
                dst = os.path.join(dst_f, name)
                os.makedirs(dst_f, exist_ok=True)
                cv2.imwrite(dst, img)
        except Exception as e:
            print(e)
            print(img_path)


if __name__ == '__main__':
    main()
