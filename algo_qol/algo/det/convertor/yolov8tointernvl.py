# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/3 10:09
@Auth ： heshuai.sec@gmail.com
@File ：yolov8tointernvl.py

"""
import os
import cv2
import yaml
import json
import copy
from dataclasses import dataclass
from tqdm import tqdm


def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box


def get_image_size(image_path):
    """
    Gets the width and height of an image.
    """
    from PIL import Image
    img = Image.open(image_path)
    return img.width, img.height


def read_yolo_txt(txt_path, return_bbox_format='xyxy'):
    labels = []
    if os.path.exists(txt_path) is False:
        return labels
    with open(txt_path, "r") as f:
        for line in f:
            class_id, x, y, w, h = map(float, line.strip().split(" "))
            if return_bbox_format == 'xyxy':
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                labels.append([class_id, x1, y1, x2, y2])
            elif return_bbox_format == 'cxcywh':
                labels.append([class_id, x, y, w, h])
    return labels


def get_det_conversation(id, image_path, width, height, bboxes, labels,
                         prompt='Please detect and label all objects in the following image and mark their positions.'):
    # defined in https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data
    conversation = {
        'id': id,
        'image': image_path,
        'width': width,
        'height': height,
        'conversations': [
            {
                "from": "human",
                "value": "<image>\n{}".format(prompt)
            },
            {
                "from": "gpt",
                "value": 'Sure, I will detect and label all objects in the image and mark their positions.\n\n'
            }
        ]
    }

    bbox_str_template = '<ref>{}</ref><box>[{}]</box>\n'
    bbox_str = ''
    for bbox, label in zip(bboxes, labels):
        bbox_str += bbox_str_template.format(label, bbox)
    if len(bbox_str) == 0: bbox_str = 'No object found in the image.\n'
    conversation['conversations'][1]['value'] += bbox_str
    return conversation


def main(yolov8_yml, output_file='result.jsonl', phrase='train'):
    """convert yolov8 for internvl finetune
    Args:
        yolov8_yml: ultrality yolov8 data.yaml
        output_file:

    Returns:

    """
    with open(yolov8_yml) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    root_path = data['path']
    key = [phrase]
    labels_dict = data['names']
    img_list = []
    txt_list = []
    for k in key:
        folder_list = data[k]
        for f_item in folder_list:
            image_folder = os.path.join(root_path, f_item)
            label_folder = os.path.join(root_path, f_item.replace('images/', 'labels/'))
            image_list = os.listdir(image_folder)
            label_path = [os.path.join(label_folder, os.path.splitext(img)[0] + '.txt') for img in image_list]
            image_list = [os.path.join(image_folder, img) for img in image_list]
            img_list.extend(image_list)
            txt_list.extend(label_path)
    with open(output_file, 'w') as f_w:
        for id, (img_path, label_path) in tqdm(enumerate(zip(img_list, txt_list)), total=len(img_list)):
            try:
                image_width, image_height = get_image_size(img_path)
                bboxes = read_yolo_txt(label_path)
                labels = [labels_dict[c[0]] for c in bboxes]
                bboxes = [[c[1], c[2], c[3], c[4]] for c in bboxes]
                bboxes = [normalize_coordinates(c, 1, 1) for c in bboxes]
                img_relative_path = os.path.relpath(img_path, root_path)
                conversation = get_det_conversation(id, img_relative_path, image_width, image_height,
                                                    bboxes, labels)
                f_w.write(json.dumps(conversation) + '\n')
            except Exception as e:
                print(e)
                print(img_path, label_path)
    print('dataset saved to {}'.format(output_file))
    print('image root path: {}'.format(root_path))


if __name__ == '__main__':
    main(
        '/mnt/juicefs/cv_tikv/heshuai/projects/algo_det/projects/gaming_character/train/yolov8/config_20240711_h73.yaml')
