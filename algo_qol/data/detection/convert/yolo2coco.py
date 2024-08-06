# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/11 11:59
@Auth ： heshuai.sec@gmail.com
@File ：yolo2coco.py

coco bbox: xywh
yolo bbox: cxcywh(0-1)

"""
import time
import re
import os
import json
from tqdm import tqdm
from loguru import logger
from collections import Counter


def get_file_recursively(folder_dir):
    """
    iteratively get file list under folder_dir
    :param folder_dir: folder
    :return: a list of files
    """
    file_list = []
    for root, dirs, files in os.walk(folder_dir, topdown=False):
        for name in files:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
        for name in dirs:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
    return file_list


def convert_yolov8_to_coco(images_dir, labels_dir, output_json, class_txt, absolute_path=False):
    images = []
    annotations = []
    categories = []
    with open(class_txt, "r") as f:
        class_list = [f.strip() for f in f.readlines()]
    for idx, class_name in enumerate(class_list):
        categories.append({"id": idx + 1, "name": class_name})

    # Iterate through images and labels
    if isinstance(images_dir, str):
        images_dir = [images_dir]
    images_list = []
    for images_dir_item in images_dir:
        images_list.extend(get_file_recursively(images_dir_item))
    images_list = list(
        filter(lambda x: re.search(r'.*\.jpg', x) or re.search(r'.*\.png', x) or re.search(r'.*\.jpeg', x),
               images_list))
    labels_list = get_file_recursively(labels_dir)
    labels_list = list(filter(lambda x: re.search(r'.*\.txt', x), labels_list))
    labels_dict = {os.path.basename(x).replace('.txt', ''): x for x in labels_list}
    bar = tqdm(total=len(images_list))
    log_count = {'positive': 0, 'negative': 0, 'corrupt': 0}

    for image_path in images_list:
        bar.update()
        filename = os.path.basename(image_path)
        filename = os.path.splitext(filename)[0]
        if filename not in labels_dict:
            label_path = None
        else:
            label_path = labels_dict[filename]
        try:
            width, height = get_image_size(image_path)
        except:
            logger.warning(f"can't get image size of {image_path}, skipping")
            log_count['corrupt'] += 1
            continue
        images.append({"id": len(images), "width": width, "height": height,
                       "file_name": os.path.basename(image_path) if not absolute_path else image_path})
        if label_path is None:
            log_count['negative'] += 1
        else:
            log_count['positive'] += 1
            # Read labels and convert to COCO format
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split(" "))
                    x0 = max(0, x - w / 2)
                    y0 = max(0, y - h / 2)
                    bbox = [int(x0 * width), int(y0 * height), int(w * width), int(h * height)]
                    annotations.append({
                        "id": len(annotations),
                        "image_id": len(images) - 1,
                        "category_id": int(class_id) + 1,  # COCO starts category IDs from 1
                        "bbox": bbox,  # xywh
                        "area": float(w * h),
                        "iscrowd": 0
                    })
                    if class_list[int(class_id)] in log_count:
                        log_count[class_list[int(class_id)]] += 1
                    else:
                        log_count[class_list[int(class_id)]] = 1

    coco_data = {"images": images, "annotations": annotations, "categories": categories,
                 "info": {'date': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())},
                 "licenses": []}
    # Create COCO JSON
    with open(output_json, "w") as f:
        json.dump(coco_data, f)

    # logging
    for k, v in log_count.items():
        logger.info(f"{k}: {v}")
    image_f_list = [os.path.dirname(x) for x in images_list]
    Count = Counter(image_f_list)
    for k, v in Count.items():
        logger.info(f"{k}: {v}")
    logger.info('json saved to {}'.format(output_json))


def get_image_size(image_path):
    """
    Gets the width and height of an image.
    """
    from PIL import Image
    img = Image.open(image_path)
    return img.width, img.height


if __name__ == '__main__':
    from fire import Fire

    # Fire(convert_yolov8_to_coco)

    # # Example usage
    # val_path = ['images/val', 'images/20240126_g112', 'images/20240126_l36', 'images/20240126_u5',
    #                 'images/20240130_g117_val', 'images/20240221_val']
    # convert_yolov8_to_coco(r"/logs/heshuai03/datasets/game_character/detection/images",
    #                        r"/logs/heshuai03/datasets/game_character/detection/labels",
    #                        "/logs/heshuai03/datasets/game_character/detection/coco_train.json",
    #                        class_txt=r'/logs/heshuai03/datasets/game_character/detection/classes.txt',
    #                        absolute_path=True,
    #                        ignore_path=val_path,
    #                        )
    #
    # convert_yolov8_to_coco(r"/logs/heshuai03/datasets/game_character/detection/images",
    #                        r"/logs/heshuai03/datasets/game_character/detection/labels",
    #                        "/logs/heshuai03/datasets/game_character/detection/coco_val.json",
    #                        class_txt=r'/logs/heshuai03/datasets/game_character/detection/classes.txt',
    #                        absolute_path=True,
    #                        include_path=val_path,
    #                        )
    train_path = ['images/20240620', 'images/yuanmeng', 'images/online_data_product_group']

    val_path = ['images/20240620_val', 'images/yuanmeng_val']

    base_f = r'/logs/heshuai03/datasets/logo'
    train_path = [os.path.join(base_f, x) for x in train_path]
    val_path = [os.path.join(base_f, x) for x in val_path]
    # val
    convert_yolov8_to_coco(val_path,
                           fr"{base_f}/labels",
                           fr"{base_f}/coco_val.json",
                           class_txt=fr'{base_f}/classes.txt',
                           absolute_path=True,
                           )
    # train
    convert_yolov8_to_coco(train_path,
                           fr"{base_f}/labels",
                           fr"{base_f}/coco_train.json",
                           class_txt=fr'{base_f}/classes.txt',
                           absolute_path=True,
                           )
