# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/18 11:43
@Auth ： heshuai.sec@gmail.com
@File ：coco_utils.py
"""
import os.path
import json
import random

import supervision as sv


def val_split(img_path, ann_path, val_ratio=0.1):
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=img_path,
        annotations_path=ann_path
    )
    train_dataset, test_dataset = dataset.split(split_ratio=val_ratio)
    ann_f = os.path.dirname(ann_path)
    train_dataset.as_coco(
        images_directory_path=img_path,
        annotations_path=os.path.join(ann_f, 'train.json')
    )
    test_dataset.as_coco(
        images_directory_path=img_path,
        annotations_path=os.path.join(ann_f, 'val.json')
    )
    return


def split_coco_dataset(coco_file, train_val_split=0.9):
    """
    Splits a COCO dataset into train and val sets.

    Args:
        coco_file: Path to the COCO JSON file.
        train_val_split: The proportion of data to be included in the train set.

    Returns:
        A tuple containing the paths to the train and val JSON files.
    """

    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    num_images = len(images)
    train_val_split_index = int(num_images * train_val_split)

    train_images = images[:train_val_split_index]
    val_images = images[train_val_split_index:]
    train_image_ids = {image['id'] for image in train_images}
    train_annotations = [
        annotation for annotation in annotations
        if annotation['image_id'] in train_image_ids
    ]
    val_image_ids = {image['id'] for image in train_images}
    val_annotations = [
        annotation for annotation in annotations
        if annotation['image_id'] in val_image_ids
    ]

    train_coco_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }
    val_coco_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories']
    }

    train_file = os.path.splitext(coco_file)[0] + '_train.json'
    val_file = os.path.splitext(coco_file)[0] + '_val.json'

    with open(train_file, 'w') as f:
        json.dump(train_coco_data, f)
    with open(val_file, 'w') as f:
        json.dump(val_coco_data, f)

    return train_file, val_file


import json


def analyze_coco_dataset(annotations_file):
    """
    Analyzes a COCO detection format dataset and returns a dictionary containing various statistics.

    Args:
      annotations_file: Path to the COCO annotations file.

    Returns:
      A dictionary containing statistics about the dataset.
    """

    # Load the annotations
    with open(annotations_file, "r") as f:
        data = json.load(f)

    # Get information about the images
    images = data["images"]
    num_images = len(images)

    # Get information about the annotations
    annotations = data["annotations"]
    num_annotations = len(annotations)

    # Get information about the categories
    categories = data["categories"]
    num_categories = len(categories)

    # Analyze image sizes
    image_widths = [image["width"] for image in images]
    image_heights = [image["height"] for image in images]
    avg_image_width = sum(image_widths) / len(image_widths)
    avg_image_height = sum(image_heights) / len(image_heights)

    # Analyze object sizes
    object_areas = [annotation["area"] for annotation in annotations]
    avg_object_area = sum(object_areas) / len(object_areas)

    # Analyze object aspect ratios
    object_aspect_ratios = [annotation["bbox"][2] / annotation["bbox"][3] for annotation in annotations]
    avg_object_aspect_ratio = sum(object_aspect_ratios) / len(object_aspect_ratios)

    # Analyze object counts per category
    object_counts_per_category = {}
    for annotation in annotations:
        category_id = annotation["category_id"]
        if category_id not in object_counts_per_category:
            object_counts_per_category[category_id] = 0
        object_counts_per_category[category_id] += 1

    # Return the statistics
    return {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "num_categories": num_categories,
        "avg_image_width": avg_image_width,
        "avg_image_height": avg_image_height,
        "avg_object_area": avg_object_area,
        "avg_object_aspect_ratio": avg_object_aspect_ratio,
        "object_counts_per_category": object_counts_per_category,
    }


if __name__ == '__main__':
    # train_file, val_file = split_coco_dataset(r'/mnt/juicefs/cv_tikv/heshuai/datasets/openbrand/openbrand.json')
    print(analyze_coco_dataset(r'E:\downloads\dataset_all_val_v0.1.4/dataset_all_val_v0.1.4.json'))
