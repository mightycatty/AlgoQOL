# -*- coding: utf-8 -*-
"""
@Time ： 2023/6/18 11:43
@Auth ： heshuai.sec@gmail.com
@File ：coco_utils.py
"""
import os.path
import json
import random
import json

__name__ = [
    'coco_split',
    'coco_analyze',
]


def coco_split(coco_file: str,
               val_ratio: float = 0.1,
               seed: int = 0,
               print_stats: bool = True
               ):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    random.seed(seed)
    random.shuffle(images)

    num_images = len(images)
    train_val_split_index = int(num_images * val_ratio)

    val_images = images[:train_val_split_index]
    train_images = images[train_val_split_index:]
    train_image_ids = {image['id'] for image in train_images}
    train_annotations = [
        annotation for annotation in annotations
        if annotation['image_id'] in train_image_ids
    ]
    val_image_ids = {image['id'] for image in val_images}
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
    print('COCO split done!')
    print('train file: ', train_file)
    print('val file: ', val_file)
    if print_stats:
        print('train summary:\n')
        coco_analyse(train_file, verbose=True)
        print('val summary:\n')
        coco_analyse(val_file, verbose=True)
    return train_file, val_file


def coco_analyse(annotations_file:str,
                 verbose: bool = True
                 ):
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

    # get category names


    # Get information about the annotations
    annotations = data["annotations"]
    num_annotations = len(annotations)

    # Get information about the categories
    categories = data["categories"]
    num_categories = len(categories)
    id2name = {category["id"]: category["name"] for category in categories}

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
        name = id2name[category_id]
        if name not in object_counts_per_category:
            object_counts_per_category[name] = 0
        object_counts_per_category[name] += 1
    # sort
    object_counts_per_category = dict(sorted(object_counts_per_category.items(), key=lambda x: x[1], reverse=False))
    # Return the statistics
    stat = {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "num_categories": num_categories,
        'id2name': id2name,
        "avg_image_width": avg_image_width,
        "avg_image_height": avg_image_height,
        "avg_object_area": avg_object_area,
        "avg_object_aspect_ratio": avg_object_aspect_ratio,
        "object_counts_per_category": object_counts_per_category,
    }
    if verbose:
        print(json.dumps(stat, indent=2))
    return stat
