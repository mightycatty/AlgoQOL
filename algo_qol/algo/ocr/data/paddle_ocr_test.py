# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/28 10:20
@Auth ： heshuai.sec@gmail.com
@File ：paddle_ocr_test.py

pip install "paddleocr>=2.0.1" --upgrade PyMuPDF==1.21.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
"""
import os

import cv2
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from algo_qol.utils.file_utils import get_file_recursively
from tqdm import tqdm
from loguru import logger


def draw_boxed_with_four_points(image, points, color=(0, 255, 0), thickness=2):
    """Draws a box with given 4 points.

    Args:
      image: a numpy array representing an image.
      points: a list of 4 points, each point is a tuple (x, y) representing the
        coordinates of the point.
      color: a tuple of 3 integers (r, g, b) representing the color of the box.
      thickness: an integer representing the thickness of the box in pixels.

    Returns:
      a numpy array representing the image with the box drawn on it.
    """

    # Convert the points to integers.
    points = np.int32(points)

    # Draw the box.
    cv2.line(image, points[0], points[1], color, thickness)
    cv2.line(image, points[1], points[2], color, thickness)
    cv2.line(image, points[2], points[3], color, thickness)
    cv2.line(image, points[3], points[0], color, thickness)

    return image


# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_gpu=True, det_model_dir='./ch_PP-OCRv4_det_server_infer',
                show_log=False)  # need to run only once to download and load model into memory
# ocr = PaddleOCR(use_gpu=True, show_log=False)  # need to run only once to download and load model into memory

base_f = r'/logs/heshuai03/datasets/game_character/detection/images/20240507_L33_positive'
save_f = 'padding_ocr_results'
os.makedirs(save_f, exist_ok=True)
img_list = get_file_recursively(base_f)
with open('paddle_ocr_results.txt', 'w') as f:
    bar = tqdm(total=len(img_list))
    for img_path in img_list:
        bar.update()
        if os.path.splitext(img_path)[-1] in ['.jpg', '.png', 'jpeg']:
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    result = ocr.ocr(img_path, rec=True, det=True)

                    result = result[0]
                    if result is not None:
                        image = Image.open(img_path).convert('RGB')
                        image = np.array(image)
                        boxes = [line[0] for line in result]
                        f.write(f'{img_path}\t{boxes}\n')
                        f.flush()
                        # txts = [line[1][0] for line in result]
                        # scores = [line[1][1] for line in result]
                        for points in boxes:
                            image = draw_boxed_with_four_points(image, points)
                        # im_show = draw_ocr(image, boxes, txts, scores, font_path=r'./chinese_cht.ttf')
                        im_show = Image.fromarray(image)
                        dst = os.path.join(save_f, os.path.basename(img_path))
                        im_show.save(dst)
            except Exception as e:
                logger.error(f'{img_path}:{e}')
#
