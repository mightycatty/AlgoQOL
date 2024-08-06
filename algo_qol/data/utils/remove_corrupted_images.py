# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/3 15:58
@Auth ： heshuai.sec@gmail.com
@File ：remove_corrupted_images.py
"""
import shutil
import os
from loguru import logger
from tqdm import tqdm
import cv2
from tqdm.contrib.concurrent import thread_map
import os
from algo_qol.data.utils.file_utils import get_all_images


def worker(src, remove=True):
    try:
        img = cv2.imread(src)
        if img is None: raise ValueError
    except Exception as e:
        dst = os.path.join(os.path.dirname(src), '../corrupted', os.path.basename(src))
        if remove:
            os.remove(src)
        else:
            shutil.move(src, dst)


def rm_corrupted_image(src_f=r'/logs/heshuai03/datasets/politics/detection/train/images/110_online_negative_20240708',
                       threads_num=128, remove=False):
    """
    remove corrupted images
    Args:
        src_f: src folder， support sub folders
        threads_num:
        remove: whether to remove or move to a corrupted folder

    Returns:

    """
    img_list = get_all_images(src_f)
    logger.info(f'total:{len(img_list)}')
    if not remove:
        os.makedirs(os.path.join(os.path.dirname(src_f), 'corrupted'), exist_ok=True)
    thread_map(lambda x: worker(x, remove), img_list, max_workers=threads_num)
    return


if __name__ == '__main__':
    from fire import Fire

    Fire(main)
