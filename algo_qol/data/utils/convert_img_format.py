import os
import cv2
from skimage import io
from tqdm import tqdm
import numpy as np


def convert_image_format(src_f, dst_format='jpg'):
    """
    convert all images in a folder to dst format, support sub-folders

    Args:
        src_f:
        dst_format: support 'jpg', 'png', 'jpeg'

    Returns:

    """
    img_list = [os.path.join(src_f, item) for item in os.listdir(src_f)]
    bar = tqdm(total=len(img_list))
    for item in img_list:
        bar.update()
        try:
            if '.' + dst_format in item:
                continue
            img = io.imread(item)
            if np.ndim(img) == 3:
                img = img[:, :, :3]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            dst = os.path.join(src_f, os.path.basename(item))
            img_format = os.path.splitext(dst)[1]
            dst = dst.replace(img_format, '.' + dst_format)
            if item != dst:
                os.remove(item)
            io.imsave(dst, img)
        except Exception as e:
            print(e)
            print(item)