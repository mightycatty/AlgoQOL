import sys
import hashlib
import uuid
import cv2
import imageio
from skimage import io
# import matplotlib.pyplot as plt
import numpy as np
from imagededup.methods import PHash, CNN
from imagededup.utils import plot_duplicates
import os
import shutil
from tqdm import tqdm


def rm_duplicated_image(src_f, distance=1, debug=False):
    handle = PHash()
    # handle = CNN()
    src_f = src_f.rstrip('/').rstrip('\\')
    if not debug:
        dst_f = src_f + '_duplicated'
    else:
        dst_f = src_f + '_duplicated_debug'
    os.makedirs(dst_f, exist_ok=True)
    # 生成图像目录中所有图像的二值hash编码
    encodings = handle.encode_images(image_dir=src_f)

    # 对已编码图像寻找重复图像
    # duplicates = handle.find_duplicates(encoding_map=encodings, scores=True)
    if debug:
        duplicates_to_remove = handle.find_duplicates(encoding_map=encodings, max_distance_threshold=distance)
        for item in duplicates_to_remove.keys():
            dp = duplicates_to_remove[item]
            if len(dp) > 0:
                try:
                    img_list = [os.path.join(src_f, item)] + dp
                    img_list = [os.path.join(src_f, item_) for item_ in img_list]
                    img_list = [imageio.imread(item_) for item_ in img_list]
                    img_list = [cv2.resize(item_, (128, 256))[:, :, :3] for item_ in img_list]
                    vis = np.hstack(img_list)
                    save_name = os.path.join(dst_f, str(uuid.uuid4()) + '.jpg')
                    io.imsave(save_name, vis)
                except Exception as e:
                    print(e)
                    continue
                # plt.imshow(vis)
                # plt.show()
    else:
        duplicates_to_remove = handle.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=distance)
        bar = tqdm(total=len(duplicates_to_remove))
        for item in duplicates_to_remove:
            bar.update()
            src = os.path.join(src_f, item)
            dst = os.path.join(dst_f, item)
            shutil.move(src, dst)