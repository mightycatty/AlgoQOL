# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/29 17:26
@Auth ： heshuai.sec@gmail.com
@File ：netease2yolo.py
"""
import json
import os
import shutil
from algo_qol.utils.file_utils import get_all_images
from algo_qol.utils.video_img_downloader import download_img
import numpy as np
import cv2
from tqdm import tqdm
from loguru import logger
import hashlib

new_to_old = {
    "nazi": "flag_Nazi",
    'japanflag': 'flag_japan_military',
    'chinamap': 'flag_ChineseMap',
    'chinaflag': 'flag_naFlag',
    'chinaemblem': 'flag_naEmblem',
    'zhongguojinghui': 'flag_police',
    'communistemblem': 'flag_paFE',
    'gamingtext': 'gaming_text_h73',
}

logger.add('netease2yolo.log', rotation='500 MB')


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
    points = np.array([points]).reshape(-1, 2)

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return min_x, min_y, max_x, max_y


def read_jsons(result_json_or_folder):
    data = []
    if os.path.isfile(result_json_or_folder):
        result_json = [result_json_or_folder]
    elif os.path.isdir(result_json_or_folder):
        json_list = os.listdir(result_json_or_folder)
        json_list = [os.path.join(result_json_or_folder, f) for f in json_list if f.endswith('.json')]
        result_json = json_list
    for result_json_item in result_json:
        with open(result_json_item, 'r') as f:
            data += f.readlines()
    return list(set(data))


def main(result_json_or_folder=r'E:\data\politics\detection\pool\all-全产品通用-成都-政治敏感检测07-09-20240710',
         img_f=r'E:\data\politics\detection\pool\all-全产品通用-成都-政治敏感检测07-09-20240710',
         save_f=r'politics',
         class_txt=r'E:\data\politics\detection\train\classes.txt',
         ):
    """
    Args:
        result_json_or_folder:
        img_f:
        save_f:
    Returns:
    """
    img_dst_f = os.path.join(save_f, 'images')
    txt_dst_f = os.path.join(save_f, 'labels')
    os.makedirs(img_dst_f, exist_ok=True)
    os.makedirs(txt_dst_f, exist_ok=True)
    label_list = []
    if class_txt is not None:
        with open(class_txt, 'r') as f:
            lines = f.readlines()
            label_list = [item.strip() for item in lines]
    img_list = get_all_images(img_f)
    img_dict = {}
    for img_item in img_list:
        img_name = os.path.basename(img_item)
        img_dict[img_name] = img_item

    data = read_jsons(result_json_or_folder)
    bar = tqdm(total=len(data))
    for line in data:
        bar.update()
        line = json.loads(line)
        img_name = line['url'].split('?')[0].split('/')[-1]
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        img_dst = os.path.join(img_dst_f, img_name)
        if img_name in img_dict:
            img_src = img_dict[img_name]
            shutil.copy(img_src, img_dst)
        else:
            logger.info(f'not found {img_name}, download from remote')
            new_name = hashlib.md5(json.dumps(line).encode('utf-8')).hexdigest() + os.path.splitext(img_name)[1]
            img_dst = os.path.join(img_dst_f, new_name)
            txt_name = os.path.splitext(new_name)[0] + '.txt'
            if not os.path.exists(img_dst):
                download_img(line['url'], img_dst, reformat='jpg')
            img_dst = os.path.splitext(img_dst)[0] + '.jpg'

        anns = line['annotated_zones']

        lines = []

        if len(anns) >= 1:
            for ann in anns:
                if 'tags' in ann:
                    label = ann['tags'][0]
                else:
                    label = ann['tag']
                label = label.lower()
                if label in ['*', 'normal3', 'notsure3', 'background', 'normal', 'notsure']:
                    # if label not in label2id:
                    print(f'skip label:{label}')
                    continue
                if label in new_to_old:
                    label = new_to_old[label]
                    # print(label)
                points = ann['points']
                img = cv2.imread(img_dst)
                if img is None:
                    logger.warning(f'img not found: {img_dst}')
                    # print('skip:', img_dst)
                    continue
                h, w = img.shape[:2]
                x0, y0, x1, y1 = bbox_from_points(points)
                centerx, centery, w, h = _bbox_2_yolo([x0, y0, x1 - x0, y1 - y0], w, h)

                if label not in label_list:
                    label_list.append(label)
                label_index = label_list.index(label)

                line = f'{label_index} {centerx} {centery} {w} {h}\n'
                lines.append(line)
        if len(lines) > 0:
            with open(os.path.join(txt_dst_f, txt_name), 'w') as f:
                for line in lines:
                    f.write(line)
    with open(os.path.join(save_f, 'classes.txt'), 'w') as f:
        for item in label_list:
            f.write(f'{item}\n')


if __name__ == '__main__':
    import fire

    fire.Fire(main)
