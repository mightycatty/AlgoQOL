# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/14 11:16
@Auth ： heshuai.sec@gmail.com
@File ：yolo_utils.py
"""
import os
import random
import shutil

from algo_qol.data.utils.file_utils import get_file_recursively, get_all_images


def _read_ann_txt(txt_dir):
    anns = []
    if not os.path.exists(txt_dir): return anns
    with open(txt_dir, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.split(' ')
            line = [float(item) for item in line]
            line[0] = int(line[0])
            anns.append(line)
    return anns


def count_instance(txt_folder, classes_txt=r'E:\data\politics\detection\train\classes.txt'):
    with open(classes_txt, 'r') as f:
        lines = f.readlines()
        class_list = [item.strip() for item in lines]
    count = {}
    txt_list = get_file_recursively(txt_folder)
    for item in txt_list:
        with open(item, 'r') as f:
            data = f.readlines()
            for line in data:
                line = line.split(' ')
                class_id = int(line[0])
                class_name = class_list[class_id]
                if class_name not in count:
                    count[class_name] = 0
                else:
                    count[class_name] += 1
    count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    with open('results.csv', 'w') as f:
        for idx, item in enumerate(count):
            line = item[0] + ',' + str(item[1]) + '\n'
            f.write(line)
    # print(count)


def remap_class_id(txt_folder, save_f):
    os.makedirs(save_f, exist_ok=True)
    map_dist = {'0': '1', '1': '3', '2': '2', '3': '0'}
    txt_list = get_file_recursively(txt_folder)
    class_list = []
    for item in txt_list:
        txt_name = os.path.basename(item)
        dst = os.path.join(save_f, txt_name)
        with open(dst, 'w') as f_write:
            with open(item, 'r') as f:
                data = f.readlines()
                for line in data:
                    class_id = line.split(' ')[0]
                    new_class_id = map_dist[class_id]
                    print(line)
                    line = new_class_id + line[len(class_id):]
                    print(line)
                    f_write.write(line)


def remove_yolo_none_pair(img_f, label_f=None):
    if label_f is None:
        label_f = img_f.replace('images', 'labels')
    neg_f = os.path.join(os.path.dirname(img_f), 'negative_images')
    os.makedirs(neg_f, exist_ok=True)
    img_list = os.listdir(img_f)
    count = 0
    label_src_list = []
    for item in img_list:
        name, _ = os.path.splitext(item)
        img_src = os.path.join(img_f, item)
        label_src = os.path.join(label_f, name + '.txt')

        labels = _read_ann_txt(label_src)
        try:
            if len(labels) == 0:
                count += 1
                img_dst = os.path.join(neg_f, item)
                label_dst = os.path.join(neg_f, os.path.basename(label_src))
                # print(img_src)
                # print(img_dst)
                shutil.move(img_src, img_dst)
                shutil.move(label_src, label_dst)
                print(label_dst)
            else:
                label_src_list.append(label_src)
        except:
            pass
    txt_list = [os.path.join(label_f, item) for item in os.listdir(label_f)]
    diff_list = list(set(txt_list) - set(label_src_list))
    for item in diff_list:
        os.remove(item)
    print(f'total:{len(img_list)} negative:{count}')


def split_val(base_f, val_num=1000):
    img_f = os.path.join(base_f, 'images')
    label_f = os.path.join(base_f, 'labels')
    img_dst_f = os.path.join(base_f, 'val', 'image')
    label_dst_f = os.path.join(base_f, 'val', 'labels')
    [os.makedirs(item) for item in [img_dst_f, label_dst_f]]

    img_list = os.listdir(img_f)
    random.shuffle(img_list)
    assert val_num < len(img_list)
    img_list = img_list[:val_num]
    for item in img_list:
        img_src = os.path.join(img_f, item)
        img_dst = os.path.join(img_dst_f, item)
        txt_name = os.path.splitext(item)[0] + '.txt'
        label_src = os.path.join(label_f, txt_name)
        label_dst = os.path.join(label_dst_f, txt_name)
        [shutil.move(src, dst) for src, dst in zip([img_src, label_src], [img_dst, label_dst])]


def split_val(img_f=r'E:\data\game_character\detection\images\20240704_h73',
              txt_f=None,
              val_num=200,
              ):
    if txt_f is None:
        txt_f = img_f.replace('images', 'labels')
    img_list = os.listdir(img_f)
    random.shuffle(img_list)
    img_val_f = img_f + '_val'
    txt_val_f = txt_f + '_val'
    os.makedirs(img_val_f, exist_ok=True)
    os.makedirs(txt_val_f, exist_ok=True)
    for img_item in img_list[:val_num]:
        txt_name = img_item.replace(os.path.splitext(img_item)[-1], '.txt')
        src = os.path.join(img_f, img_item)
        dst = os.path.join(img_val_f, img_item)
        shutil.move(src, dst)

        src = os.path.join(txt_f, txt_name)
        if os.path.exists(src):
            dst = os.path.join(txt_val_f, txt_name)
            shutil.move(src, dst)


def merge_two_dataset(dataset_root_0, dataset_root_1, save_f):
    class_name_0 = os.path.join(dataset_root_0, 'classes.txt')
    class_name_1 = os.path.join(dataset_root_1, 'classes.txt')
    os.makedirs(save_f, exist_ok=True)
    with open(class_name_0, 'r') as f:
        class_name_0 = f.readlines()
        class_name_0 = {i: x.strip() for i, x in enumerate(class_name_0)}
    with open(class_name_1, 'r') as f:
        class_name_1 = f.readlines()
        class_name_1 = {i: x.strip() for i, x in enumerate(class_name_1)}
    class_merge = list(class_name_0.values())
    for i in list(class_name_1.values()):
        if i not in class_merge:
            class_merge.append(i)
    with open(os.path.join(save_f, 'classes.txt'), 'w') as f:
        for i, x in enumerate(class_merge):
            f.write(str(x) + '\n')
    image_0_list = get_all_images(os.path.join(dataset_root_0, 'images'))
    image_1_list = get_all_images(os.path.join(dataset_root_1, 'images'))
    image_0_dict = {os.path.basename(x): x for x in image_0_list}
    image_1_dict = {os.path.basename(x): x for x in image_1_list}
    image_names = list(set(list(image_0_dict.keys()) + list(image_1_dict.keys())))
    for img_name in image_names:
        lines = []
        try:
            if img_name in image_0_dict:
                img_src = image_0_dict[img_name]
                img_format = img_src.split('.')[-1]

                txt_dir = image_0_dict[img_name].replace('\\images\\', '\\labels\\').replace('/images/',
                                                                                             '/labels/').strip(
                    img_format) + 'txt'
                with open(txt_dir, 'r') as f:
                    data = f.readlines()
                    for line_item in data:
                        label_idx = line_item.split(' ')[0]
                        label_name = class_name_0[int(label_idx)]
                        line_item = str(class_merge.index(label_name)) + ' ' + line_item[len(label_idx) + 1:]
                        lines.append(line_item)
                        img_dst_f = os.path.join(save_f, 'images',
                                                 os.path.dirname(txt_dir).split('labels')[-1].strip('\\').strip('/'))
                        label_dst_f = os.path.join(save_f, 'labels',
                                                   os.path.dirname(txt_dir).split('labels')[-1].strip('\\').strip('/'))
            if img_name in image_1_dict:
                img_src = image_1_dict[img_name]
                img_format = img_src.split('.')[-1]

                txt_dir = image_1_dict[img_name].replace('\\images\\', '\\labels\\').replace('/images/',
                                                                                             '/labels/').strip(
                    img_format) + 'txt'
                with open(txt_dir, 'r') as f:
                    data = f.readlines()
                    for line_item in data:
                        line_item = line_item.strip()
                        label_idx = line_item.split()[0]
                        label_name = class_name_1[int(label_idx)]
                        line_item = str(class_merge.index(label_name)) + ' ' + line_item[len(label_idx) + 1:]
                        lines.append(line_item)
                        img_dst_f = os.path.join(save_f, 'images',
                                                 os.path.dirname(txt_dir).split('labels')[-1].strip('\\').strip('/'))
                        label_dst_f = os.path.join(save_f, 'labels',
                                                   os.path.dirname(txt_dir).split('labels')[-1].strip('\\').strip('/'))
        except Exception as e:
            print(e)
            # print(img_src)
            continue
        os.makedirs(img_dst_f, exist_ok=True)
        os.makedirs(label_dst_f, exist_ok=True)
        if len(lines) > 0:
            shutil.copy(img_src, os.path.join(img_dst_f, img_name))
            with open(os.path.join(label_dst_f, os.path.basename(txt_dir)), 'w') as f:
                f.writelines(lines)


if __name__ == '__main__':
    remove_yolo_none_pair(r'E:\projects\algo_detection\src\utils\data\convert\politics\images')
    # count_instance(r'E:\data\logo\ads_logo\yolo\labels', r'E:\data\logo\ads_logo\yolo\classes.txt')
    # split_val()
