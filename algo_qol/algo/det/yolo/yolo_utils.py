import os
import random
import shutil
from algo_qol.data_utils.file_utils import get_file_recursively, get_all_images


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


def read_ultralytics_yolo_dataset(img_root, txt_root, classes_txt):
    with open(classes_txt, 'r') as f:
        classes = f.readlines()
    classes = [item.strip() for item in classes]
    img_list = get_all_images(img_root)
    for img in img_list:
        sub_f = os.path.relpath(img, img_root)
        txt = os.path.join(txt_root, os.path.splitext(sub_f)[0] + '.txt')
        if not os.path.exists(txt):
            labels = []
        else:
            labels = _read_ann_txt(txt)
        data_item = {
            'image': img,
            'bboxes': [],
            'labels': [],
            'labels_index': [],
        }
        bboxes = []
        labels_ret = []
        labels_index = []
        for item in labels:
            bboxes.append(item[1:])
            labels_ret.append(classes[item[0]])
            labels_index.append(int(item[0]))
        data_item['bboxes'] = bboxes
        data_item['labels'] = labels_ret
        data_item['labels_index'] = labels_index
        yield data_item


def yolo_analyse(txt_root, classes_txt, save=None, verbose=True):
    def _parse_txt(txt):
        with open(txt, 'r', ) as f:
            data = f.readlines()
            id_lis = []
            for line in data:
                line = line.split(' ')
                class_id = int(line[0])
                id_lis.append(class_id)
            return id_lis

    with open(classes_txt, 'r') as f:
        lines = f.readlines()
        class_list = [item.strip() for item in lines]
    txt_list = get_file_recursively(txt_root)
    txt_list = list(filter(lambda x: x.endswith('.txt'), txt_list))
    from tqdm.contrib.concurrent import thread_map
    id_lis = thread_map(_parse_txt, txt_list)
    # flatten id_list
    id_lis = [item for sublist in id_lis for item in sublist]
    # count the results
    from collections import Counter
    count_result = Counter(id_lis)
    count = sorted(count_result.items(), key=lambda x: x[1], reverse=True)
    count = list(count)
    if save:
        with open(save, 'w') as f:
            for idx, item in enumerate(count):
                class_name = class_list[item[0]]
                count = item[1]
                line = f'{idx}, {class_name}, {count}\n'
                if verbose:
                    print(line)
                f.write(line)


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


def split_val(img_root,
              txt_root=None,
              val_num=500,
              ):
    if txt_root is None:
        txt_root = img_root.replace('images', 'labels')
    img_list = os.listdir(img_root)
    random.shuffle(img_list)
    img_val_f = img_root + '_val'
    txt_val_f = txt_root + '_val'
    os.makedirs(img_val_f, exist_ok=True)
    os.makedirs(txt_val_f, exist_ok=True)
    for img_item in img_list[:val_num]:
        txt_name = img_item.replace(os.path.splitext(img_item)[-1], '.txt')
        src = os.path.join(img_root, img_item)
        dst = os.path.join(img_val_f, img_item)
        shutil.move(src, dst)
        src = os.path.join(txt_root, txt_name)
        if os.path.exists(src):
            dst = os.path.join(txt_val_f, txt_name)
            shutil.move(src, dst)
