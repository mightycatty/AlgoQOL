# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/7 19:59
@Auth ： heshuai.sec@gmail.com
@File ：paddle_data_utils.py
"""
import os
import random


def split_train_val(txt_dir, ratio=0.2):
    with open(txt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    val_lines = lines[:int(len(lines) * ratio)]
    train_lines = lines[int(len(lines) * ratio):]
    with open(os.path.join(os.path.dirname(txt_dir), 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(os.path.join(os.path.dirname(txt_dir), 'val.txt'), 'w', encoding='utf-8') as f:
        f.writelines(val_lines)


if __name__ == '__main__':
    split_train_val(r'E:\data\game_character\ocr\rec\paddle_ocr_rec.txt')
