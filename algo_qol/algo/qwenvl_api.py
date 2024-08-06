# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/14 10:18
@Auth ： heshuai.sec@gmail.com
@File ：qwenvl.py

ref: https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api?spm=a2c4g.11186623.0.0.3f5c28c4i0hOB1#BQnl3
max_tokens: 6000
"""
import os

import dashscope
from dashscope import MultiModalConversation

dashscope.api_key = "sk-3c8beae63c2b48c5b5feaa210c53ee32"


def qwenvl_call(img_list,
                prompt='画面中是否存在字体，或者类似字体的物品。请找出起位置，并识别为什么字体。请以json格式返回，两个字段，character_exit，以及text。勿返回json以外的内容',
                verbose=False):
    if not isinstance(img_list, list):
        img_list = [img_list]
    images_content = [{'image': item} for item in img_list]
    messages = [{
        'role': 'user',
        'content': [
            *images_content,
            {
                'text': prompt,
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
    if verbose:
        print(response)
    return response.output.choices[0].message.content[0]['text']


if __name__ == '__main__':
    img_f = r'/logs/heshuai03/datasets/game_character/detection/images/20240704_h73_val'
    img_list = [os.path.join(img_f, item) for item in os.listdir(img_f)]
    for item in img_list:
        result = qwenvl_call(item, verbose=False)
        print(result)
