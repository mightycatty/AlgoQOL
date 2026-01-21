# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/21 16:44
@Auth ： heshuai.sec@gmail.com
@File ：utils.py
"""


def cxcywh_to_xyxy(cxcywh: list, w: int = None, h: int = None) -> list:
    """
    Convert cxcywh to xyxy.
    Args:
        cxcywh: normalized cxcywh
    Returns:

    """
    x0 = cxcywh[0] - cxcywh[2] / 2
    y0 = cxcywh[1] - cxcywh[3] / 2
    x1 = cxcywh[0] + cxcywh[2] / 2
    y1 = cxcywh[1] + cxcywh[3] / 2
    ret = [x0, y0, x1, y1]
    # clip
    ret = [min(max(item, 0.), 1.) for item in ret]
    if w is not None and h is not None:
        ret[0] = w * ret[0]
        ret[1] = h * ret[1]
        ret[2] = w * ret[2]
        ret[3] = h * ret[3]
        ret = [int(item) for item in ret]
    return ret
