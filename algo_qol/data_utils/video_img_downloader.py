# @Time    : 2022/1/12 6:21 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
# multi-thread video/img downloader, adapted from @xiaoxin
import os
import random
import threading
import requests
import csv
import cv2
import hashlib
from requests.auth import HTTPBasicAuth
from loguru import logger

__all__ = [
    'download'
]


def get_md5_encoder_name(img_url):
    """
    rename image with md5 of original name
    """
    or_img_name = os.path.basename(img_url)
    img_format_list = ['jpg', 'png', 'jpeg']
    for format in img_format_list:
        if or_img_name.split('.')[-1].startswith(format):
            break
    or_img_name = '{}.{}'.format(or_img_name.split('.')[0], format)
    return hashlib.md5(or_img_name.encode('utf-8')).hexdigest() + '.' + format


def guess_image_format_from_url(url):
    support_formats = ['jpg', 'png', 'jpeg', 'webp', 'gif', 'tiff', 'bmp', 'jfif']
    for format in support_formats:
        if url.endswith(format):
            return format
        else:
            if '.' + format in url:
                return format
    return None


def download_img_to_folder(img_url, save_f,
                           reformat=None,
                           md5_rename=True,
                           username=None, password=None
                           ):
    """download img to folder, skip if file exists
    """
    or_img_name = os.path.basename(img_url)
    imgpath = os.path.join(save_f, or_img_name)
    or_format = guess_image_format_from_url(img_url)
    if or_format is None:
        logger.error('fail detect image format:{}'.format(img_url))
        return None
    if not imgpath.endswith(or_format):
        imgpath = imgpath + '.' + or_format
        # logger.warning('save format might be incorrect:{}'.format(imgpath))
        # logger.warning('url:{}'.format(img_url))
    if md5_rename:
        imgpath = os.path.join(save_f, hashlib.md5(or_img_name.encode('utf-8')).hexdigest() + '.' + or_format)
    if os.path.exists(imgpath):
        return imgpath
    if username is None or password is None:
        r = requests.get(img_url)
    else:
        r = requests.get(img_url, auth=HTTPBasicAuth(username, password))
    try:
        with open(imgpath, 'wb') as f:
            f.write(r.content)
        if reformat is not None:
            img = cv2.imread(imgpath)
            os.remove(imgpath)
            imgpath = os.path.splitext(imgpath)[0] + '.' + reformat
            cv2.imwrite(imgpath, img)
        return imgpath
    except Exception as e:
        logger.error('fail to download image:{}'.format(img_url) + ' / ' + str(e))
        return None


def download_img(img_url, save_path):
    if os.path.exists(save_path):
        return save_path
    r = requests.get(img_url)
    try:
        with open(save_path, 'wb') as f:
            f.write(r.content)
        return save_path
    except Exception as e:
        print(e)
        return None


class Downloader(object):
    def __init__(self, txt_src, save_f, col_index=0, remove_http_head=True):
        self._lock = threading.Lock()
        os.makedirs(save_f, exist_ok=True)
        self._to_download = 0
        self._download_count = 0
        self._img_gen = self.get_img_url_generate(txt_src, save_f, col_index)
        self._remove_http_head = remove_http_head

    @staticmethod
    def download(img_url, imgpath):
        if os.path.exists(imgpath):
            return
        r = requests.get(img_url)
        try:
            with open(imgpath, 'wb') as f:
                f.write(r.content)
        except:
            pass
            # print('savefail\t %s' % img_url)

    def get_img_url_generate(self, txt_src, save_f, col_index=0):
        if not isinstance(txt_src, list):
            if 'csv' in txt_src or 'xlsx' in txt_src:
                try:
                    with open(txt_src, 'r', encoding='utf-8') as f:
                        csv_reader = csv.reader(f)
                        data = [item[col_index] for item in csv_reader]
                except UnicodeDecodeError:
                    # 如果 UTF-8 不成功，尝试其他编码
                    with open(txt_src, 'r', encoding='latin-1') as f:
                        csv_reader = csv.reader(f)
                        for item in csv_reader:
                            print(item)
                        data = [item[col_index] for item in csv_reader]
            elif 'txt' in txt_src:
                with open(txt_src, 'r') as f:
                    data = f.readlines()
                    data = [item.strip() for item in data]
                    # data = [item.split()[col_index] for item in data]
            else:
                print('not support format')
                exit(0)
        else:
            data = txt_src
        data = list(set(data))  # remove duplicate
        random.shuffle(data)
        self._to_download = len(data)
        for item in data:
            imgs = []
            img_url = item.strip()
            img_url = img_url.replace('"', '')
            if self._remove_http_head:
                save_name = img_url.split('/')[-1]
            else:
                save_name = img_url.replace('https://', '')
                save_name = save_name.replace('/', '-')
            imgpath = os.path.join(save_f, save_name)
            format = imgpath.split('.')[-1]
            if format.lower() not in ['jpg', 'jpeg', 'png']:
                imgpath += '.jpg'
            imgs.append(img_url)
            imgs.append(imgpath)
            try:
                if img_url:
                    yield imgs
            except:
                break

    def loop(self, imgs):
        # print('thread %s is running...' % threading.current_thread().name)
        self._fail_num = 0
        while True:
            if (threading.current_thread().name == 'LoopThread 0'):
                print('downloaded/fail/total:{}/{}/{}'.format(self._download_count, self._fail_num, self._to_download))
            try:
                with self._lock:
                    img_url, imgpath = next(imgs)
            except StopIteration:
                break
            try:
                if not os.path.exists(imgpath):
                    self.download(img_url, imgpath)
                self._download_count += 1
            except Exception as e:
                self._fail_num += 1
                # print(img_url)
                # print(e)
                pass
                # print('exceptfail\t%s' % img_url)
        # print('thread %s is end...' % threading.current_thread().name)

    def __call__(self, thread_num=16, *args, **kwargs):
        print(self._to_download)
        thread_list = []
        for i in range(0, thread_num):
            t = threading.Thread(target=self.loop, name='LoopThread %s' % i, args=(self._img_gen,))
            t.start()
            thread_list.append(t)
        for t in thread_list:
            t.join()


def download(txt_or_csv_or_list, save_f: str, thread_num: int = 16, col_index_or_name=0,
             remove_http_head: bool = True, ):
    """
    download images or videos from give txt or csv
    """
    d = Downloader(txt_or_csv_or_list, save_f, col_index_or_name, remove_http_head)
    d(thread_num)
    return


if __name__ == '__main__':
    import fire

    fire.Fire(download)
