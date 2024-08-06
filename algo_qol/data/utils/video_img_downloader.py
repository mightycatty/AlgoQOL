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

"""
usage:
python video_img_downloader.py your_csv_or_txt, save_folder
"""


def download_img(img_url, imgpath, reformat=None):
    if os.path.exists(imgpath):
        return
    r = requests.get(img_url)
    try:
        with open(imgpath, 'wb') as f:
            f.write(r.content)
        if reformat is not None:
            img = cv2.imread(imgpath)
            new_dst = os.path.splitext(imgpath)[0] + '.' + reformat
            os.remove(imgpath)
            cv2.imwrite(new_dst, img)
    except Exception as e:
        print(e)
        pass


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
        if 'csv' in txt_src or 'xlsx' in txt_src:
            csv_reader = csv.reader(open(txt_src))
            line_txt = next(csv_reader)
            data = [item[col_index] for item in csv_reader]
        elif 'txt' in txt_src:
            with open(txt_src, 'r') as f:
                data = f.readlines()
                data = [item.strip() for item in data]
                # data = [item.split()[col_index] for item in data]
        else:
            print('not support format')
            exit(0)
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
        for i in range(0, thread_num):
            t = threading.Thread(target=self.loop, name='LoopThread %s' % i, args=(self._img_gen,))
            t.start()


def main(txt_or_csv_src, save_f, thread_num=16, col_index=0, remove_http_head=True):
    """
    download images or videos from give txt or csv
    Args:
        txt_or_csv_src: each line is an image url
        save_f:
        thread_num:
        col_index: the index of column for url in csv
        remove_http_head: image will saved with name without http head

    Returns:

    """
    d = Downloader(txt_or_csv_src, save_f, col_index, remove_http_head)
    d(thread_num)
    return


if __name__ == '__main__':
    import fire

    fire.Fire(main)
