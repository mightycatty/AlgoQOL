#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""images crawler from google and baidu

Requirement:
1. [Google Image scraper](https://github.com/hardikvasa/google-images-download/issues/325)
2. [BaiduImageSpider](https://github.com/kong36088/BaiduImageSpider/blob/master/crawling.py)
"""
import argparse
import os
import re
import sys
import urllib
import json
import socket
import urllib.request
import urllib.parse
import urllib.error
# 设置超时
import time
from uuid import uuid4
import string
import random

timeout = 5
socket.setdefaulttimeout(timeout)


class BaiduCrawler:
    # 睡眠时长
    __time_sleep = 0.1
    __amount = 0
    __start_amount = 0
    __counter = 0
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0', 'Cookie': ''}
    __per_page = 30

    # 获取图片url内容等
    # t 下载图片时间间隔
    def __init__(self, t=0.1):
        self.time_sleep = t

    # 获取后缀名
    @staticmethod
    def get_suffix(name):
        m = re.search(r'\.[^\.]*$', name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    @staticmethod
    def handle_baidu_cookie(original_cookie, cookies):
        """
        :param string original_cookie:
        :param list cookies:
        :return string:
        """
        if not cookies:
            return original_cookie
        result = original_cookie
        for cookie in cookies:
            result += cookie.split(';')[0] + ';'
        result.rstrip(';')
        return result

    # 保存图片
    def save_image(self, rsp_data, word):
        os.makedirs(self.__save_folder, exist_ok=True)
        # 判断名字是否重复，获取图片长度
        self.__counter = len(os.listdir(self.__save_folder)) + 1
        for image_info in rsp_data['data']:
            try:
                if 'replaceUrl' not in image_info or len(image_info['replaceUrl']) < 1:
                    continue
                obj_url = image_info['replaceUrl'][0]['ObjUrl']
                thumb_url = image_info['thumbURL']
                url = 'https://image.baidu.com/search/down?tn=download&ipn=dwnl&word=download&ie=utf8&fr=result&url=%s&thumburl=%s' % (
                    urllib.parse.quote(obj_url), urllib.parse.quote(thumb_url))
                time.sleep(self.time_sleep)
                suffix = self.get_suffix(obj_url)
                # 指定UA和referrer，减少403
                opener = urllib.request.build_opener()
                opener.addheaders = [
                    ('User-agent',
                     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'),
                ]
                urllib.request.install_opener(opener)
                # 保存图片
                filepath = '%s/%s' % (self.__save_folder, str(self.__counter) + "_" + str(uuid4()) + str(suffix))
                urllib.request.urlretrieve(url, filepath)
                if os.path.getsize(filepath) < 5:
                    print("下载到了空文件，跳过!")
                    os.unlink(filepath)
                    continue
            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                continue
            except Exception as err:
                time.sleep(1)
                print(err)
                print("产生未知错误，放弃保存")
                continue
            else:
                print("sample+1,已有" + str(self.__counter) + "张sample")
                self.__counter += 1
        return

    # 开始获取
    def get_images(self, word):
        search = urllib.parse.quote(word)
        # pn int 图片数
        pn = self.__start_amount
        while pn < self.__amount:
            url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%s&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word=%s&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn=%s&rn=%d&gsm=1e&1594447993172=' % (
                search, search, str(pn), self.__per_page)
            # 设置header防403
            try:
                time.sleep(self.time_sleep)
                req = urllib.request.Request(url=url, headers=self.headers)
                page = urllib.request.urlopen(req)
                self.headers['Cookie'] = self.handle_baidu_cookie(self.headers['Cookie'],
                                                                  page.info().get_all('Set-Cookie'))
                rsp = page.read()
                page.close()
            except UnicodeDecodeError as e:
                print(e)
                print('-----UnicodeDecodeErrorurl:', url)
            except urllib.error.URLError as e:
                print(e)
                print("-----urlErrorurl:", url)
            except socket.timeout as e:
                print(e)
                print("-----socket timout:", url)
            else:
                # 解析json
                try:
                    rsp_data = json.loads(rsp, strict=False)
                    if 'data' not in rsp_data:
                        print("触发了反爬机制，自动重试！")
                    else:
                        self.save_image(rsp_data, word)
                        # 读取下一页
                        # print("下载下一页")
                        pn += self.__per_page
                except Exception as e:
                    print(e)
                    print('unknow error')
        print("下载任务结束")
        return

    def start(self, word, total_page=1, start_page=1, per_page=30, save_folder=None):
        """
        爬虫入口
        :param word: 抓取的关键词
        :param total_page: 需要抓取数据页数 总抓取图片数量为 页数 x per_page
        :param start_page:起始页码
        :param per_page: 每页数量
        :return:
        """
        self.__per_page = per_page
        self.__start_amount = (start_page - 1) * self.__per_page
        self.__amount = total_page * self.__per_page + self.__start_amount
        self.__save_folder = os.path.join(save_folder, word) if word else word
        self.get_images(word)


def baidu():
    key_words_list = ['GTV']
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--word", type=str, help="抓取关键词", required=True)
        parser.add_argument("-tp", "--total_page", type=int, help="需要抓取的总页数", required=True)
        parser.add_argument("-sp", "--start_page", type=int, help="起始页数", required=True)
        parser.add_argument("-pp", "--per_page", type=int, help="每页大小",
                            choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], default=30, nargs='?')
        parser.add_argument("-d", "--delay", type=float, help="抓取延时（间隔）", default=0.05)
        args = parser.parse_args()

        c = BaiduCrawler(args.delay)
        c.start(args.word, args.total_page, args.start_page,
                args.per_page)  # 抓取关键词为 “美女”，总数为 1 页（即总共 1*60=60 张），开始页码为 2
    else:
        c = BaiduCrawler(0.5)  # 抓取延迟为 0.05

        c.start(key_words_list[0], 100, 10, 30,
                save_folder=save_folder)  # 抓取关键词为 “美女”，总数为 1 页，开始页码为 2，每页30张（即总共 2*30=60 张）


def google(*args, **kwargs):
    from google_images_download import google_images_download  # importing the library
    # todo: workaround for google 400 query limit
    # for item in string.ascii_letters:
    response = google_images_download.googleimagesdownload()  # class instantiation
    arguments = {
        "keywords":
            "cat",
        "limit": 100,
        "print_urls": True,
        'related_images': True,
        # 'format': 'jpg',
        # 'image_directory': save_folder,
        'chromedriver':
            'chromedriver.ext'
    }  # creating list of arguments

    paths = response.download(arguments)  # passing the arguments to the function
    pass


def bing(key_word, save_folder=r'./downloads', retry=5, name='Image'):
    name = uuid4().hex
    # from bing_image_downloader import downloader
    from better_bing_image_downloader import downloader
    downloader(key_word, limit=200, output_dir=save_folder, adult_filter_off=True,
               force_replace=False, timeout=60, filter="", verbose=False, badsites=[], name=name)


if __name__ == '__main__':
    import multiprocessing as mp

    # with open('key_words.txt', 'r', encoding='gbk') as f:
    #     key_words = f.read().splitlines()
    #     key_words = [key_word.strip() for key_word in key_words]
        # key_words = [item+'logo' for item in key_words]
    key_words = ['管制刀具', '二次元刀剑', '大型作战武器', '爆炸物', '爆炸', '动漫火焰', '人体自焚', '肢体冲突', '斩首', '自残', '割腕'
                 '集会', '示威游行', '动漫血腥', '血腥', '动物血腥', '人类尸体', '动物尸体', '恶心']
    subfix_list = ['']
    key_words = [key_word + subfix for key_word in key_words for subfix in subfix_list]
    # bing('cat')
    with mp.Pool(32) as p:
        p.map(bing, key_words)
