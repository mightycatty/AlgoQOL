import os
import shutil
import wave
from uuid import uuid4

import cv2
import numpy as np
import requests
from PIL import Image
from decord import VideoReader, cpu
from loguru import logger
from skimage import io
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.concurrent import thread_map
import uuid


def convert_image_format_worker(args):
    src, dst_format = args
    try:
        img = io.imread(src)
        if np.ndim(img) == 3:
            img = img[:, :, :3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_format = os.path.splitext(src)[1]
        dst = src.replace(img_format, '.' + dst_format)
        if src != dst:
            os.remove(src)
        io.imsave(dst, img)
    except Exception as e:
        print(e)
        print(src)


def convert_image_format(src_f: str, dst_format: str = 'jpg', threads_num: int = 16):
    """
    convert all images in a folder to dst format, support sub-folders
    """
    img_list = get_all_images(src_f)
    args = [(src, dst_format) for src in img_list]
    process_map(convert_image_format_worker, args, max_workers=threads_num)


def rm_corrupted_image_worker(src, move_out_folder=None):
    try:
        img = Image.open(src)
        if img is None: raise ValueError
    except Exception as e:
        if move_out_folder is not False:
            dst = os.path.join(move_out_folder, os.path.basename(src))
            shutil.move(src, dst)
        else:
            os.remove(src)


def rm_corrupted_image(src_f: str,
                       threads_num: int = 128, move_out: bool = False):
    """
    remove corrupted images
    """
    img_list = get_all_images(src_f)
    logger.info(f'total:{len(img_list)}')
    if move_out:
        move_out = os.path.join(os.path.dirname(src_f), 'corrupted')
        os.makedirs(move_out, exist_ok=True)
    thread_map(lambda x: rm_corrupted_image_worker(x, move_out), img_list, max_workers=threads_num)
    return


def get_all_images(f):
    image_paths = []
    for root, dirs, files in os.walk(f):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def rename_files(src_f: str):
    """
    rename all images in a folder to uuid new name
    """
    img_list = get_all_images(src_f)
    bar = tqdm(total=len(img_list))
    for src in img_list:
        bar.update()
        try:
            format = os.path.splitext(src)[-1]
            new_name = str(uuid4()) + format
            dst = os.path.join(os.path.dirname(src), new_name)
            # print(src, dst)
            os.rename(src, dst)
        except Exception as e:
            print(e)
            print(src)


def resize_img_entry(args):
    img_src, img_dst, max_size = args
    if img_dst is None:
        img_dst = img_src
    try:
        img = Image.open(img_src)
        ratio = max_size / max(img.size)
        if ratio < 1:
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
            img.save(img_dst)
        return True
    except Exception as e:
        logger.error('fail to resize image:{}'.format(img_src))
        logger.error(e)
        return False


def resize_image(img_src: str, img_dst=None, max_size: int = 720, max_workers: int = 16):
    img_list = get_all_images(img_src)
    params = [(src, src, max_size) for src in img_list]
    logger.info('total {} images to resize'.format(len(params)))
    results = process_map(resize_img_entry, params, max_workers=max_workers)
    return


def download_video(url, save_path, timeout=1):
    try:
        data = requests.get(url, timeout=timeout, stream=True)
        data.raise_for_status()
        with open(save_path, 'wb') as fp:
            count = 0
            for chunk in data.iter_content(chunk_size=2 * 1024 * 1024, decode_unicode=False):
                if chunk:
                    fp.write(chunk)
                    count += 1
                else:
                    break
    except Exception as e:
        print(e)
        return False
    return True


def split_audio(wav_path, output_dir, max_duration=10):
    split_results = []
    with wave.open(wav_path, "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        max_samples = int(max_duration * sample_rate)
        frames = wav_file.readframes(wav_file.getnframes())
        start = 0
        while start < len(frames):
            end = min(len(frames), start + max_samples)
            output_filename = os.path.join(output_dir, f"segment_{start // max_samples}.wav")
            split_results.append(output_filename)
            with wave.open(output_filename, "wb") as output_wav_file:
                output_wav_file.setnchannels(num_channels)
                output_wav_file.setsampwidth(sample_width)
                output_wav_file.setframerate(sample_rate)
                output_wav_file.writeframes(frames[start:end])
            start = end
    return split_results


def load_video(video_path: str, max_frames: int = 32) -> np.ndarray:
    def _save_images_tmp(img_list):
        tmp_save_names = [uuid.uuid4().hex + '.jpg' for _ in range(len(img_list))]
        for i in range(len(img_list)):
            tmp_save_name = tmp_save_names[i]
            img_list[i] = img_list[i].astype('uint8')
            cv2.imwrite(tmp_save_name, img_list[i])
        return tmp_save_names

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    if max_frames > 0:
        if max_frames > total_frame_num:
            max_frames = total_frame_num
            logger.warning('num_segments is larger than total_frame_num, set num_segments to total_frame_num')
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    else:
        frame_idx = [i for i in range(total_frame_num)]
    try:
        spare_frames = vr.get_batch(frame_idx).asnumpy()
    except:
        spare_frames = vr.get_batch(frame_idx).numpy()
    return spare_frames
