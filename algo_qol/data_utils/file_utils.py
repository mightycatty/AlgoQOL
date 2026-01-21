# @Time    : 2022/5/8 6:36 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
import logging
import os
import shutil
import uuid
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


def get_all_images(f):
    _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif", 'jfif')
    image_paths = []
    for root, dirs, files in os.walk(f):
        for file in files:
            if file.endswith(_IMAGE_EXTS):
                image_paths.append(os.path.join(root, file))
    return image_paths


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


def load_video(video_path, num_segments=32, save_tmp=False):
    def _save_images_tmp(img_list):
        tmp_save_names = [uuid.uuid4().hex + '.jpg' for _ in range(len(img_list))]
        for i in range(len(img_list)):
            tmp_save_name = tmp_save_names[i]
            img_list[i] = img_list[i].astype('uint8')
            cv2.imwrite(tmp_save_name, img_list[i])
        return tmp_save_names

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    # fps = round(vr.get_avg_fps())
    # frame_idx = [i for i in range(0, len(vr), fps)]
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_segments, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    try:
        spare_frames = vr.get_batch(frame_idx).asnumpy()
    except:
        spare_frames = vr.get_batch(frame_idx).numpy()
    if save_tmp:
        spare_frames_path = _save_images_tmp(spare_frames)
        return spare_frames, spare_frames_path
    return spare_frames


def video_reader(video_dir, loop=False, *args, **kwargs):
    """video reader generator, each run yields a img of RGB format
    acapture is recommended over cv2, which yields a better streaming performance. however, if you don't bother to install it,
     just stick with OpenCV.
    """
    try:
        import acapture
    except ImportError:
        pass
    import cv2
    # try:
    #     cap = acapture.open(video_dir, loop=loop)  # RGB
    #     cvt_format = 'RGB'
    # except Exception as e:
    cap = cv2.VideoCapture(video_dir)  # BGR
    cvt_format = 'BGR'
    error_count = 0
    max_error_num = 5
    while True:
        ret, frame = cap.read()
        if ret:
            if cvt_format == 'BGR':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        else:
            # error_count += 1
            # if error_count > max_error_num:
            #     cap.release()
            #     return
            if loop and (type(video_dir) is str):
                del cap
                cap = cv2.VideoCapture(video_dir)
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                try:
                    cap.release()
                except:
                    pass
                return


class VideoWriter:
    """
    usage:
    with VideoWriter('test.mp4') as w:
        img = None # BRG
        w.write_frame(img, verbose=False)
    """

    def __init__(self, save_name, fps=30, *args, **kwargs):
        self.save_name = save_name
        self.fps = fps
        self._out_size = None
        self._video_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._video_writer.release()
        finally:
            pass

    def __del__(self):
        try:
            self._video_writer.release()
        finally:
            pass

    def reset(self):
        self._video_writer.release()
        self._video_writer = None

    def _video_writer_init(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(self.save_name, fourcc, self.fps, self._out_size)

    def write_frame(self, image, verbose=False):
        try:
            if not self._video_writer:
                self._out_size = (image.shape[1], image.shape[0])
                self._video_writer_init()
            assert (image.shape[0] == self._out_size[1]) & (
                    image.shape[1] == self._out_size[0]), 'image shape not compilable with video saver shape'
            self._video_writer.write(image)
            if verbose:
                cv2.namedWindow("video_writer", cv2.WINDOW_NORMAL)
                cv2.imshow('video_writer', image)
                cv2.waitKey(1)
        except Exception as e:
            logging.error('write frame error:{}').format(e)
            return False
        return True

    def release(self):
        self._video_writer.release()


def get_latest_file(folder):
    """get latest created file, helpful when finding your latest model checkpoint
    """
    import os
    file_list = os.listdir(folder)
    latest_file = max(file_list, key=os.path.getctime)
    latest_file = os.path.join(folder, latest_file)
    return latest_file


def get_file_recursively(folder_dir):
    """
    iteratively get file list under folder_dir
    :param folder_dir: folder
    :return: a list of files
    """
    file_list = []
    for root, dirs, files in os.walk(folder_dir, topdown=False):
        for name in files:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
        for name in dirs:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
    return file_list


def read_textlines(txt_file):
    with open(txt_file) as f:
        provided_id_list = f.readlines()
    val_list = [item.strip('\n') for item in provided_id_list]
    return val_list


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


def rm_corrupted_image(src_f: str,
                       move_out: bool = False,
                       threads_num: int = 128):
    """
    remove corrupted images
    """

    def rm_corrupted_image_worker(src, move_out_folder=None):
        try:
            img = Image.open(src)
            if img is None: raise ValueError
            return 0
        except Exception as e:
            if move_out_folder is not False:
                dst = os.path.join(move_out_folder, os.path.basename(src))
                shutil.move(src, dst)
            else:
                os.remove(src)
            return 1

    img_list = get_all_images(src_f)
    logger.info(f'total:{len(img_list)}')
    if move_out:
        move_out = os.path.join(os.path.dirname(src_f), 'corrupted')
        os.makedirs(move_out, exist_ok=True)
    results = thread_map(lambda x: rm_corrupted_image_worker(x, move_out), img_list, max_workers=threads_num)
    # count the number that is corrupted
    count = 0
    for result in results:
        if result == 1:
            count += 1
    logger.info(f'total:{len(img_list)}, corrupted:{count}')
    return


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


def _resize_img_entry(args):
    img_src, img_dst, max_size = args
    if img_dst is None:
        img_dst = img_src
    try:
        img = Image.open(img_src)
        img = img.convert('RGB')
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
    """
    resize image

    Args:
        img_src:
        img_dst:
        max_size: default 720
        max_workers:

    Returns:

    """

    img_list = get_all_images(img_src)
    params = [(src, src, max_size) for src in img_list]
    logger.info('total {} images to resize'.format(len(params)))
    results = process_map(_resize_img_entry, params, max_workers=max_workers)
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


def merge_files(src_f: str, dst_f: str, max_workers: int = 32, copy_only: bool = False, rename: bool = False,
                append_sub_folder_name: bool = False):
    """
    merge all files in a folder(include sub-folders) to a new folder
    """

    def worker(src_file, dst_folder, copy_only=False, rename=False, append_sub_folder_name=False):
        file_name = os.path.basename(src_file)
        if rename:
            # append sub-folder name to image name
            file_name = f'{uuid.uuid4()}.{file_name.split(".")[-1]}'
        if append_sub_folder_name:
            sub_folder_list = os.path.normpath(src_file).split(os.sep)[:-1]
            sub_f_name = sub_folder_list[-2:][::-1]
            for sub_f_name in sub_f_name:
                file_name = f'{sub_f_name}_{file_name}'
        dst_file = os.path.join(dst_folder, file_name)
        if copy_only:
            shutil.copy(src_file, dst_file)
        else:
            shutil.move(src_file, dst_file)

    src_list = get_file_recursively(src_f)
    os.makedirs(dst_f, exist_ok=True)
    thread_map(worker, src_list, [dst_f] * len(src_list), [copy_only] * len(src_list), [rename] * len(src_list),
               [append_sub_folder_name] * len(src_list),
               max_workers=max_workers)
    print("All files copied successfully!")


def _copy_file_worker(args):
    src, dst = args
    try:
        shutil.copy(src, dst)
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")


def sample_files(src_f: str, dst_f: str, sample_num: int = 1000, threads_num: int = 16):
    import random
    os.makedirs(dst_f, exist_ok=True)
    file_list = get_file_recursively(src_f)
    logger.info(f'total files:{len(file_list)}')
    if len(file_list) > sample_num:
        file_list = random.sample(file_list, sample_num)

    args = []
    for src in file_list:
        dst = os.path.join(dst_f, os.path.basename(src))
        args.append((src, dst))

    process_map(_copy_file_worker, args, max_workers=threads_num)