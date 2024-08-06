# @Time    : 2022/5/8 6:36 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
import logging
import os

import cv2


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


class ImgSaver:
    pass


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


def get_all_images(f):
    image_paths = []
    for root, dirs, files in os.walk(f):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths
