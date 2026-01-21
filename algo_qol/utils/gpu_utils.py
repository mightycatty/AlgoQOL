# @Time    : 2022/4/1 5:44 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
import logging
import os

import cv2


# TODO: BUGs, "failed call to cuInit: CUDA_ERROR_NOT_INITIALIZED: initialization error"
class RunAsCUDASubprocess:
    """
    transparent gpu management for tensorflow and other gpu-required applications.
    1. make desired number of gpus visible to tensorflow
    2. completely release gpu resource when tf session closed
    3. select desired number of gpus with at least fraction of memory

    Credit to ed-alertedh:
        https://gist.github.com/ed-alertedh/85dc3a70d3972e742ca0c4296de7bf00
    """

    def __init__(self, num_gpus=0, memory_fraction=0.95, verbose=False):
        self._num_gpus = num_gpus
        self._memory_fraction = memory_fraction
        if not verbose:
            logging.getLogger('py3nvml.utils').setLevel(logging.ERROR)  # mute py3nvml logging info

    @staticmethod
    def _subprocess_code(num_gpus, memory_fraction, fn, args):
        # set the env vars inside the subprocess so that we don't alter the parent env
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see tensorflow issue #152
        try:
            import py3nvml
            num_grabbed = py3nvml.grab_gpus(num_gpus, gpu_fraction=memory_fraction)
        except Exception as e:
            print(e)
            print('\n try "pip install py3nvml" and try again')
            exit(0)
            # either CUDA is not installed on the system or py3nvml is not installed (which probably means the env
            # does not have CUDA-enabled packages). Either way, block the visible devices to be sure.
            # num_grabbed = 0
            # os.environ['CUDA_VISIBLE_DEVICES'] = ""

        assert num_grabbed == num_gpus, 'Could not grab {} GPU devices with {}% memory available'.format(
            num_gpus,
            memory_fraction * 100)
        if os.environ['CUDA_VISIBLE_DEVICES'] == "":
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # see tensorflow issues: #16284, #2175

        # using cloudpickle because it is more flexible about what functions it will
        # pickle (lambda functions, notebook code, etc.)
        return cloudpickle.loads(fn)(*args)

    def __call__(self, f):
        def wrapped_f(*args):
            with Pool(1) as p:
                return p.apply(RunAsCUDASubprocess._subprocess_code,
                               (self._num_gpus, self._memory_fraction, cloudpickle.dumps(f), args))

        return wrapped_f


def run_command(cmd, verbose=False):
    """
    run commands , Linux only
    :param cmd:
    :return: string output if success otherwise False
    """
    import subprocess
    try:
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0].decode("ascii")
        if verbose:
            print(output)
        return output
    except Exception as e:
        print(e)
        return False


# ====================================== gpu utilities =========================================================
def list_available_gpus(verbose=False):
    """
    list available gpus and return id list by running cmd "nvidia-smi" in Linux
    :param verbose:
    :return:
    """
    import re
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    if verbose:
        print(result)
    return result


def gpu_memory_map(verbose=False):
    import re
    """Returns map of GPU id to memory allocated on that GPU.
    working on ubuntu 16 with CUDA TOOLKIT10.2
    """
    output = run_command("nvidia-smi")
    gpu_output = output[output.find("Memory-Usage"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    gpu_id = 0
    for row in gpu_output.split("\n"):
        # m = memory_regex.search(row)
        if '%' in row:
            gpu_memory = int(row.split('MiB /')[0][-5:])
            result[gpu_id] += gpu_memory
            gpu_id += 1
        # if not m:
        #     continue
        # gpu_id = int(m.group("gpu_id"))
        # gpu_memory = int(m.group("gpu_memory"))
        # result[gpu_id] += gpu_memory
    if verbose:
        print(result)
    return result


def pick_n_gpu_lowest_memory(n=1):
    """Returns GPU with the least allocated memory"""
    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_gpu = []
    for item in sorted(memory_gpu_map)[:n]:
        if item[0] < 500:  # 500MB
            best_gpu.append(item[1])
    if len(best_gpu) == n:
        return best_gpu
    else:
        print('not enough gpus available')
        exit(0)


def set_gpus_visiable(gpu_num=1, verbose=True):
    import os
    best_gpu = pick_n_gpu_lowest_memory(gpu_num)
    if isinstance(best_gpu, list):
        assert len(best_gpu) >= gpu_num, 'not enough gpus found'
        gpu_id_str = str(best_gpu[0])
        for i in best_gpu[1:]:
            gpu_id_str += ',{}'.format(i)
            if (i + 2) >= gpu_num:
                break
    else:
        gpu_id_str = str(best_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    if verbose:
        print('Using GPUs:{}'.format(gpu_id_str))
    return


def say_hi_to_your_program():
    """
    python program initialization, run before the main function in your python program
    1. clean up console, filter warnings
    :return:
    """
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore',
                          category=FutureWarning)  # disable nasty future warning in tensorflow and numpy


def disable_gpu():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def select_n_gpus(num=1, max_load=0.1, max_mem=0.1):
    import os
    try:
        import GPUtil as gputil
    except:
        print('import GPUtil error, "pip install gputil" and try again')
    available_gpu_ids = gputil.getAvailable(order='first', limit=num, maxLoad=max_load, maxMemory=max_mem,
                                            includeNan=False,
                                            excludeID=[], excludeUUID=[])
    if len(available_gpu_ids) < num:
        print('not enough gpus found!')
        exit(0)
    else:
        if num == 0:
            gpu_str = '-1'
        else:
            gpu_str = ''
            for i in range(num):
                gpu_str += available_gpu_ids[i]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str


class EasyDict(dict):
    from typing import Any
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


# =============================================others ========================================================


def flat_list(list_input):
    """Flatten nested list"""
    list_flatten = []
    for item in list_input:
        if isinstance(item, list):
            list_flatten += flat_list(item)
        else:
            list_flatten += [item]
    return list_flatten


# ============================================ file relative ===============================================
# TODO: 异步模式
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
    try:
        cap = acapture.open(video_dir, loop=loop)  # RGB
        cvt_format = 'RGB'
    except Exception as e:
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
            error_count += 1
            if error_count > max_error_num:
                cap.release()
                return
            if loop and (type(video_dir) is str):
                del cap
                cap = cv2.VideoCapture(video_dir)
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                cap.release()
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


# TODO: threading to save img to disk
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


# TODO: more efficient implementation
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


if __name__ == '__main__':
    import requests
    import json

    url = r'http://niv-api.gameyw.netease.com:8080/set_backtrace_pic_status'
    d = {
        "product": "TESTP2",
        "data": [
            {
                "url": "http://egg-head.fp.ps.netease.com/file/6630c39dd84c26202d0665easwghZpZm05",
                "md5": "d7e5de72e6a88ffe9cb65c4684a83ae0",
                "id": 538559435,
                "reason": {
                    "porn": ["xinganshi", "xingshinvxia"],
                    "politic_info": ["flag_nazi"],
                    "subject": [],
                    "politician": [],
                    "content_check": [],
                    "advertise": [],
                    "character_exist_detect": [],
                    "qrcode": []
                }
            }
        ]
    }
    r = requests.post(url, data=json.dumps(d))
    print(r.content.decode('utf-8'))
