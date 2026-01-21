import uuid
import cv2
import imageio
from skimage import io
import numpy as np
from imagededup.methods import PHash, CNN
import os
import shutil
from tqdm import tqdm
from imagededup.utils import CustomModel
import torch
from transformers import AutoProcessor

VIT_MODEL = "google/vit-base-patch16-224-in21k"

HANDLE = None


def get_transform(model_name='DeepGlint-AI/mlcd-vit-bigG-patch14-448', *args, **kwargs):
    return AutoProcessor.from_pretrained(model_name)


def init_handle(method='phash'):
    global HANDLE
    if method == 'phash':
        HANDLE = PHash()
    elif method == 'cnn':
        custom_config = CustomModel(name=HFModel.name,
                                    model=HFModel(),
                                    transform=HFModel.transform)
        HANDLE = CNN(model_config=custom_config)


class HFModel(torch.nn.Module):
    name = 'HFModel'

    def __init__(self, model='DeepGlint-AI/mlcd-vit-bigG-patch14-448', *args, **kwargs):
        super().__init__()
        self.name = model
        if self.model == 'DeepGlint-AI/mlcd-vit-bigG-patch14-448':
            from transformers import AutoProcessor, MLCDVisionModel
            model = MLCDVisionModel.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")
            processor = AutoProcessor.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        with torch.no_grad():
            out = self.vit(pixel_values=x)
        return out.pooler_output


def dedup_image_entry(src_f: str, distance: int = 1, debug: bool = False, move_out_f=None):
    src_f = src_f.rstrip('/').rstrip('\\')
    if move_out_f is not None:
        os.makedirs(move_out_f, exist_ok=True)
    # 生成图像目录中所有图像的二值hash编码
    encodings = HANDLE.encode_images(image_dir=src_f)

    # 对已编码图像寻找重复图像
    # duplicates = handle.find_duplicates(encoding_map=encodings, scores=True)
    if debug:
        duplicates_to_remove = HANDLE.find_duplicates(encoding_map=encodings, max_distance_threshold=distance)
        for item in duplicates_to_remove.keys():
            dp = duplicates_to_remove[item]
            if len(dp) > 0:
                try:
                    img_list = [os.path.join(src_f, item)] + dp
                    img_list = [os.path.join(src_f, item_) for item_ in img_list]
                    img_list = [imageio.imread(item_) for item_ in img_list]
                    img_list = [cv2.resize(item_, (128, 256))[:, :, :3] for item_ in img_list]
                    vis = np.hstack(img_list)
                    save_name = os.path.join(move_out_f, str(uuid.uuid4()) + '.jpg')
                    io.imsave(save_name, vis)
                except Exception as e:
                    print(e)
                    continue
                # plt.imshow(vis)
                # plt.show()
    else:
        duplicates_to_remove = HANDLE.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=distance)
        bar = tqdm(total=len(duplicates_to_remove))
        for item in duplicates_to_remove:
            bar.update()
            src = os.path.join(src_f, item)
            if move_out_f is not None:
                dst = os.path.join(move_out_f, item)
                shutil.move(src, dst)
            else:
                os.remove(src)


def dedup_image(src_f: str, distance: int = 5, move_out_f: str = None, debug: bool = False, recursive: bool = True,
                method='phash'
                ):
    global HANDLE
    if HANDLE is None:
        init_handle(method)
    src_f = src_f.rstrip('/').rstrip('\\')
    file_list = os.listdir(src_f)
    sub_dir = list(filter(lambda x: os.path.isdir(os.path.join(src_f, x)), file_list))
    if len(sub_dir) == 0:
        try:
            dedup_image_entry(src_f, distance, debug, move_out_f)
        except:
            print('fail to dedup ' + src_f)
    elif recursive:
        for item in sub_dir:
            dedup_image(os.path.join(src_f, item), distance, move_out_f, debug, recursive)
    if move_out_f is not None and os.path.exists(move_out_f):
        shutil.rmtree(move_out_f)
