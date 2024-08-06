import os
from uuid import uuid4
from tqdm import tqdm


def get_all_images(f):
    image_paths = []
    for root, dirs, files in os.walk(f):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def rename_files(src_f):
    """
    rename all images in a folder to uuid new name

    Args:
        src_f: support sub-folders

    Returns:

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


if __name__ == '__main__':
    base_f = r'E:\downloads\archive\Dataset'
    # rename_entry(base_f)
    sub_f = os.listdir(base_f)
    for sub_f_item in sub_f:
        src_f = os.path.join(base_f, sub_f_item)
        rename_entry(src_f)
    # rename_seg_pairs(img_f)
