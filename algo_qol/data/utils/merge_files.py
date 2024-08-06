import os
import shutil
import threading
from tqdm import tqdm


def worker(src_folder, dst_folder, copy_only=False):
    bar = tqdm(total=len(os.listdir(src_folder)))
    for filename in os.listdir(src_folder):
        bar.update()
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)

        if os.path.isfile(src_file):
            if copy_only:
                shutil.copy(src_file, dst_file)
            else:
                shutil.move(src_file, dst_file)
        else:
            os.makedirs(dst_file, exist_ok=True)
            worker(src_file, dst_file, copy_only)


def merge_files(src_f, dst_f, copy_only=False):
    """
    merge all files in a folder(include sub-folders) to a new folder

    Args:
        src_f:
        dst_f:
        copy_only: only copy files instead of moving

    Returns:

    """
    src_folders = os.listdir(src_f)
    src_folders = [os.path.join(src_f, i) for i in src_folders]
    os.makedirs(dst_f, exist_ok=True)

    threads = []
    for src_folder in src_folders:
        thread = threading.Thread(target=worker, args=(src_folder, dst_f, copy_only))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All files copied successfully!")
