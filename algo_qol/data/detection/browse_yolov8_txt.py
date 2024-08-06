# coding:utf-8
import cv2
import os
import random
import shutil

# optional
IMAGE_NAME_LIST_PATH = './name_list.txt'  # The file name of images will be saved into this text file. 内含有检测图片名的txt文件路径


def get_image_path(img_f, image_name):
    valid_format = ['.jpg', '.png', '.jpeg']
    for v in valid_format:
        dst = os.path.join(img_f, image_name + v)
        if os.path.exists(dst):
            return dst


def plot_one_box(x, image, color=None, label=None, line_thickness=None, score=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        if score is not None:
            score = round(score, 4)
            label = f'{label}_{score}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_box_on_image(image_name, classes, colors, LABEL_FOLDER, RAW_IMAGE_FOLDER, OUTPUT_IMAGE_FOLDER,
                      valid_class_index=None, draw_bbox=False):
    """
    This function will add rectangle boxes on the images.
    """
    txt_path = os.path.join(LABEL_FOLDER, '%s.txt' %
                            (image_name))  # 本次检测结果txt路径
    if image_name == '.DS_Store' or not os.path.exists(txt_path):
        return 0
    image_path = get_image_path(RAW_IMAGE_FOLDER, image_name)
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    save_file_path = os.path.join(
        OUTPUT_IMAGE_FOLDER, '%s.jpg' % (image_name))  # 本次保存图片jpg路径

    # flag_people_or_car_data = 0  #变量 代表类别
    source_file = open(txt_path) if os.path.exists(txt_path) else []
    image = cv2.imread(image_path)
    try:
        height, width, channels = image.shape
    except:
        print('no shape info.')
        return 0

    box_number = 0
    max_score = 0.
    neg = True
    for line in source_file:  # 例遍 txt文件得每一行
        staff = line.split()  # 对每行内容 通过以空格为分隔符对字符串进行切片
        class_idx = int(staff[0])
        if valid_class_index is not None and class_idx not in valid_class_index: continue
        neg = False
        x_center, y_center, w, h = float(
            staff[1]) * width, float(staff[2]) * height, float(staff[3]) * width, float(staff[4]) * height
        x1 = round(x_center - w / 2)
        y1 = round(y_center - h / 2)
        x2 = round(x_center + w / 2)
        y2 = round(y_center + h / 2)
        if len(staff) > 5:
            score = float(staff[-1])
            max_score = max(score, max_score)
        if draw_bbox:
            plot_one_box([x1, y1, x2, y2], image, color=colors[class_idx],
                         label=classes[class_idx], line_thickness=None)
        box_number += 1
    # save max score in file name for data sorting
    if not neg:
        if max_score > 0:
            max_score = int(max_score * 10000)
            name = f'{max_score}_' + os.path.basename(save_file_path)
            dir_name = os.path.dirname(save_file_path)
            save_file_path = os.path.join(dir_name, name)
        cv2.imwrite(save_file_path, image)
    return box_number


def make_name_list(RAW_IMAGE_FOLDER, IMAGE_NAME_LIST_PATH):
    """
    This function will collect the image names without extension and save them in the name_list.txt.
    """
    image_file_list = os.listdir(RAW_IMAGE_FOLDER)  # 得到该路径下所有文件名称带后缀

    text_image_name_list_file = open(
        IMAGE_NAME_LIST_PATH, 'w')  # 以写入的方式打开txt ，方便更新 不要用追加写

    for image_file_name in image_file_list:  # 例遍写入
        image_name, file_extend = os.path.splitext(image_file_name)  # 去掉扩展名
        text_image_name_list_file.write(image_name + '\n')  # 写入

    text_image_name_list_file.close()


def main(
        img_f=r'E:\projects\algo_detection\src\utils\data\convert\politics\images',
        label_f=r'E:\projects\algo_detection\src\utils\data\convert\politics\labels',
        output_f=r'politics',
        class_txt_path=r'E:\projects\algo_detection\src\utils\data\convert\politics\classes.txt',
        draw_bbox=True,
        valid_class_index=None,  # visualize all if label_index = None
):
    if isinstance(valid_class_index, int):
        valid_class_index = [valid_class_index]
    make_name_list(img_f, IMAGE_NAME_LIST_PATH)  # 执行写入txt函数
    # random name
    if class_txt_path is None or not os.path.exists(class_txt_path):
        classes = [str(i) for i in range(20)]
    else:

        classes = open(class_txt_path).read().strip().split('\n')
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(classes))]

    image_names = open(IMAGE_NAME_LIST_PATH).read(
    ).strip().split()

    box_total = 0
    image_total = 0
    for image_name in image_names:
        box_num = draw_box_on_image(
            image_name, classes, colors, label_f, img_f, output_f, valid_class_index=valid_class_index,
            draw_bbox=draw_bbox)  # 对图片画框
        box_total += box_num
        image_total += 1
        print('Box number:', box_total, 'Image number:', image_total)


if __name__ == '__main__':
    from fire import Fire

    Fire(main)
