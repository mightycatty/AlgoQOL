"""convert coco json to yolo texts
reference: https://github.com/ultralytics/JSON2YOLO/tree/master
"""
import json
import os
import argparse
import random


class COCO2YOLO:
    def __init__(self, json_file, output, category_txt: str = 'class_id.txt', val_num=0, output_set='train'):
        self.category_txt = category_txt
        self.val_num = val_num
        self.output_set = output_set
        self.json_file = json_file
        self.output = output
        self._check_file_and_dir(json_file, output)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\') + 1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        with open(self.category_txt, 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")

        print(f'saving category list to {self.category_txt}')
        self.save_classes()

    def _save_txt(self, anno_dict):
        rename_cate = False
        if rename_cate:
            category_id_set = 0
            print(f'all label force to relabel as:{category_id_set}')
        # shuffle
        random.seed(0)
        l = list(anno_dict.items())
        random.shuffle(l)
        if self.output_set == 'train':
            l = l[self.val_num:]
        elif self.output_set == 'val':
            assert self.val_num > 0, 'val_num > 0 required for outputting val'
            l = l[:self.val_num]
        anno_dict = dict(l)

        for k, v in anno_dict.items():
            file_name = os.path.splitext(v[0][0])[0] + ".txt"
            with open(os.path.join(self.output, file_name), 'w', encoding='utf-8') as f:
                # print(k, v)
                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    if rename_cate: category_id = category_id_set
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')


def main(json_file: str,
         output: str,
         output_set='train',
         val_num=0):
    COCO2YOLO(json_file, output, category_txt='classes.txt', output_set=output_set, val_num=val_num).coco2yolo()


if __name__ == '__main__':
    from fire import Fire

    Fire(main)
