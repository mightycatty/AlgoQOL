# Algo QOL API Reference

Automated generated documentation.

### Module: `algo/cls/metrics.py`

#### Function `cls_report`
```python
def cls_report(gt, pred)
```
*No docstring*

---

### Module: `algo/det/utils.py`

@Time ： 2025/4/21 16:44
@Auth ： heshuai.sec@gmail.com
@File ：utils.py

#### Function `cxcywh_to_xyxy`
```python
def cxcywh_to_xyxy(cxcywh, w, h)
```
Convert cxcywh to xyxy.
Args:
    cxcywh: normalized cxcywh
Returns:

---

### Module: `algo/det/yolov8_onnx_infer.py`

#### Function `resize_image`
```python
def resize_image(image, size, letterbox_image)
```
:param image:
:param size:
:param letterbox_image: pad the image(center) to square while keeping original ratio, resize scheme used by yolov5/yolov8
:return:

#### Function `nms`
```python
def nms(bounding_boxes, confidence_score, labels, threshold)
```
*No docstring*

#### Function `xywh2xyxy`
```python
def xywh2xyxy(box)
```
*No docstring*

#### Function `draw_results`
```python
def draw_results(img_rbg, bboxes, scores, labels, show)
```
*No docstring*

#### Class `Yolov8DetOnnx`
```python
class Yolov8DetOnnx
```
*No docstring*

- **Method** `__init__(model_dir, classes, post_processing_fn, preprocessing_on_gpu, input_size, iou, conf, intra_op_num_threads)`
  > *No docstring*

- **Method** `predict()`
  > *No docstring*

---

### Module: `algo/det/yolov8_torch_infer.py`

@Time ： 2023/12/14 16:39
@Auth ： heshuai.sec@gmail.com
@File ：yolov8_torch_infer.py

#### Function `nms`
```python
def nms(bounding_boxes, confidence_score, labels, threshold)
```
*No docstring*

#### Class `Yolov8DetTorch`
```python
class Yolov8DetTorch
```
*No docstring*

- **Method** `__init__(model_dir, device, post_processing_fn, iou, conf)`
  > *No docstring*

---

### Module: `algo/det/coco/browse_coco_json.py`

#### Function `show_coco_json`
```python
def show_coco_json(args)
```
*No docstring*

#### Function `show_bbox_only`
```python
def show_bbox_only(coco, anns, show_label_bbox, is_filling)
```
Show bounding box of annotations Only.

#### Function `crop_samples`
```python
def crop_samples(ann_file, category_name, num, save_f)
```
*No docstring*

#### Function `parse_args`
```python
def parse_args()
```
*No docstring*

#### Function `main`
```python
def main()
```
*No docstring*

---

### Module: `algo/det/coco/coco_utils.py`

@Time ： 2023/6/18 11:43
@Auth ： heshuai.sec@gmail.com
@File ：coco_utils.py

#### Function `coco_split`
```python
def coco_split(coco_file, val_ratio, seed, print_stats)
```
*No docstring*

#### Function `coco_analyse`
```python
def coco_analyse(annotations_file, verbose)
```
Analyzes a COCO detection format dataset and returns a dictionary containing various statistics.

Args:
  annotations_file: Path to the COCO annotations file.

Returns:
  A dictionary containing statistics about the dataset.

---

### Module: `algo/det/convertor/coco2yolo.py`

convert coco json to yolo texts
reference: https://github.com/ultralytics/JSON2YOLO/tree/master

#### Class `COCO2YOLO`
```python
class COCO2YOLO
```
*No docstring*

- **Method** `__init__(json_file, output, category_txt, val_num, output_set)`
  > *No docstring*

- **Method** `save_classes()`
  > *No docstring*

- **Method** `coco2yolo()`
  > *No docstring*

#### Function `main`
```python
def main(json_file, output, output_set, val_num)
```
*No docstring*

---

### Module: `algo/det/convertor/labelme2yolo.py`

#### Class `Labelme2YOLO`
```python
class Labelme2YOLO
```
*No docstring*

- **Method** `__init__(json_dir, to_seg)`
  > *No docstring*

- **Method** `convert(val_size)`
  > *No docstring*

- **Method** `convert_one(json_name)`
  > *No docstring*

---

### Module: `algo/det/convertor/ocr2yolo.py`

@Time ： 2023/12/29 17:26
@Auth ： heshuai.sec@gmail.com
@File ：netease2yolo.py

#### Function `bbox_from_points`
```python
def bbox_from_points(points)
```
*No docstring*

---

### Module: `algo/det/convertor/yolo2coco.py`

@Time ： 2023/5/11 11:59
@Auth ： heshuai.sec@gmail.com
@File ：yolo2coco.py

convert coco detection dataset to yolo detection dataset
- coco bbox: xywh / yolo bbox: cxcywh(0-1)

#### Function `get_file_recursively`
```python
def get_file_recursively(folder_dir)
```
iteratively get file list under folder_dir
:param folder_dir: folder
:return: a list of files

#### Function `yolo2coco`
```python
def yolo2coco(images_dir, labels_dir, output_json, classes_or_txt, absolute_path)
```
*No docstring*

#### Function `get_image_size`
```python
def get_image_size(image_path)
```
Gets the width and height of an image.

#### Function `yolov8_to_coco`
```python
def yolov8_to_coco(yolov8_data_yaml, dataset_prefix, absolute_path)
```
parse ultrality yolov8 data.yaml to prepare a coco dataset

---

### Module: `algo/det/convertor/yolov8tointernvl.py`

@Time ： 2024/9/3 10:09
@Auth ： heshuai.sec@gmail.com
@File ：yolov8tointernvl.py

#### Function `normalize_coordinates`
```python
def normalize_coordinates(box, image_width, image_height)
```
*No docstring*

#### Function `get_image_size`
```python
def get_image_size(image_path)
```
Gets the width and height of an image.

#### Function `read_yolo_txt`
```python
def read_yolo_txt(txt_path, return_bbox_format)
```
*No docstring*

#### Function `get_det_conversation`
```python
def get_det_conversation(id, image_path, width, height, bboxes, labels, prompt)
```
*No docstring*

#### Function `main`
```python
def main(yolov8_yml, output_file, phrase)
```
convert yolov8 for internvl finetune
Args:
    yolov8_yml: ultrality yolov8 data.yaml
    output_file:

Returns:

---

### Module: `algo/det/yolo/browse_yolov8_txt.py`

#### Function `get_image_path`
```python
def get_image_path(img_f, image_name)
```
*No docstring*

#### Function `plot_one_box`
```python
def plot_one_box(x, image, color, label, line_thickness, score)
```
*No docstring*

#### Function `draw_box_on_image`
```python
def draw_box_on_image(img_path, label_path, classes, colors, output_f, valid_class_index, draw_bbox)
```
This function will add rectangle boxes on the images.

#### Function `make_name_list`
```python
def make_name_list(RAW_IMAGE_FOLDER, IMAGE_NAME_LIST_PATH)
```
This function will collect the image names without extension and save them in the name_list.txt.

#### Function `yolo_dataset_visualization`
```python
def yolo_dataset_visualization(img_f, label_f, output_f, class_txt_path, draw_bbox, valid_class_index)
```
*No docstring*

---

### Module: `algo/det/yolo/yolo_utils.py`

#### Function `read_ultralytics_yolo_dataset`
```python
def read_ultralytics_yolo_dataset(img_root, txt_root, classes_txt)
```
*No docstring*

#### Function `yolo_analyse`
```python
def yolo_analyse(txt_root, classes_txt, save, verbose)
```
*No docstring*

#### Function `remove_yolo_none_pair`
```python
def remove_yolo_none_pair(img_f, label_f)
```
*No docstring*

#### Function `split_val`
```python
def split_val(img_root, txt_root, val_num)
```
*No docstring*

---

### Module: `algo/llm/vllm_utils.py`

@Time ： 2025/4/17 15:59
@Auth ： heshuai.sec@gmail.com
@File ：vllm_client.py

ref: https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py

#### Function `extract_value_from_text`
```python
def extract_value_from_text(text, key)
```
使用正则表达式从文本中提取。

Args:
    text (str): 包含JSON数据的字符串
    key (str): 要提取的键名

Returns:
    Optional[Any]: 提取到的值，如果未找到则返回None

#### Function `parse_llm_json`
```python
def parse_llm_json(text)
```
Parse JSON output from LLM with mild repair on common formatting issues.

Args:
    text (str): LLM输出的文本，包含JSON数据

Returns:
    Optional[dict]: 解析后的JSON对象，失败时返回None

#### Function `parse_cot_response`
```python
def parse_cot_response(response, parse_answer)
```
Parses a CoT (Chain of Thought) response string, tries to fix malformed tags or JSON.

Args:
    response (str): LLM-generated CoT response string.
    parse_answer (bool): 是否尝试将answer部分解析为JSON，默认True

Returns:
    dict: {
        'think': str or None,  # 思考过程
        'answer': dict/str or None  # 答案内容，解析成功时为dict，否则为原始字符串
    }

#### Function `extract_json_blocks`
```python
def extract_json_blocks(text)
```
从文本中提取所有JSON块。

Args:
    text (str): 包含JSON数据的文本

Returns:
    List[dict]: 提取到的JSON对象列表

#### Function `extract_code_blocks`
```python
def extract_code_blocks(text, language)
```
从文本中提取代码块。

Args:
    text (str): 包含代码块的文本
    language (str, optional): 指定语言类型，如'python', 'json'等

Returns:
    List[str]: 提取到的代码块列表

#### Function `clean_llm_output`
```python
def clean_llm_output(text)
```
清理LLM输出中的常见问题。

Args:
    text (str): 原始LLM输出

Returns:
    str: 清理后的文本

#### Function `extract_structured_response`
```python
def extract_structured_response(text, sections, default_sections)
```
从结构化LLM响应中提取各个部分。

Args:
    text (str): LLM响应文本
    sections (List[str], optional): 要提取的部分名称列表
    default_sections (List[str], optional): 默认部分列表

Returns:
    dict: 提取的各部分内容

#### Function `get_bool_value_score`
```python
def get_bool_value_score(response, label_key)
```
return None, None if fail

#### Function `get_int_value_score`
```python
def get_int_value_score(response, label_key, score_method, max_range)
```
score_method: weighted_avg, normalized_avg, top1

#### Function `get_list_value_score`
```python
def get_list_value_score(response, label_key)
```
*No docstring*

#### Function `encode_base64_content_from_url`
```python
def encode_base64_content_from_url(content_url)
```
Encode a content retrieved from a remote url to base64 format.

#### Function `encode_image`
```python
def encode_image(image)
```
encoder local image to base64 string, either from local file path or numpy array or PIL Image.
Args:
    image:

Returns:

#### Function `make_message`
```python
def make_message(prompt, images, videos, system_prompt, add_think_tag)
```
make a typical sandwich(system_prompt-mm_data(images/videos)-txt) messages for MLLM completion.
Args:
    prompt:
    images:
    system_prompt:
Returns:

#### Function `get_content`
```python
def get_content(response)
```
*No docstring*

#### Class `VLLMClient`
```python
class VLLMClient
```
*No docstring*

- **Method** `__init__(api_base, api_key, model, system_prompt)`
  > *No docstring*

- **Method** `set_model(model)`
  > *No docstring*

---

### Module: `algo/ocr/data/netease2paddlerec.py`

@Time ： 2023/12/29 17:26
@Auth ： heshuai.sec@gmail.com
@File ：

crop

#### Function `bbox_from_points`
```python
def bbox_from_points(points)
```
*No docstring*

#### Function `main`
```python
def main(result_json, img_f)
```
*No docstring*

---

### Module: `algo/ocr/data/paddle_data_utils.py`

@Time ： 2024/3/7 19:59
@Auth ： heshuai.sec@gmail.com
@File ：paddle_data_utils.py

#### Function `split_train_val`
```python
def split_train_val(txt_dir, ratio)
```
*No docstring*

---

### Module: `algo/ocr/data/paddle_ocr_test.py`

@Time ： 2023/12/28 10:20
@Auth ： heshuai.sec@gmail.com
@File ：paddle_ocr_test.py

pip install "paddleocr>=2.0.1" --upgrade PyMuPDF==1.21.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

#### Function `draw_boxed_with_four_points`
```python
def draw_boxed_with_four_points(image, points, color, thickness)
```
Draws a box with given 4 points.

Args:
  image: a numpy array representing an image.
  points: a list of 4 points, each point is a tuple (x, y) representing the
    coordinates of the point.
  color: a tuple of 3 integers (r, g, b) representing the color of the box.
  thickness: an integer representing the thickness of the box in pixels.

Returns:
  a numpy array representing the image with the box drawn on it.

---

### Module: `data_utils/file_utils.py`

#### Function `get_all_images`
```python
def get_all_images(f)
```
*No docstring*

#### Function `split_audio`
```python
def split_audio(wav_path, output_dir, max_duration)
```
*No docstring*

#### Function `load_video`
```python
def load_video(video_path, num_segments, save_tmp)
```
*No docstring*

#### Function `video_reader`
```python
def video_reader(video_dir, loop)
```
video reader generator, each run yields a img of RGB format
acapture is recommended over cv2, which yields a better streaming performance. however, if you don't bother to install it,
 just stick with OpenCV.

#### Class `VideoWriter`
```python
class VideoWriter
```
usage:
with VideoWriter('test.mp4') as w:
    img = None # BRG
    w.write_frame(img, verbose=False)

- **Method** `__init__(save_name, fps)`
  > *No docstring*

- **Method** `reset()`
  > *No docstring*

- **Method** `write_frame(image, verbose)`
  > *No docstring*

- **Method** `release()`
  > *No docstring*

#### Function `get_latest_file`
```python
def get_latest_file(folder)
```
get latest created file, helpful when finding your latest model checkpoint


#### Function `get_file_recursively`
```python
def get_file_recursively(folder_dir)
```
iteratively get file list under folder_dir
:param folder_dir: folder
:return: a list of files

#### Function `read_textlines`
```python
def read_textlines(txt_file)
```
*No docstring*

#### Function `convert_image_format_worker`
```python
def convert_image_format_worker(args)
```
*No docstring*

#### Function `convert_image_format`
```python
def convert_image_format(src_f, dst_format, threads_num)
```
convert all images in a folder to dst format, support sub-folders

#### Function `rm_corrupted_image`
```python
def rm_corrupted_image(src_f, move_out, threads_num)
```
remove corrupted images

#### Function `rename_files`
```python
def rename_files(src_f)
```
rename all images in a folder to uuid new name

#### Function `resize_image`
```python
def resize_image(img_src, img_dst, max_size, max_workers)
```
resize image

Args:
    img_src:
    img_dst:
    max_size: default 720
    max_workers:

Returns:

#### Function `download_video`
```python
def download_video(url, save_path, timeout)
```
*No docstring*

#### Function `merge_files`
```python
def merge_files(src_f, dst_f, max_workers, copy_only, rename, append_sub_folder_name)
```
merge all files in a folder(include sub-folders) to a new folder

#### Function `sample_files`
```python
def sample_files(src_f, dst_f, sample_num, threads_num)
```
*No docstring*

---

### Module: `data_utils/image_dedup.py`

Image Deduplication (Unified): CLIP/SigLIP or pHash via one interface.

Examples:
  # CLIP with default similarity=0.95
  python image_dedup.py dedup_images --root /path/to/images --method clip

  # CLIP with similarity threshold
  python image_dedup.py dedup_images --root /path/to/images --method clip --similarity 0.93

  # pHash with integer Hamming distance
  python image_dedup.py dedup_images --root /path/to/images --method phash --distance 6

  # pHash but user thinks in similarity: map to distance internally
  python image_dedup.py dedup_images --root /path/to/images --method phash --similarity 0.9

Notes on parameter compatibility:
- CLIP uses cosine similarity threshold in [0,1].
- pHash uses Hamming distance threshold on a 64-bit hash.
  If only `similarity` is provided, we convert: distance ≈ round((1 - similarity) * 64).

#### Class `CLIP`
```python
class CLIP
```
*No docstring*

- **Method** `__init__(model_name)`
  > *No docstring*

- **Method** `processor(images)`
  > *No docstring*

- **Method** `forward()`
  > *No docstring*

#### Function `init_model_preprocessor`
```python
def init_model_preprocessor(model_name, device)
```
*No docstring*

#### Function `extract_features`
```python
def extract_features(data, model, device, norm)
```
*No docstring*

#### Function `list_images`
```python
def list_images(directory, extensions)
```
*No docstring*

#### Class `ImgDS`
```python
class ImgDS
```
*No docstring*

- **Method** `__init__(paths, tfm)`
  > *No docstring*

#### Function `custom_collate_fn`
```python
def custom_collate_fn(batch)
```
*No docstring*

#### Function `build_faiss_index`
```python
def build_faiss_index(mat, gpu)
```
*No docstring*

#### Function `group_duplicates_cosine`
```python
def group_duplicates_cosine(index, feats, files, thr)
```
*No docstring*

#### Function `draw_gallery`
```python
def draw_gallery(index, feats, files, root, num_samples, group_size)
```
*No docstring*

#### Function `dedup_images`
```python
def dedup_images(root, method, similarity, distance, model, batch_size, cpu_index, skip_gallery_vis, move_out, debug, recursive)
```
Run image dedup using a single unified interface.

Args:
    root: directory of images
    method: 'clip' or 'phash'
    similarity: float in [0,1]; primary for CLIP; for pHash it will be mapped to distance if distance is None
    distance: int; primary for pHash
    model: CLIP/SigLIP HF id if method='clip'
    batch_size: CLIP extraction batch size (per GPU *adjusted*)
    cpu_index: use CPU FAISS even if GPUs exist
    skip_gallery_vis: skip gallery png export (CLIP only, non-debug)
    move_out: if set, move duplicates there instead of deleting
    debug: limit samples & avoid deletion; creates sample galleries (CLIP) or sample stacks (pHash)
    recursive: pHash per-subdir recursion behavior

Returns:
    dict with {"total": int, "duplicates": int, "groups": List[List[str]]}

---

### Module: `data_utils/image_dedup_legacy.py`

#### Function `get_transform`
```python
def get_transform(model_name)
```
*No docstring*

#### Function `init_handle`
```python
def init_handle(method)
```
*No docstring*

#### Class `HFModel`
```python
class HFModel
```
*No docstring*

- **Method** `__init__(model)`
  > *No docstring*

- **Method** `forward(x)`
  > *No docstring*

#### Function `dedup_image_entry`
```python
def dedup_image_entry(src_f, distance, debug, move_out_f)
```
*No docstring*

#### Function `dedup_image`
```python
def dedup_image(src_f, distance, move_out_f, debug, recursive, method)
```
*No docstring*

---

### Module: `data_utils/video_img_downloader.py`

#### Function `get_md5_encoder_name`
```python
def get_md5_encoder_name(img_url)
```
rename image with md5 of original name

#### Function `guess_image_format_from_url`
```python
def guess_image_format_from_url(url)
```
*No docstring*

#### Function `download_img_to_folder`
```python
def download_img_to_folder(img_url, save_f, reformat, md5_rename, username, password)
```
download img to folder, skip if file exists


#### Function `download_img`
```python
def download_img(img_url, save_path)
```
*No docstring*

#### Class `Downloader`
```python
class Downloader
```
*No docstring*

- **Method** `__init__(txt_src, save_f, col_index, remove_http_head)`
  > *No docstring*

- **Method** `download(img_url, imgpath)`
  > *No docstring*

- **Method** `get_img_url_generate(txt_src, save_f, col_index)`
  > *No docstring*

- **Method** `loop(imgs)`
  > *No docstring*

#### Function `download`
```python
def download(txt_or_csv_or_list, save_f, thread_num, col_index_or_name, remove_http_head)
```
download images or videos from give txt or csv

---

### Module: `data_utils/video_utils.py`

#### Function `convert_image_format_worker`
```python
def convert_image_format_worker(args)
```
*No docstring*

#### Function `convert_image_format`
```python
def convert_image_format(src_f, dst_format, threads_num)
```
convert all images in a folder to dst format, support sub-folders

#### Function `rm_corrupted_image_worker`
```python
def rm_corrupted_image_worker(src, move_out_folder)
```
*No docstring*

#### Function `rm_corrupted_image`
```python
def rm_corrupted_image(src_f, threads_num, move_out)
```
remove corrupted images

#### Function `get_all_images`
```python
def get_all_images(f)
```
*No docstring*

#### Function `rename_files`
```python
def rename_files(src_f)
```
rename all images in a folder to uuid new name

#### Function `resize_img_entry`
```python
def resize_img_entry(args)
```
*No docstring*

#### Function `resize_image`
```python
def resize_image(img_src, img_dst, max_size, max_workers)
```
*No docstring*

#### Function `download_video`
```python
def download_video(url, save_path, timeout)
```
*No docstring*

#### Function `split_audio`
```python
def split_audio(wav_path, output_dir, max_duration)
```
*No docstring*

#### Function `load_video`
```python
def load_video(video_path, max_frames)
```
*No docstring*

---

### Module: `utils/asyn_utils.py`

@Time ： 2024/09/23 14:13
@Auth ： heshuai.sec@gmail.com

#### Class `AsyncExecutor`
```python
class AsyncExecutor
```
*No docstring*

- **Method** `__init__(qps, max_workers)`
  > *No docstring*

- **Method** `submit(func)`
  > *No docstring*

- **Method** `get_result(task_id)`
  > *No docstring*

- **Method** `get_all_results()`
  > *No docstring*

- **Method** `close()`
  > *No docstring*

---

### Module: `utils/decorators.py`

#### Function `line_profile`
```python
def line_profile(func)
```
function decoderator for line-wise profile
usage:
    from evaluation_utils import line_profile
    @line_profile
    some_fn()
reference:
    https://github.com/rkern/line_profiler#kernprof
:param func:
:return:

#### Function `timethis`
```python
def timethis(func)
```
Decorator that reports the execution time.

#### Function `exception_handler`
```python
def exception_handler(func, msg)
```
*No docstring*

---

### Module: `utils/discord_bot.py`

#### Function `discord_send`
```python
def discord_send(message, file_path)
```
*No docstring*

#### Class `DiscordLogger`
```python
class DiscordLogger
```
*No docstring*

- **Method** `info(message, image_path)`
  > *No docstring*

- **Method** `warm()`
  > *No docstring*

---

### Module: `utils/gpu_utils.py`

#### Class `RunAsCUDASubprocess`
```python
class RunAsCUDASubprocess
```
transparent gpu management for tensorflow and other gpu-required applications.
1. make desired number of gpus visible to tensorflow
2. completely release gpu resource when tf session closed
3. select desired number of gpus with at least fraction of memory

Credit to ed-alertedh:
    https://gist.github.com/ed-alertedh/85dc3a70d3972e742ca0c4296de7bf00

- **Method** `__init__(num_gpus, memory_fraction, verbose)`
  > *No docstring*

#### Function `run_command`
```python
def run_command(cmd, verbose)
```
run commands , Linux only
:param cmd:
:return: string output if success otherwise False

#### Function `list_available_gpus`
```python
def list_available_gpus(verbose)
```
list available gpus and return id list by running cmd "nvidia-smi" in Linux
:param verbose:
:return:

#### Function `gpu_memory_map`
```python
def gpu_memory_map(verbose)
```
*No docstring*

#### Function `pick_n_gpu_lowest_memory`
```python
def pick_n_gpu_lowest_memory(n)
```
Returns GPU with the least allocated memory

#### Function `set_gpus_visiable`
```python
def set_gpus_visiable(gpu_num, verbose)
```
*No docstring*

#### Function `say_hi_to_your_program`
```python
def say_hi_to_your_program()
```
python program initialization, run before the main function in your python program
1. clean up console, filter warnings
:return:

#### Function `disable_gpu`
```python
def disable_gpu()
```
*No docstring*

#### Function `select_n_gpus`
```python
def select_n_gpus(num, max_load, max_mem)
```
*No docstring*

#### Class `EasyDict`
```python
class EasyDict
```
*No docstring*

#### Function `flat_list`
```python
def flat_list(list_input)
```
Flatten nested list

#### Function `video_reader`
```python
def video_reader(video_dir, loop)
```
video reader generator, each run yields a img of RGB format
acapture is recommended over cv2, which yields a better streaming performance. however, if you don't bother to install it,
 just stick with OpenCV.

#### Class `VideoWriter`
```python
class VideoWriter
```
usage:
with VideoWriter('test.mp4') as w:
    img = None # BRG
    w.write_frame(img, verbose=False)

- **Method** `__init__(save_name, fps)`
  > *No docstring*

- **Method** `reset()`
  > *No docstring*

- **Method** `write_frame(image, verbose)`
  > *No docstring*

- **Method** `release()`
  > *No docstring*

#### Class `ImgSaver`
```python
class ImgSaver
```
*No docstring*

#### Function `get_latest_file`
```python
def get_latest_file(folder)
```
get latest created file, helpful when finding your latest model checkpoint


#### Function `get_file_recursively`
```python
def get_file_recursively(folder_dir)
```
iteratively get file list under folder_dir
:param folder_dir: folder
:return: a list of files

#### Function `read_textlines`
```python
def read_textlines(txt_file)
```
*No docstring*

---

### Module: `utils/logger.py`

#### Class `MyLog`
```python
class MyLog
```
requirement: telebot
    https://github.com/eternnoir/pyTelegramBotAPI

- **Method** `__init__(log_name, log_dir, logging_level, clean_format, clear_file)`
  > *No docstring*

- **Method** `telegram_bot_init(token)`
  > *No docstring*

- **Method** `debug(message)`
  > *No docstring*

- **Method** `info(message)`
  > *No docstring*

- **Method** `warn(message)`
  > *No docstring*

- **Method** `error(message)`
  > *No docstring*

- **Method** `clear_logfile()`
  > *No docstring*

- **Method** `fire_message_via_bot(message, chat_id)`
  > 向指定的chat id发送文字信息

- **Method** `send_image_via_bot(image_path, chat_id)`
  > *No docstring*

---

### Module: `utils/torch_utils.py`

@Time ： 2024/10/30 15:21
@Auth ： heshuai.sec@gmail.com
@File ：torch_utils.py

#### Function `find_free_network_port`
```python
def find_free_network_port()
```
Finds a free port on localhost.

It is useful in single-node training when we don't want to connect to a real main node but have to set the
`MASTER_PORT` environment variable.

---

### Module: `utils/utils.py`

@Time ： 2024/10/10 10:45
@Auth ： heshuai.sec@gmail.com
@File ：utils.py

#### Class `Timer`
```python
class Timer
```
*No docstring*

- **Method** `__init__()`
  > *No docstring*

- **Method** `start(name)`
  > *No docstring*

- **Method** `end(name, verbose)`
  > *No docstring*

- **Method** `duration(name)`
  > *No docstring*

- **Method** `reset()`
  > *No docstring*

---
