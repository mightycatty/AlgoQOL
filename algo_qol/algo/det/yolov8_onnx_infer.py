import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch  # required for onnxruntime CUDAExecutionProvider
import onnxruntime as ort


def resize_image(image: np.ndarray, size: tuple,
                 letterbox_image=True):
    """
    :param image:
    :param size:
    :param letterbox_image: pad the image(center) to square while keeping original ratio, resize scheme used by yolov5/yolov8
    :return:
    """
    ih, iw, _ = image.shape
    h, w = size
    resize_params = {}
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        resize_params['scale'] = scale
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
        resize_params['padding_w'] = (w - nw) // 2
        resize_params['padding_h'] = (h - nh) // 2
    else:
        image_back = image
    return image_back, resize_params


def nms(bounding_boxes, confidence_score, labels, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_label.append(labels[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_label


def xywh2xyxy(box):
    x0 = box[:, 0] - box[:, 2] // 2
    y0 = box[:, 1] - box[:, 3] // 2
    x1 = box[:, 0] + box[:, 2] // 2
    y1 = box[:, 1] + box[:, 3] // 2
    return np.stack([x0, y0, x1, y1], axis=-1)


def draw_results(img_rbg: np.ndarray, bboxes, scores, labels, show=True):
    img_rbg = img_rbg.copy()
    for bbox, label, score in zip(bboxes, labels, scores):
        cv2.rectangle(img_rbg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(img_rbg, label + ' ' + str(round(score, 3)), (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if show:
        cv2.namedWindow('image', 0)
        cv2.imshow('image', img_rbg[:, :, ::-1])
        cv2.waitKey(0)
    return img_rbg


class Yolov8DetOnnx:
    def __init__(self,
                 model_dir: str,
                 classes: list,
                 post_processing_fn=None,
                 preprocessing_on_gpu=True,
                 input_size=(640, 640),
                 iou=0.5,
                 conf=0.1,
                 intra_op_num_threads=1,
                 *args, **kwargs
                 ):
        self.iou = iou
        self.conf = conf
        self.classes = classes
        self.input_size = input_size
        self.preprocessing_on_gpu = preprocessing_on_gpu
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        self._ort_session = ort.InferenceSession(model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                                 sess_options=sess_opt)
        self._input_name = self._ort_session.get_inputs()[0].name
        self._output_name = self._ort_session.get_outputs()[0].name
        self.post_processing_fn = post_processing_fn

    def _input_preprocessing(self, image: np.ndarray, return_type='pt'):
        image, resize_params = resize_image(image, self.input_size, letterbox_image=True)
        if return_type == 'pt':
            # cost 5-6ms on cpu
            image = torch.from_numpy(image).cuda().to(torch.float16) / 255.
            image = torch.unsqueeze(image, dim=0)
            image = image.permute(0, 3, 1, 2).contiguous()
        else:
            # processing as torch gpu tensor, best runtime performance
            image = np.float32(image)
            image = np.expand_dims(image, axis=0) / 255.
            image = image.transpose(0, 3, 1, 2)
            # image = np.float16(image)
        return image, resize_params

    def _sess_run(self, image):
        if isinstance(image, np.ndarray):
            outputs = self._ort_session.run(None, {self._input_name: image})[0]
        else:
            if not hasattr(self, '_io_binding'):
                self._io_binding = self._ort_session.io_binding()
            self._io_binding.bind_input(
                name=self._input_name,
                device_type='cuda',
                device_id=0,
                element_type=np.float16,
                shape=tuple(image.shape),
                buffer_ptr=image.data_ptr(),
            )
            self._io_binding.bind_output(self._output_name)
            self._ort_session.run_with_iobinding(self._io_binding)
            outputs = self._io_binding.copy_outputs_to_cpu()[0]
        return outputs

    def _output_processing(self, outputs, resize_params):
        padding_h = resize_params['padding_h']
        padding_w = resize_params['padding_w']
        scale = resize_params['scale']
        outputs = np.squeeze(outputs)
        bboxes = outputs[:4, :]
        scores = outputs[4:, :]

        class_ids = np.argmax(scores, axis=0)
        scores = np.max(scores, axis=0)

        valid_index = scores > self.conf
        bboxes = bboxes[:, valid_index].T
        # scale back
        scale = 1. / scale
        bboxes[:, 0] = bboxes[:, 0] - padding_w
        bboxes[:, 1] = bboxes[:, 1] - padding_h
        bboxes = bboxes * scale
        # xywh2xyxy
        bboxes = np.int32(xywh2xyxy(bboxes))
        scores = np.float32(scores[valid_index])
        class_ids = class_ids[valid_index]
        return bboxes, scores, class_ids

    def __call__(self, image_rgb: np.ndarray, nms_processing=True, visualize=False, **kwargs):
        """
        Args:
            image_rgb: input image
            nms_processing: apply nms processing
            **kwargs:
        Returns:

        """
        image_input, resize_params = self._input_preprocessing(image_rgb,
                                                               return_type='pt' if self.preprocessing_on_gpu else 'numpy')
        outputs = self._sess_run(image_input)
        bboxes, scores, class_ids = self._output_processing(outputs, resize_params)
        labels = [self.classes[item] for item in class_ids]
        ret = (bboxes, scores, labels)
        if nms_processing:
            ret = nms(*ret, self.iou)

        if self.post_processing_fn is not None:
            ret = self.post_processing_fn(ret)

        labels = ret[2]
        scores = [round(s, 3) for s in ret[1]]
        bboxes = ret[0]
        if visualize:
            print(ret)
            image = draw_results(image_rgb, bboxes, scores, labels)
        result = dict(labels=labels, scores=scores, bboxes=bboxes)
        return result

    def predict(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)



