import os
import cv2
import json
import logging
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# voc dataset
voc_class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]  # 类别名称请务必与 YOLO 格式的标签对应

# coco dataset
coco_class_names = ["person", "bicycle", "car", "motorcycle", "airplane",
               "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
               "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
               "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
               "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
               "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
               "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
               "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
               "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
               "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
               "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush"]

def coco80_to_coco91_class():
    r"""
    Converts 80-index (val2014) to 91-index (paper).
    For details see https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/.

    Example:
        ```python
        import numpy as np

        a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")
        x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
        x2 = [
            list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)
        ]  # coco to darknet
        ```
    """
    return [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,
            31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,
            55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,
            82,84,85,86,87,88,89,90]

# 设置日志
LOGGER = logging.getLogger(__name__)

def process_img(image_filename, data_path, label_path):
    """处理单个图像并返回其信息，若标签文件不存在则只返回图像信息"""
    # 构建图像路径
    image_path = os.path.join(data_path, image_filename)

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        LOGGER.warning(f"无法读取图像: {image_path}")
        return image_filename, None
    height, width = img.shape[:2]

    # 初始化结果字典
    result = {"shape": (height, width), "labels": []}

    # 检查并读取标签文件
    label_file = os.path.join(label_path, os.path.splitext(image_filename)[0] + ".txt")
    if os.path.exists(label_file):
        with open(label_file, "r") as file:
            lines = file.readlines()

        # 处理标签
        for line in lines:
            category, x, y, w, h = map(float, line.strip().split())
            result["labels"].append((category, x, y, w, h))
    else:
        LOGGER.info(f"标签文件不存在，已跳过: {label_file}")

    return image_filename, result

def get_img_info(data_path, label_path):
    """
    并行获取图像信息，若标签文件不存在则跳过标签

    Args:
        data_path (str): 图像文件目录路径
        label_path (str): 标签文件目录路径

    Returns:
        dict: 包含图像信息的字典
    """
    LOGGER.info("Get img info")

    image_filenames = sorted(os.listdir(data_path))
    img_info = {}

    # 使用 ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        process_func = partial(process_img, data_path=data_path, label_path=label_path)
        results = list(tqdm(executor.map(process_func, image_filenames), total=len(image_filenames)))

    # 整理结果
    for filename, info in results:
        if info is not None:  # 只添加有效结果
            img_info[filename] = info

    return img_info

def generate_coco_format_labels(img_info,
        class_names, save_path, data_type, task_shape=(640, 640)):

    if data_type == "coco" :
        coco80 = coco80_to_coco91_class()

    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    LOGGER.info(f"Convert to COCO format")
    for i, (img_path, info) in enumerate(tqdm(img_info.items())):
        labels = info["labels"] if info["labels"] else []
        img_id = int(osp.splitext(osp.basename(img_path))[0].lstrip('0'))
        img_h, img_w = info["shape"]
        dataset["images"].append(
            {
                "file_name": os.path.basename(img_path),
                "id": img_id,
                "width": img_w,
                "height": img_h,
            }
        )
        if labels:
            for label in labels:
                c, x, y, w, h = label[:5]
                # convert x,y,w,h to x1,y1,x2,y2

                # center_x, center_y, w, h(normalized origin image size)
                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h

                # cls_id starts from 0
                if data_type == "coco" :
                    cls_id = coco80[int(c)]
                elif data_type == "voc" :
                    cls_id = int(c)

                new_w = max(0, x2 - x1)
                new_h = max(0, y2 - y1)

                dataset["annotations"].append(
                    {
                        "area": new_h * new_w,
                        "bbox": [x1, y1, new_w, new_h],
                        "category_id": cls_id,
                        "id": ann_id,
                        "image_id": img_id,
                        "iscrowd": 0,
                        # mask
                        "segmentation": [],
                    }
                )
                ann_id += 1

    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=4)
        LOGGER.info(
            f"Convert to COCO format finished. Resutls saved in {save_path}"
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, help="coco, voc")
    parser.add_argument('--data_path', type=str, help="image data path")
    parser.add_argument('--label_path',type=str, help="label data path")
    parser.add_argument('--save_path', type=str, help="save result path")
    args = parser.parse_args()

    img_info = get_img_info(args.data_path, args.label_path)

    if args.data_type == "coco":
        generate_coco_format_labels(img_info, coco_class_names, args.save_path, args.data_type)
    elif args.data_type == "voc":
        generate_coco_format_labels(img_info, voc_class_names, args.save_path, args.data_type)
