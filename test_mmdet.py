# from mmdet.apis import init_detector, inference_detector

# config_file = 'configs/faster_rcnn_r50_fpn_1x_coco.py'

# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# device = 'cuda:0'

# # init a detector
# model = init_detector(config_file, checkpoint_file, device=device)

# # inference the demo image
# img_path = '../yolov5/experiments/scooter-ds-cp/images/val/live_0002.jpg'
# inference_detector(model, img_path)


import os
import json
import cv2
import math
from typing import Tuple
from pprint import pprint
from tqdm.auto import tqdm
import os.path as osp


def xywhn2xyxy(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[int]:
    """Convert normalised labels (x, y, width, height) to format (xmin, ymin, xmax, ymax)"""
    x1 = round(img_w * (x - w / 2), 4)
    x2 = round(img_w * (x + w / 2), 4)
    y1 = round(img_h * (y - h / 2), 4)
    y2 = round(img_h * (y + h / 2), 4)
    return x1, y1, x2, y2


def xyxy2xywhn(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[int]:
    """Convert format (xmin, ymin, xmax, ymax) to normalised labels (x, y, width, height)"""
    x = round(((x1 + x2) / 2) / img_w, 6)
    y = round(((y1 + y2) / 2) / img_h, 6)
    w = round((x2 - x1) / img_w, 6)
    h = round((y2 - y1) / img_h, 6)
    return x, y, w, h


# COCO annotation format
# stored as a single JSON file
"""
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
"""


def convert_dataset_to_coco(dataset_path, split="train"):
    images_path = osp.join(dataset_path, "labelled_images", split)
    labels_path = osp.join(dataset_path, "labels", split)
    annotations_save_path = osp.join(dataset_path, "annotations")

    images = []
    annotations = []
    categories = []

    with open(osp.join(labels_path, "labels.txt")) as f:
        classes = list(map(lambda line: line.strip(), f.readlines()))

    for idx, class_name in enumerate(classes):
        categories.append(
            dict(
                id=idx,
                name=class_name,
            )
        )

    annot_id = 0

    for image_idx, file in tqdm(enumerate(os.scandir(images_path))):

        if file.name.endswith(("json", "txt")):
            continue

        H, W, _ = cv2.imread(file.path).shape
        images.append(dict(id=image_idx, file_name=file.name, height=H, width=W))

        name, ext = file.name.split(".")
        label_file = osp.join(labels_path, name + ".txt")

        with open(label_file) as f:
            annots = list(map(lambda line: line.strip().split(), f.readlines()))

        for category_id, cx, cy, w, h in annots:
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)

            xmin, ymin, xmax, ymax = xywhn2xyxy(cx, cy, w, h, img_w=W, img_h=H)

            annot = dict(
                id=annot_id,
                image_id=image_idx,
                category_id=int(category_id),
                bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                area=(xmax - xmin) * (ymax - ymin),
                # segmentation=[],
                iscrowd=0,
            )

            annotations.append(annot)

            annot_id += 1

    labelled_combined = dict(
        images=images, annotations=annotations, categories=categories
    )

    with open(osp.join(annotations_save_path, split + ".json"), "w") as f:
        json.dump(labelled_combined, f)

    # unlabelled images
    unlabelled_images_path = osp.join(dataset_path, "unlabelled_images")
    unlabelled_images = []
    for un_image_idx, un_file in tqdm(enumerate(os.scandir(unlabelled_images_path))):
        unlabelled_images.append(
            dict(id=un_image_idx, file_name=un_file.name, height=640, width=360)
        )

    unlabelled_combined = dict(
        images=unlabelled_images,
        annotations=[],
        categories=labelled_combined["categories"],
    )
    
    with open(osp.join(annotations_save_path, "unlabelled.json"), "w") as f:
        json.dump(unlabelled_combined, f)


if __name__ == "__main__":
    ds_path = "../yolov5/experiments/scooter-ds-cp"
    convert_dataset_to_coco(ds_path, split="train")
    convert_dataset_to_coco(ds_path, split="val")
