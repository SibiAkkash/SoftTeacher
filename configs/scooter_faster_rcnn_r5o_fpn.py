_base_ = "./faster_rcnn_r50_fpn_1x_coco.py"

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        )
    )
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        ann_file="../yolov5/experiments/scooter-ds-cp/annotations/train.json",
        img_prefix="../yolov5/experiments/scooter-ds-cp/images/train/",
    
    ),
    val=dict(
        ann_file="../yolov5/experiments/scooter-ds-cp/annotations/val.json",
        img_prefix="../yolov5/experiments/scooter-ds-cp/images/val/",
    ),
    test=dict(
        ann_file="../yolov5/experiments/scooter-ds-cp/annotations/val.json",
        img_prefix="../yolov5/experiments/scooter-ds-cp/images/val/",
    )
)

load_from = "../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"