_base_ = "base.py"

classes = ("scooter",)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            ann_file="../yolov5/experiments/scooter-ds-cp/annotations/train.json",
            img_prefix="../yolov5/experiments/scooter-ds-cp/images/train/",
            classes=classes,
        ),
        unsup=dict(
            ann_file="../yolov5/experiments/scooter-ds-cp/annotations/unlabelled.json",
            img_prefix="../yolov5/experiments/scooter-ds-cp/unlabelled/",
            classes=classes,
        ),
    ),
    val=dict(
        ann_file="../yolov5/experiments/scooter-ds-cp/annotations/val.json",
        img_prefix="../yolov5/experiments/scooter-ds-cp/images/val/",
        classes=classes,
    ),
    test=dict(
        ann_file="../yolov5/experiments/scooter-ds-cp/annotations/val.json",
        img_prefix="../yolov5/experiments/scooter-ds-cp/images/val/",
        classes=classes,
    ),
    sampler=dict(train=dict(sample_ratio=[1, 4])),
)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        )
    )
)

# semi_wrapper = dict(
#     train_cfg=dict(
#         unsup_weight=2.0,
#     )
# )

# lr_config = dict(step=[120000 * 4, 160000 * 4])
# runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)


fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
