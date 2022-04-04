_base_ = "base.py"

classes = ("scooter",)
data_root = "../scooter-ds-cp"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            ann_file=f"{data_root}/annotations/train.json",
            img_prefix=f"{data_root}/labelled_images/train/",
            classes=classes,
        ),
        unsup=dict(
            ann_file=f"{data_root}/annotations/unlabelled.json",
            img_prefix=f"{data_root}/unlabelled_images/",
            classes=classes,
        ),
    ),
    val=dict(
        ann_file=f"{data_root}/annotations/val.json",
        img_prefix=f"{data_root}/labelled_images/val/",
        classes=classes,
    ),
    test=dict(
        ann_file=f"{data_root}/annotations/val.json",
        img_prefix=f"{data_root}/labelled_images/val/",
        classes=classes,
    ),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
            by_prob=False,          # errors out when this is True
            # at_least_one=True,
            epoch_length=7330,      # this should be dynamic, depends on size of dataset and batch size
        )
    ),
)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        )
    )
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

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

evaluation = dict(type="SubModulesDistEvalHook", interval=4000)
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000, 160000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=20)

fp16 = dict(loss_scale="dynamic")
load_from = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
