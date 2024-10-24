_base_ = [
    '../_base_/models/FAT_retinanet.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        embed_dims=[48, 96, 192, 384],
        depths = [2, 2, 6, 2],
        kernel_sizes = [3, 5, 7, 9],
        num_heads = [3, 6, 12, 24],
        window_sizes = [8, 4, 2, 1],
        mlp_kernel_sizes = [5, 5, 5, 5],
        mlp_ratios = [4, 4, 4, 4],
        drop_path_rate = 0.05,
        use_checkpoint = False,
        out_indices = (0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5)
)


optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001,
                 paramwise_cfg=dict({'norm': dict(decay_mult=0.)}))


lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)