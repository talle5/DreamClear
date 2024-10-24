_base_ = [
    '../_base_/models/FAT_rcnn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        embed_dims=[64, 128, 256, 512],
        depths = [2, 2, 6, 2],
        kernel_sizes = [3, 5, 7, 9],
        num_heads = [2, 4, 8, 16],
        window_sizes = [8, 4, 2, 1],
        mlp_kernel_sizes = [5, 5, 5, 5],
        mlp_ratios = [4, 4, 4, 4],
        drop_path_rate = 0.05,
        use_checkpoint = False,
        out_indices = (0, 1, 2, 3)
    ),
    neck = dict(in_channels=[64, 128, 256, 512])
)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)})
                 )

lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

optimizer_config = dict(
    #type="DistOptimizerHook",
    #update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)