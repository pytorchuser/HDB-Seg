_base_ = [
    '../_base_/models/upernet_custom_swin.py', '../_base_/datasets/oct_hcms2018.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_epoch.py'
]
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
NUM_CLASSES = 9

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        attn_drop_rate=0.2,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=NUM_CLASSES, dropout_ratio=0.2,
                     # TODO 此处添加配置信息msc_module_cfg
                     # msc_module_cfg=[
                     #     dict(type='PPM', layer_idx=0), dict(type='PPM', layer_idx=1),
                     #     dict(type='PPM', layer_idx=2), dict(type='PPM', layer_idx=3)]
                     msc_module_cfg=[
                        dict(type='UFE', layer_idx=2, ufe_cfg=dict(
                            num_stages=3,
                            strides=(1, 1, 1),
                            enc_num_convs=(1, 1, 1),
                            dec_num_convs=(1, 1),
                            downsamples=(True, True),
                            enc_dilations=(1, 1, 1),
                            dec_dilations=(1, 1),)),
                        dict(type='PPM', layer_idx=3)
                     ]
                     # msc_module_cfg=[
                     #     dict(type='PPM', layer_idx=0), dict(type='PPM', layer_idx=1)]
                     # msc_module_cfg=[
                     #     dict(type='PPM', layer_idx=1), dict(type='PPM', layer_idx=2),
                     #     dict(type='PPM', layer_idx=3)]
                     ),
    auxiliary_head=dict(in_channels=384, num_classes=NUM_CLASSES))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
     _delete_=True,
     type='AdamW',
     lr=0.00006,
     betas=(0.9, 0.999),
     weight_decay=0.01,
     )
optim_wrapper = dict(
    # 优化器包装器(Optimizer wrapper)为更新参数提供了一个公共接口
    type='AmpOptimWrapper',
    # 用于更新模型参数的优化器(Optimizer)
    optimizer=optimizer,
    # 如果 'clip_grad' 不是None，它将是 ' torch.nn.utils.clip_grad' 的参数。
    clip_grad=None,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=50),
    dict(
        type='PolyLR',
        power=1.0,
        begin=50,
        end=100,
        eta_min=0.0,
        by_epoch=True,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
# CUDA out of memory。 加载验证数据集时内存会爆掉， workers_per_gpu设置的小一些可避免这个问题
train_dataloader = dict(
    batch_size=8
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(
        type='DefaultSampler',
        # 训练时进行随机洗牌(shuffle)
        shuffle=True)
)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(
        type='DefaultSampler',
        # 训练时进行随机洗牌(shuffle)
        shuffle=True)
)
