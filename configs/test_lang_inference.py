_base_ = [
    "_base_/default_runtime.py",
]

# misc custom setting
gpu_nums = 1
batch_size = 1 * gpu_nums
batch_size_val = 1 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 4 * gpu_nums
mix_prob = 0.8
empty_cache = False
enable_amp = True

# 语言预训练模型设置 - 匹配预训练模型架构
model = dict(
    type="LangPretrainer",
    backbone=dict(
        type="PT-v3m1",
        in_channels=11,  # 匹配预训练模型：color 3, quaternion 4, scale 3, opacity 1 (无normal)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2),  # 匹配预训练模型
        enc_depths=(2, 2, 2, 6),  # 匹配预训练模型
        enc_channels=(32, 64, 128, 256),  # 匹配预训练模型
        enc_num_head=(2, 4, 8, 16),  # 匹配预训练模型
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),  # 匹配预训练模型
        dec_channels=(768, 512, 256),  # 匹配预训练模型
        dec_num_head=(16, 16, 16),  # 匹配预训练模型
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
        dict(type="L2Loss", reduction="mean", loss_weight=1.0),
    ],
)

# scheduler settings
eval_epoch = 100
epoch = 8 * eval_epoch
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings - 去除normal特征
dataset_type = "ScanNetGSDataset"
data_root = "/home/huajianzeng/project/SceneSplat/scannet_mcmc_3dgs_preprocessed"

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        sample_tail_classes=False,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "lang_feat",  # 语言特征
                ),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "lang_feat"),
                feat_keys=("color", "opacity", "quat", "scale"),  # 不包含normal
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "lang_feat",  # 语言特征
                ),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "lang_feat"),
                feat_keys=("color", "opacity", "quat", "scale"),  # 不包含normal
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test_grid1.0cm_chunk6x6_stride3x3",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "lang_feat",  # 语言特征
                ),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "opacity", "quat", "scale"),  # 不包含normal
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "opacity", "quat", "scale"),  # 不包含normal
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)