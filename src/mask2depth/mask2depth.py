# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Depth Pro: Sharp Monocular Metric Depth in Less Than a Second


from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

from network.decoder import MultiresConvDecoder
from network.encoder import DepthProEncoder
from network.vit_factory import VIT_CONFIG_DICT, ViTPreset, create_vit
from network.ins_seg_head import SegmentationHead


from pathlib import Path
import yaml

def load_config(yaml_path: str) -> dict:
    """加载YAML配置文件并返回字典"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 假设配置文件路径为当前目录下的config.yaml
config_path = Path(__file__).parent / "./cfg/config.yaml"
project_config = load_config(config_path)["project"]



@dataclass
class DepthProConfig:
    """Configuration for DepthPro."""

    patch_encoder_preset: ViTPreset
    image_encoder_preset: ViTPreset
    decoder_features: int

    checkpoint_uri: Optional[str] = None
    fov_encoder_preset: Optional[ViTPreset] = None
    use_fov_head: bool = True


DEFAULT_MONODEPTH_CONFIG_DICT = DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="./checkpoints/mask2depth.pt",
    decoder_features=256,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)


def create_backbone_model(
        preset: ViTPreset
) -> Tuple[nn.Module, ViTPreset]:
    """Create and load a backbone model given a config.

    Args:
    ----
        preset: A backbone preset to load pre-defind configs.

    Returns:
    -------
        A Torch module and the associated config.

    """
    if preset in VIT_CONFIG_DICT:
        config = VIT_CONFIG_DICT[preset]
        model = create_vit(preset=preset, use_pretrained=False)
    else:
        raise KeyError(f"Preset {preset} not found.")

    return model, config


def create_model_and_transforms(
        config: DepthProConfig = DEFAULT_MONODEPTH_CONFIG_DICT,
        device: torch.device = torch.device("cuda"),
        precision: torch.dtype = torch.float32,
) -> Tuple[DepthPro, Compose]:
    """Create a DepthPro model and load weights from `config.checkpoint_uri`."""
    patch_encoder, patch_encoder_config = create_backbone_model(
        preset=config.patch_encoder_preset
    )
    image_encoder, _ = create_backbone_model(
        preset=config.image_encoder_preset
    )

    fov_encoder = None
    if config.use_fov_head and config.fov_encoder_preset is not None:
        fov_encoder, _ = create_backbone_model(preset=config.fov_encoder_preset)

    dims_encoder = patch_encoder_config.encoder_feature_dims
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=config.decoder_features,
    )
    decoder = MultiresConvDecoder(
        dims_encoder=[config.decoder_features] + list(encoder.dims_encoder),
        dim_decoder=config.decoder_features,
    )

    # 创建实例分割头
    num_classes = project_config["num_class"] + 1  # 获取YAML中的11 +1 =12
    instance_seg_head = SegmentationHead(
    in_channels=256,
    num_classes=num_classes
    )

    model = DepthPro(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
        use_fov_head=config.use_fov_head,
        fov_encoder=fov_encoder,
        instance_seg_head=instance_seg_head,  # 传递实例分割头
    ).to(device)

    if precision == torch.half:
        model.half()

    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(precision),
        ]
    )

    if config.checkpoint_uri is not None:
        state_dict = torch.load(config.checkpoint_uri, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False
        )

        if len(unexpected_keys) != 0:
            print(f"Unexpected keys ignored: {unexpected_keys}")
        if len(missing_keys) != 0:
            print(f"Missing keys ignored: {missing_keys}")

    return model, transform


class DepthPro(nn.Module):
    """DepthPro 深度网络."""

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            last_dims: Tuple[int, int],
            use_fov_head: bool = False,
            fov_encoder: Optional[nn.Module] = None,
            instance_seg_head: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.use_fov_head = use_fov_head
        self.instance_seg_head = instance_seg_head  # 实例分割头

        # 动态获取输入图像尺寸
        self.img_size = getattr(encoder, "img_size", None) or getattr(decoder, "img_size", None)
        if self.img_size is None:
            raise ValueError("Encoder or decoder must define `img_size` attribute.")

        # 初始化深度估计头
        dim_decoder = decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.head[4].bias.data.fill_(0)

        if use_fov_head:
            self.fov = FOVNetwork(num_features=dim_decoder, fov_encoder=fov_encoder)

    def forward(self, x: torch.Tensor, targets: Optional[list] = None, mode: str = "train") -> dict:
        """
        前向传播，支持训练、评估和推理模式.

        Args:
            x (torch.Tensor): 输入图像，形状为 [batch_size, C, H, W].
            targets (Optional[list]): 每张图像的目标字典列表，包含 boxes, labels 和 masks.
            mode (str): 模式，可选值为 "train", "eval", "infer".

        Returns:
            dict: 包含不同模式下的输出.
        """
        results = {}

        # 如果输入是单张图像（3维张量），添加批次维度
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        _, _, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # 编码器前向传播
        encodings = self.encoder(x)
        # 解码器前向传播
        features, features_0 = self.decoder(encodings)
        # 打印 features 的形状
        # print(f"Shape of features: {features.shape}")
        # 实例分割头处理
        if self.instance_seg_head is not None:
            if mode == "train":
                if targets is None:
                    raise ValueError("训练模式下需要提供 targets.")
                seg_results = self.instance_seg_head(features, targets=targets, mode="train")
                results["segmentation_loss"] = seg_results["loss"]

            elif mode == "eval":
                seg_results = self.instance_seg_head(features, targets=targets, mode="eval")
                results["segmentation_eval"] = seg_results["metrics"]

            elif mode == "infer":
                seg_results = self.instance_seg_head(features, mode="infer")

                # 检查 seg_results 是否为 None
                if seg_results is None:
                    raise ValueError("推理模式下，seg_results 为 None，请检查实例分割头的推理逻辑。")
                print("推理结果: ", seg_results)
                # 确保 seg_results 包含 "infer_results" 键
                if "segmentation_infer" not in seg_results:
                    raise KeyError("推理结果中没有找到 'infer_results' 键，请检查实例分割头的推理输出。")

                results = seg_results

            else:
                raise ValueError(f"未知模式 '{mode}', 请使用 'train'、'eval' 或 'infer'.")

        # 如果有深度估计模块，可在这里添加对应逻辑
        if mode in ["train", "eval", "infer"] and self.head is not None:
            depth_output = self.head(features_0)
            results["depth"] = depth_output

        return results

