import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from .loss_function import InstanceAwareSegmentationLoss

from pathlib import Path
import yaml

def load_config(yaml_path: str) -> dict:
    """加载YAML配置文件并返回字典"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 假设配置文件路径为当前目录下的config.yaml
config_path = Path(__file__).parent / "../cfg/config.yaml"
project_config = load_config(config_path)["project"]

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, threshold=0.5):
        """
        初始化分割头。

        参数:
        - in_channels: 输入特征图的通道数。
        - num_classes: 类别数量（包括背景）。
        - threshold: 后处理时的概率阈值，用于生成二值图。
        """
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold

        # 分割头模块
        self.segmentation_head = nn.Sequential(
            # 第一层卷积：降维
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 反卷积：上采样
            nn.ConvTranspose2d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),

            # 第二层卷积：提取特征
            nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 第三层卷积：输出通道匹配分割类别数
            nn.Conv2d(
                in_channels=in_channels // 4,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        # 提取独立权重字段
        class_weights = torch.tensor(project_config["class_weights"])

        # 假设有4个类别
        # class_weights = torch.tensor([1.0, 1.0, 3, 1.5,0.4])  # 类别权重，调整每个类别的损失
        # class_weights = torch.tensor([10.6,5.4,9.6,9.0,4.5,9.21,10.5,115.3,31.1,28,183.2,10])
        # class_weights = torch.tensor([2.0,1.0,2.0,1.8,1.0,0.8,2.0,5.0,3.0,2.8,5.0,1.0])
        # class_weights = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
        # 使用传入的权重
        self.loss_fn = InstanceAwareSegmentationLoss(
            use_gradient_loss=True,
            gradient_weight=0.5,
            instance_weight=1.0,
            class_weights=class_weights  # 关键修改点
        )


    def forward(self, features, targets=None, mode="train"):
        """
        进行前向传播。根据模式 (train 或 eval 或 infer) 进行训练或推理。

        参数：
        - features: 输入的特征图，形状为 [batch_size, channels, height, width]
        - targets: 目标数据，仅在训练模式下使用，包含每个实例的掩码和标签
        - mode: 当前模式，'train'、'eval' 或 'infer'

        返回：
        - results: 包含损失和评估指标的字典
        """
        batch_size, _, height, width = features.shape
        results = {}

        # 分割预测
        logits = self.segmentation_head(features)  # [batch_size, num_classes, new_height, new_width]
        # 打印 logits 的形状
        # print(f"Shape of logits: {logits.shape}")
        logits_height, logits_width = logits.shape[2], logits.shape[3]

        if mode == "train":
            # 初始化语义分割标签，形状为 [batch_size, height, width]
            semantic_labels = torch.full(
                (batch_size, height, width), fill_value=-1, device=features.device, dtype=torch.long
            )
        
            for i in range(batch_size):
                target = targets[i]
                masks = target["masks"]
                labels = target["labels"]
        
                # 检查 masks 和 labels 数量
                if len(masks) != len(labels):
                    raise ValueError(
                        f"Batch {i} has mismatched masks ({len(masks)}) and labels ({len(labels)})."
                    )
        
                for j in range(len(labels)):
                    mask = masks[j]
                    label = labels[j].item()
        
                    # 验证 label 范围
                    if label < 0 or label >= logits.shape[1]:
                        raise ValueError(f"Invalid label {label} for batch {i}, instance {j}.")
        
                    # 调整 mask 到特征图大小
                    resized_mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(height, width),
                        mode="nearest"
                    ).squeeze(0).squeeze(0).bool()
        
                    if resized_mask.sum() == 0:
                        print(f"Mask {j} for label {label} has no valid pixels.")
                        continue
        
                    # 更新语义标签
                    semantic_labels[i][resized_mask] = label
        
                # # 打印当前 batch 语义标签中的类别分布
                # unique_labels = torch.unique(semantic_labels[i])  # 获取当前 batch 中出现的唯一标签
                # print(f"Batch {i} - Unique categories in semantic labels: {unique_labels.tolist()}")  # 打印类别ID
                # print(f"Batch {i} - Number of unique categories: {len(unique_labels)}")  # 打印类别数量
        
            # 上采样语义标签到 logits 的大小
            semantic_labels = F.interpolate(
                semantic_labels.unsqueeze(1).float(),  # [batch_size, 1, height, width]
                size=(logits_height, logits_width),
                mode="nearest"
            ).squeeze(1).long()  # [batch_size, logits_height, logits_width]
        
            # # 打印整个 batch 中语义标签的类别分布
            # unique_labels_all_batches = torch.unique(semantic_labels)  # 获取整个 batch 中出现的唯一标签
            # print(f"All batches - Unique categories in semantic labels: {unique_labels_all_batches.tolist()}")
            # print(f"All batches - Number of unique categories: {len(unique_labels_all_batches)}")
        
            # 计算损失
            loss = self.loss_fn(
                predicted_logits=logits,
                target_labels=semantic_labels,
                predicted_instances=None,
                target_instances=None
            )
            results["loss"] = loss
        
            return results



        elif mode == "eval":
            # 计算概率图，仅在评估模式中使用
            probs = logits.softmax(dim=1)  # [batch_size, num_classes, logits_height, logits_width]

            # 初始化指标
            total_pixels = 0
            correct_pixels = 0
            iou_per_class = torch.zeros(self.num_classes, device=features.device)
            total_intersections = torch.zeros(self.num_classes, device=features.device)
            total_unions = torch.zeros(self.num_classes, device=features.device)

            for i in range(batch_size):
                target = targets[i]
                masks = target["masks"]
                labels = target["labels"]
                # 构造 ground truth 标签
                pixel_labels = torch.full((height, width), fill_value=-1, device=features.device, dtype=torch.long)

                for j in range(len(labels)):
                    mask = masks[j]
                    label = labels[j]

                    resized_mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(height, width),
                        mode="nearest"
                    ).squeeze(0).squeeze(0).bool()
                    pixel_labels[resized_mask] = label

                # 上采样标签到 logits 的大小
                pixel_labels = F.interpolate(
                    pixel_labels.unsqueeze(0).unsqueeze(0).float(),
                    size=(logits_height, logits_width),
                    mode="nearest"
                ).squeeze(0).squeeze(0).long()

                # 获取预测
                preds = probs[i].argmax(dim=0)  # [logits_height, logits_width]
                # 计算像素准确率
                valid_mask = pixel_labels != -1
                total_pixels += valid_mask.sum().item()
                correct_pixels += (preds[valid_mask] == pixel_labels[valid_mask]).sum().item()

                # 计算每类 IoU
                for cls in range(self.num_classes):
                    pred_mask = preds == cls
                    true_mask = pixel_labels == cls
                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()
                    total_intersections[cls] += intersection
                    total_unions[cls] += union

            # 计算每类 IoU
            for cls in range(self.num_classes):
                if total_unions[cls] > 0:
                    iou_per_class[cls] = total_intersections[cls] / total_unions[cls]
            # 平均 IoU，考虑所有类别
            mean_iou = iou_per_class.mean().item()
            # 添加指标到结果
            results["metrics"] = {
                "pixel_accuracy": correct_pixels / total_pixels,
                "mean_iou": mean_iou
            }
            return results



        elif mode == "infer":
            # print("特征图的形状",logits.shape)
            # 推理模式下，返回预测结果
            probs = logits.softmax(dim=1)  # [batch_size, num_classes, logits_height, logits_width]
            pred_labels = probs.argmax(dim=1)  # [batch_size, logits_height, logits_width]
            # 对每个样本生成彩色实例分割图
            color_maps = []
            for i in range(batch_size):
                # 使用softmax后得到的概率图，生成彩色实例分割图
                color_map = self.generate_instance_color_map(probs[i].cpu().numpy())
                color_maps.append(color_map)
            # 将推理结果添加到字典中
            results["segmentation_infer"] = {
                "pred_labels": pred_labels,  # 返回预测的标签图
                "color_maps": color_maps,  # 返回彩色实例分割图
            }
            # 打印推理结果内容，查看是否包含 'segmentation_infer'

            return results

    def generate_instance_color_map(self, class_probs):
        num_classes, height, width = class_probs.shape
        color_map = np.zeros((height, width, 3), dtype=np.uint8)  # 初始化彩色图
        for cls in range(0, num_classes):  # 跳过背景类（通常背景类为类0）
            # 二值化：阈值可以调整
            binary_mask = class_probs[cls] > 0.01
            # 连通域标记，标记每个实例
            labeled_mask = label(binary_mask)  # 每个实例得到一个独特的标号
            # 将标记的实例转换为颜色
            instance_colors = label2rgb(labeled_mask, bg_label=0, bg_color=(0, 0, 0), colors=None)
            # 将生成的颜色图添加到整体颜色图中
            color_map += (instance_colors * 255).astype(np.uint8)  # 转换到 0-255 范围

        return color_map

    def display_semantic_labels(self, semantic_labels):
        """
        使用 matplotlib 展示填充后的语义标签图像。
        """
        # 假设类别数目为 num_classes

        num_classes = semantic_labels.max() + 1
        # print("类别数", num_classes)
        # 创建一个颜色映射，按类别分配不同的颜色
        color_map = plt.get_cmap('tab20', num_classes)  # 使用 tab20 色图，最多支持 20 类

        # 如果 batch_size > 1，我们选择展示第一个样本
        label_image = semantic_labels[1].cpu().numpy()
        # 显示语义标签图像
        plt.imshow(label_image, cmap=color_map)
        plt.colorbar()
        plt.title('Semantic Segmentation Labels')
        plt.show()
