import torch
import torch.nn as nn
import torch.nn.functional as F

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

class InstanceAwareSegmentationLoss(nn.Module):
    """
    Segmentation Loss with Instance Awareness:
    - Combines weighted cross-entropy loss (for semantic segmentation).
    - Instance-level loss (for separating objects of the same class).
    - Gradient-based boundary loss (for optimizing object boundaries).
    - Jaccard loss (IoU loss) to improve overlap accuracy.
    """

    def __init__(self, use_gradient_loss=True, gradient_weight=1.0, instance_weight=1.0, class_weights=None):
        super().__init__()
        self.use_gradient_loss = use_gradient_loss
        self.gradient_weight = gradient_weight
        self.instance_weight = instance_weight
        self.class_weights = class_weights  # 可以传入类别的权重

    def compute_gradient_loss(self, predicted, target, valid_mask, device):
        """
        Compute boundary gradient loss for semantic segmentation, ignoring invalid pixels.
        Args:
            predicted: Predicted probabilities [B, C, H, W].
            target: Target one-hot labels [B, C, H, W].
            valid_mask: Mask to ignore invalid pixels [B, H, W].
        Returns:
            Gradient loss value.
        """
        C = predicted.size(1)  # Number of channels (classes)
        scharr_kernel_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).to(device)
        scharr_kernel_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).to(device)

        # Expand kernels to match the number of channels
        scharr_kernel_x = scharr_kernel_x.repeat(C, 1, 1, 1)  # [C, 1, 3, 3]
        scharr_kernel_y = scharr_kernel_y.repeat(C, 1, 1, 1)  # [C, 1, 3, 3]

        grad_pred_x = F.conv2d(predicted, scharr_kernel_x, padding=1, groups=C)
        grad_pred_y = F.conv2d(predicted, scharr_kernel_y, padding=1, groups=C)

        grad_target_x = F.conv2d(target, scharr_kernel_x, padding=1, groups=C)
        grad_target_y = F.conv2d(target, scharr_kernel_y, padding=1, groups=C)

        gradient_loss = torch.abs(grad_pred_x - grad_target_x) + torch.abs(grad_pred_y - grad_target_y)

        # Apply valid mask to ignore invalid pixels
        valid_mask = valid_mask.unsqueeze(1).to(device)  # [B, 1, H, W]
        gradient_loss = gradient_loss * valid_mask

        return gradient_loss.sum() / valid_mask.sum()  # Normalize by valid pixels

    def compute_instance_loss(self, predicted_instances, target_instances, device):
        """
        Compute instance separation loss to distinguish overlapping objects.
        Args:
            predicted_instances: Predicted instance maps [B, I, H, W].
            target_instances: Ground truth instance maps [B, I, H, W].
        Returns:
            Instance loss value.
        """
        predicted_instances = predicted_instances.to(device)
        target_instances = target_instances.to(device)
        return F.mse_loss(predicted_instances, target_instances)

    def compute_jaccard_loss(self, predicted_logits, target_labels, num_classes, device):
        
        
        # 计算预测的概率分布
        predicted_probs = torch.softmax(predicted_logits, dim=1).to(device)
        
        
        # 将标签转为 one-hot 编码
        target_labels = target_labels.to(device)
        target_one_hot = F.one_hot(target_labels, num_classes=num_classes).float().to(device)
        
        # 交换维度从 [B, H, W, C] 到 [B, C, H, W]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float().to(device)
        
    
        # 计算交集和并集
        intersection = torch.sum(predicted_probs * target_one_hot, dim=(2, 3))  # [B, C]
        union = torch.sum(predicted_probs + target_one_hot, dim=(2, 3))  # [B, C]
        union = union - intersection  # 避免交集被重复计算
    
        # 计算 Jaccard 指数 (IoU)
        jaccard_index = (intersection + 1e-6) / (union + 1e-6)
        
        # 计算 Jaccard 损失
        jaccard_loss = 1 - jaccard_index  # IoU 损失是 1 - IoU
    
        return jaccard_loss.mean()  # 对所有类别取平均



    def forward(self, predicted_logits, target_labels, predicted_instances=None, target_instances=None, device=None):
        """
        Compute the combined loss.
        Args:
            predicted_logits: Logits from the model [B, C, H, W].
            target_labels: Ground truth labels [B, H, W].
            predicted_instances: Predicted instance maps [B, I, H, W].
            target_instances: Ground truth instance maps [B, I, H, W].
        Returns:
            Combined loss value.
        """
     
        loss = 0.0
        num_classes_NUMBER = project_config["num_class"]
        target_labels = torch.where(target_labels == -1, torch.full_like(target_labels, num_classes_NUMBER), target_labels)
        
        if device is None:
            device = predicted_logits.device
    
    
        # Ensure class_weights is on the same device as predicted_logits
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device)  # Move class_weights to the same device
            ce_loss = F.cross_entropy(predicted_logits, target_labels, weight=class_weights, ignore_index=-1)
        else:
            ce_loss = F.cross_entropy(predicted_logits, target_labels, ignore_index=-1)
        
        loss += ce_loss
        # print(f"Weighted Cross-Entropy Loss: {ce_loss.item():.4f}")
    
        # Jaccard Loss (IoU Loss) for semantic segmentation
        jaccard_loss_value = self.compute_jaccard_loss(predicted_logits, target_labels, num_classes=predicted_logits.size(1), device=device)
        loss += jaccard_loss_value
        # print(f"Jaccard Loss: {jaccard_loss_value.item():.4f}")
    
        # Gradient Loss for boundary refinement
        if self.use_gradient_loss:
            predicted_probs = torch.softmax(predicted_logits, dim=1).to(device)
            valid_mask = target_labels != -1  # Ignore invalid pixels
    
            # Convert target_labels to one-hot encoding
            target_one_hot = F.one_hot(
                torch.clamp(target_labels, min=0), num_classes=predicted_logits.size(1)
            ).permute(0, 3, 1, 2).float().to(device)  # [B, C, H, W]
    
            grad_loss = self.compute_gradient_loss(predicted_probs, target_one_hot, valid_mask, device=device)
            # print(f"Gradient Loss: {grad_loss.item():.4f}")
            loss += self.gradient_weight * grad_loss
    
        # Instance Loss for separating overlapping objects
        if predicted_instances is not None and target_instances is not None: 
            instance_loss = self.compute_instance_loss(predicted_instances, target_instances, device=device)
            loss += self.instance_weight * instance_loss
    
        return loss













# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pathlib import Path
# import yaml

# def load_config(yaml_path: str) -> dict:
#     """加载YAML配置文件并返回字典"""
#     with open(yaml_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# # 假设配置文件路径为当前目录下的config.yaml
# config_path = Path(__file__).parent / "../cfg/config.yaml"
# project_config = load_config(config_path)["project"]


# class InstanceAwareSegmentationLoss(nn.Module):
#     """
#     Segmentation Loss with Instance Awareness:
#     - Combines weighted cross-entropy loss (for semantic segmentation).
#     - Instance-level loss (for separating objects of the same class).
#     - Gradient-based boundary loss (for optimizing object boundaries).
#     - Jaccard loss (IoU loss) to improve overlap accuracy.
#     """

#     def __init__(self, use_gradient_loss=True, gradient_weight=1.0, instance_weight=1.0, class_weights=None):
#         super().__init__()
#         self.use_gradient_loss = use_gradient_loss
#         self.gradient_weight = gradient_weight
#         self.instance_weight = instance_weight
#         self.class_weights = class_weights  # 可以传入类别的权重

#     def compute_jaccard_loss(self, predicted_logits, target_labels, num_classes, device):
        
        
#         # 计算预测的概率分布
#         predicted_probs = torch.softmax(predicted_logits, dim=1).to(device)
        
        
#         # 将标签转为 one-hot 编码
#         target_labels = target_labels.to(device)
#         target_one_hot = F.one_hot(target_labels, num_classes=num_classes).float().to(device)
        
#         # 交换维度从 [B, H, W, C] 到 [B, C, H, W]
#         target_one_hot = target_one_hot.permute(0, 3, 1, 2).float().to(device)
        
    
#         # 计算交集和并集
#         intersection = torch.sum(predicted_probs * target_one_hot, dim=(2, 3))  # [B, C]
#         union = torch.sum(predicted_probs + target_one_hot, dim=(2, 3))  # [B, C]
#         union = union - intersection  # 避免交集被重复计算
    
#         # 计算 Jaccard 指数 (IoU)
#         jaccard_index = (intersection + 1e-6) / (union + 1e-6)
        
#         # 计算 Jaccard 损失
#         jaccard_loss = 1 - jaccard_index  # IoU 损失是 1 - IoU
    
#         return jaccard_loss.mean()  # 对所有类别取平均

#     def compute_instance_loss(self, predicted_instances, target_instances, device):
#         """
#         Compute instance separation loss to distinguish overlapping objects.
#         Args:
#             predicted_instances: Predicted instance maps [B, I, H, W].
#             target_instances: Ground truth instance maps [B, I, H, W].
#         Returns:
#             Instance loss value.
#         """
#         predicted_instances = predicted_instances.to(device)
#         target_instances = target_instances.to(device)
#         return F.mse_loss(predicted_instances, target_instances)

#     def compute_gradient_loss(self, predicted_probs, target_one_hot, valid_mask, device, kappa=0.5, lambda_angle=0.3, eps=1e-10):
#         B, C, H, W = predicted_probs.shape

#         # 使用Scharr算子替换Sobel算子
#         scharr_x = torch.tensor([[-3, 0, 3], 
#                              [-10, 0, 10], 
#                              [-3, 0, 3]], 
#                             dtype=torch.float32, device=device).repeat(C, 1, 1, 1)  # [C, 1, 3, 3]
    
#         scharr_y = torch.tensor([[-3, -10, -3], 
#                              [0, 0, 0], 
#                              [3, 10, 3]], 
#                             dtype=torch.float32, device=device).repeat(C, 1, 1, 1)

#         # 计算预测梯度场
#         G_pred_x = F.conv2d(predicted_probs, scharr_x, padding=1, groups=C)  # [B, C, H, W]
#         G_pred_y = F.conv2d(predicted_probs, scharr_y, padding=1, groups=C)

#         # 计算真值梯度场
#         with torch.no_grad():
#             G_true_x = F.conv2d(target_one_hot, scharr_x, padding=1, groups=C)
#             G_true_y = F.conv2d(target_one_hot, scharr_y, padding=1, groups=C)

#         # 计算结构张量特征值 
#         I_xx = G_true_x.pow(2)
#         I_xy = G_true_x * G_true_y
#         I_yy = G_true_y.pow(2)

#         trace = I_xx + I_yy + eps
#         det = I_xx * I_yy - I_xy.pow(2)
#         sqrt_term = torch.sqrt(torch.abs(trace.pow(2) / 4 - det) + eps)  # 防止负数开根

#         lambda1 = trace / 2 + sqrt_term
#         lambda2 = trace / 2 - sqrt_term

#         # 结构相干性系数
#         coherency = (lambda1 - lambda2).pow(2) / (lambda1 + lambda2 + eps)

#         # 梯度强度差异 (L2范数)
#         magnitude_diff = torch.sqrt((G_pred_x - G_true_x).pow(2) +
#                                     (G_pred_y - G_true_y).pow(2) + eps)

#         # 梯度方向对齐 (余弦相似度)
#         numerator = G_pred_x * G_true_x + G_pred_y * G_true_y
#         denominator = (torch.sqrt(G_pred_x.pow(2) + G_pred_y.pow(2) + eps) *
#                        torch.sqrt(G_true_x.pow(2) + G_true_y.pow(2) + eps))
#         cos_theta = torch.clamp(numerator / denominator, -1.0 + eps, 1.0 - eps)
#         angle_loss = 1 - cos_theta

#         # 组合损失项
#         total_loss = (magnitude_diff / (1 + kappa * coherency)) + lambda_angle * angle_loss

#         # 应用有效区域掩码
#         valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
#         total_loss = total_loss * valid_mask

#         return total_loss.sum() / (valid_mask.sum() + eps)



#     def forward(self, predicted_logits, target_labels, predicted_instances=None, target_instances=None, device=None):
#         """
#         Compute the combined loss.
#         Args:
#             predicted_logits: Logits from the model [B, C, H, W].
#             target_labels: Ground truth labels [B, H, W].
#             predicted_instances: Predicted instance maps [B, I, H, W].
#             target_instances: Ground truth instance maps [B, I, H, W].
#         Returns:
#             Combined loss value.
#         """
     
#         loss = 0.0

#         num_classes_NUMBER = project_config["num_class"]
#         target_labels = torch.where(target_labels == -1, torch.full_like(target_labels, num_classes_NUMBER), target_labels)
        
#         if device is None:
#             device = predicted_logits.device
    
    
#         # Ensure class_weights is on the same device as predicted_logits
#         if self.class_weights is not None:
#             class_weights = self.class_weights.to(device)  # Move class_weights to the same device
#             ce_loss = F.cross_entropy(predicted_logits, target_labels, weight=class_weights, ignore_index=-1)
#         else:
#             ce_loss = F.cross_entropy(predicted_logits, target_labels, ignore_index=-1)
        
#         loss += ce_loss
#         # print(f"Weighted Cross-Entropy Loss: {ce_loss.item():.4f}")
    
#         # Jaccard Loss (IoU Loss) for semantic segmentation
#         jaccard_loss_value = self.compute_jaccard_loss(predicted_logits, target_labels, num_classes=predicted_logits.size(1), device=device)
#         loss += jaccard_loss_value
#         # print(f"Jaccard Loss: {jaccard_loss_value.item():.4f}")
    
#         # Gradient Loss for boundary refinement
#         if self.use_gradient_loss:
#             predicted_probs = torch.softmax(predicted_logits, dim=1).to(device)
#             valid_mask = target_labels != -1  # Ignore invalid pixels
    
#             # Convert target_labels to one-hot encoding
#             target_one_hot = F.one_hot(
#                 torch.clamp(target_labels, min=0), num_classes=predicted_logits.size(1)
#             ).permute(0, 3, 1, 2).float().to(device)  # [B, C, H, W]
    
#             grad_loss = self.compute_gradient_loss(predicted_probs, target_one_hot, valid_mask, device=device)
#             # print(f"Gradient Loss: {grad_loss.item():.4f}")
#             loss += self.gradient_weight * grad_loss
    
#         # Instance Loss for separating overlapping objects
#         if predicted_instances is not None and target_instances is not None: 
#             instance_loss = self.compute_instance_loss(predicted_instances, target_instances, device=device)
#             loss += self.instance_weight * instance_loss
    
#         return loss


















