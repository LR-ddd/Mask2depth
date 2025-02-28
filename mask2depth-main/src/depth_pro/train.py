
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
import yaml

from depth_pro import create_model_and_transforms, DepthProConfig
from network.seg_dataloader import CocoSegmentationDataset


def load_config(config_path):
    """加载 YAML 配置文件."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_classes_from_yaml(yaml_file_path):
    """从 YAML 文件中加载类别映射为字典."""
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    classes = config["project"]["classes"]
    return {item["id"]: item["name"] for item in classes}


def custom_collate_fn(batch):
    """自定义 collate 函数，过滤掉无效样本."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print("Warning: Batch is empty after filtering invalid samples.")
        return [], []

    try:
        images, targets = zip(*batch)
        return torch.stack(images, dim=0), list(targets)
    except Exception as e:
        print(f"Error during collation: {e}")
        return [], []


def train_one_epoch(model, dataloader, optimizer, scaler, device):
    """
    训练一个 epoch.
    """
    model.train()
    epoch_loss = 0.0
    batch_losses = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        if len(images) == 0 or len(targets) == 0:
            print(f"Skipping empty batch at index {batch_idx}")
            continue

        images = images.to(device)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            try:
                outputs = model(images, targets=targets, mode="train")
                segmentation_loss = outputs.get("segmentation_loss", None)
                if segmentation_loss is None:
                    raise ValueError("Model output does not contain 'segmentation_loss'.")
            except Exception as e:
                print(f"Error in forward pass for batch {batch_idx}: {e}")
                print("1\n")
                continue
        
        if torch.isnan(segmentation_loss) or torch.isinf(segmentation_loss):
            print(f"Loss is NaN or Inf for batch {batch_idx}. Skipping this batch.")
            print("2\n")
            continue
        
        try:
            scaler.scale(segmentation_loss).backward()
        except RuntimeError as e:
            print(f"Error during backward pass for batch {batch_idx}: {e}")
            print("3\n")
            continue
        
        if torch.isinf(segmentation_loss) or torch.isnan(segmentation_loss):
            print(f"Skipping optimizer step due to NaN or Inf in loss for batch {batch_idx}")
            print("4\n")
            continue
        
        try:
            scaler.step(optimizer)
        except RuntimeError as e:
            print(f"Error during optimizer step: {e}")
            continue

        scaler.update()

        batch_losses.append(segmentation_loss.item())
        epoch_loss += segmentation_loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            avg_batch_loss = sum(batch_losses[-10:]) / min(10, len(batch_losses))
            print(f"Average Loss of last 10 batches: {avg_batch_loss:.4f}")

    avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    print(f"\nEpoch Average Loss: {avg_loss:.4f}")

    return avg_loss


def validate_one_epoch(model, dataloader, device):
    """
    验证一个完整的 epoch.
    """
    model.eval()
    pixel_accuracies = []
    mean_ious = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                print(f"Skipping empty batch at index {batch_idx}")
                continue

            try:
                images, targets = batch
                images = images.to(device)
                targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
                outputs = model(images, targets=targets, mode="eval")
                metrics = outputs.get("segmentation_eval", {})
                pixel_accuracies.append(metrics.get("pixel_accuracy", 0.0))
                mean_ious.append(metrics.get("mean_iou", 0.0))
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    avg_pixel_accuracy = sum(pixel_accuracies) / len(pixel_accuracies) if pixel_accuracies else 0.0
    avg_mean_iou = sum(mean_ious) / len(mean_ious) if mean_ious else 0.0

    print(f"Validation Pixel Accuracy: {avg_pixel_accuracy:.4f}, Mean IoU: {avg_mean_iou:.4f}")
    return avg_mean_iou


def save_model(model, path, epoch):
    """保存模型检查点."""
    directory = os.path.dirname(path)  # 获取文件的父目录
    
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)  # 使用 exist_ok=True，防止目录已经存在时出错
        print(f"Directory created: {directory}")

    # 如果路径是文件并且已经存在，重命名或删除旧文件
    if os.path.isfile(path):
        os.rename(path, path + ".bak")  # 备份旧文件

    # 保存模型
    torch.save(model.state_dict(), path)
    print(f"Model saved at epoch {epoch + 1}: {path}")



def set_trainable_params(model, freeze_encoder=True, freeze_decoder=True, freeze_segmentation_head=True):
    """
    设置模型的各个部分是否需要更新其参数.
    """
    for param in model.parameters():
        param.requires_grad = False

    if not freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = True

    if not freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = True

    if not freeze_segmentation_head:
        for param in model.instance_seg_head.parameters():
            param.requires_grad = True


def get_warmup_scheduler(optimizer, warmup_epochs, max_lr):
    """
    获取线性 warmup 的学习率调度器
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性增长
        else:
            return 1.0  # 达到最大学习率后保持

    return LambdaLR(optimizer, lr_lambda)


def main():
    config_path = "./cfg/config.yaml"
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category_mapping = load_classes_from_yaml(config_path)
    print("类别映射信息", category_mapping)

    model_config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri="./checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=False,
        fov_encoder_preset=None,
    )
    model, transform = create_model_and_transforms(model_config, device=device)

    set_trainable_params(model, freeze_encoder=False, freeze_decoder=False, freeze_segmentation_head=False)

    train_dataset = CocoSegmentationDataset(
        root=cfg["dataset"]["train_root"],
        ann_file=cfg["dataset"]["train_ann_file"],
        transforms=transform,
        category_mapping=category_mapping,
        valid_categories=cfg["project"]["valid_categories"],
        min_area=1,
    )
    val_dataset = CocoSegmentationDataset(
        root=cfg["dataset"]["val_root"],
        ann_file=cfg["dataset"]["val_ann_file"],
        transforms=transform,
        category_mapping=category_mapping,
        valid_categories=cfg["project"]["valid_categories"],
        min_area=1,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=custom_collate_fn, num_workers=cfg["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=cfg["training"]["num_workers"]
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=1e-4)
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs=5, max_lr=cfg["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    scaler = torch.cuda.amp.GradScaler()

    best_val_iou = 0.0  # 初始化最好的 Mean IoU 为一个很小的值
    best_epoch = -1  # 用来跟踪最好的 epoch

    for epoch in range(cfg["training"]["num_epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['training']['num_epochs']}")

        # 训练一个 epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)

        # 更新 warmup 调度器
        warmup_scheduler.step(epoch)

        # 验证并获取 Mean IoU
        avg_mean_iou = validate_one_epoch(model, val_loader, device)

        # 比较当前 epoch 的 mean_iou 是否更好
        if avg_mean_iou > best_val_iou:
            best_val_iou = avg_mean_iou
            best_epoch = epoch

            # 保存最好的模型
            save_model(model, f"{cfg['training']['save_path']}/best_model_epoch_{epoch + 1}_mean_iou_{best_val_iou:.4f}.pt", epoch)

        # 每轮结束后更新学习率
        scheduler.step(train_loss)

        print(f"Best Mean IoU so far: {best_val_iou:.4f} at epoch {best_epoch + 1}")




if __name__ == "__main__":
    main()

