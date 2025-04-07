import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
import yaml

from mask2depth import create_model_and_transforms, DepthProConfig
from network.seg_dataloader import CocoSegmentationDataset
import multiprocessing


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
    model.train()
    epoch_loss = 0.0
    batch_losses = []
    total_batches = len(dataloader)
    grad_norms = []

    progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc="Training")

    for batch_idx, (images, targets) in progress_bar:
        if len(images) == 0 or len(targets) == 0:
            progress_bar.set_postfix({"status": "skipped empty batch"})
            continue

        try:
            images = images.to(device, non_blocking=True)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        except RuntimeError as e:
            print(f"Data moving error: {e}")
            continue

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            try:
                outputs = model(images, targets=targets, mode="train")
                segmentation_loss = outputs["segmentation_loss"]
                if torch.isnan(segmentation_loss) or torch.isinf(segmentation_loss):
                    raise RuntimeError(f"Invalid loss value: {segmentation_loss.item()}")
            except Exception as e:
                print(f"\nForward pass error: {str(e)}")
                progress_bar.set_postfix({"status": "forward error"})
                continue

        try:
            scaler.scale(segmentation_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0,
                error_if_nonfinite=False  # 修改这里
            )
            grad_norms.append(grad_norm.item())
        except RuntimeError as e:
            print(f"\nBackward error: {str(e)}")
            progress_bar.set_postfix({"status": "backward error"})
            optimizer.zero_grad()
            scaler.update()  # 重置scaler状态
            continue

        try:
            scaler.step(optimizer)
        except RuntimeError as e:
            print(f"\nOptimizer step error: {str(e)}")
        finally:
            scaler.update()

        batch_loss = segmentation_loss.item()
        batch_losses.append(batch_loss)
        epoch_loss += batch_loss

        if len(batch_losses) >= 10:
            avg_loss = sum(batch_losses[-10:]) / 10
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "grad_norm": f"{grad_norm.item():.2f}"
            })

    avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    print(f"\nEpoch Summary: Loss: {avg_epoch_loss:.4f}, Grad Norm: {avg_grad_norm:.2f}")
    return avg_epoch_loss


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
                targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in
                           targets]
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
    # 设置硬件设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载配置文件
    config_path = "./cfg/config.yaml"
    cfg = load_config(config_path)

    # 初始化类别映射
    category_mapping = load_classes_from_yaml(config_path)
    print("\n类别映射信息：")
    for k, v in category_mapping.items():
        print(f"ID {k} -> {v}")

    # 模型初始化
    model_config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=cfg["training"]["checkpoint_uri"],
        decoder_features=256,
        use_fov_head=False,
        fov_encoder_preset=None,
    )
    model, transform = create_model_and_transforms(model_config, device=device)

    # 设置可训练参数
    set_trainable_params(model,
                         freeze_encoder=False,
                         freeze_decoder=False,
                         freeze_segmentation_head=False)
    print("\n训练参数统计：")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    # 数据集初始化
    train_dataset = CocoSegmentationDataset(
        root=cfg["dataset"]["train_root"],
        ann_file=cfg["dataset"]["train_ann_file"],
        transforms=transform,
        category_mapping=category_mapping,
        valid_categories=cfg["project"]["valid_categories"],
        min_area=cfg["dataset"].get("min_area", 1),
    )
    val_dataset = CocoSegmentationDataset(
        root=cfg["dataset"]["val_root"],
        ann_file=cfg["dataset"]["val_ann_file"],
        transforms=transform,
        category_mapping=category_mapping,
        valid_categories=cfg["project"]["valid_categories"],
        min_area=cfg["dataset"].get("min_area", 1),
    )

    # 数据加载器配置（优化了num_workers设置）
    num_workers = 0  # 不超过4个worker
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=custom_collate_fn, num_workers=cfg["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=cfg["training"]["num_workers"]
    )

    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=0.05  # 增强的正则化
    )

    # 分阶段学习率调度
    warmup_epochs = 5
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs, cfg["training"]["learning_rate"])
    main_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # 根据验证IoU调整
        patience=3,
        factor=0.5,
        verbose=True
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_epoch = -1  # 新增初始化
    # 训练状态追踪
    best_val_iou = 0.0
    early_stop_counter = 0
    checkpoint_dir = cfg["training"]["save_path"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")  # 固定最佳模型路径

    print("\n开始训练...")
    for epoch in range(cfg["training"]["num_epochs"]):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{cfg['training']['num_epochs']}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 训练阶段
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device
        )

        # 验证阶段
        avg_iou = validate_one_epoch(model, val_loader, device)

        if avg_iou > best_val_iou:
            best_val_iou = avg_iou
            best_epoch = epoch  # 新增更新最佳epoch
            early_stop_counter = 0

        # 学习率调度
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(avg_iou)  # 根据验证IoU调整学习率

        last_model_path = os.path.join(checkpoint_dir, "last_model.pt")
        save_model(model, last_model_path, epoch)

        # === 修改2: 简化最佳模型判断逻辑 ===
        if avg_iou > best_val_iou:
            print(f"性能提升: {best_val_iou:.4f} → {avg_iou:.4f}")
            best_val_iou = avg_iou
            early_stop_counter = 0
            save_model(model, best_model_path, epoch)  # 保存到固定路径

        # 早停判断
        early_stop_counter += 1 if avg_iou <= best_val_iou else 0
        if early_stop_counter >= 5:
            print("验证指标连续5轮无提升，提前终止训练")
            break

        # 训练状态报告
        print(f"\nEpoch {epoch + 1} 总结:")
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证mIoU: {avg_iou:.4f}")
        print(f"最佳mIoU: {best_val_iou:.4f} (epoch {best_epoch + 1})")

    # 训练结束补充保存
    final_model_path = os.path.join(checkpoint_dir, f"final_epoch{epoch + 1}.pt")
    save_model(model, final_model_path, epoch)
    print(f"\n训练完成，最佳模型: {best_model_path}")


if __name__ == "__main__":
    main()
