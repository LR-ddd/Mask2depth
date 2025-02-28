import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import numpy as np
from PIL import Image
from torch.cuda.amp import autocast
from collections import defaultdict

# ==================== 硬件配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ==================== 颜色映射处理器（支持模糊匹配） ====================
class ColorMapperGPU:
    def __init__(self, color_map, delta=0):
        """
        :param color_map: 颜色到类别的映射字典，格式：{(R,G,B): class_id, ...}
        :param delta: 颜色容差范围（0表示精确匹配）
        """
        self.delta = delta
        self.color_tensor = torch.zeros((256, 256, 256),
                                        dtype=torch.long,
                                        device=device)

        # 记录颜色映射关系用于距离计算
        self.base_colors = torch.tensor(list(color_map.keys()),
                                        dtype=torch.float32,
                                        device=device)
        self.class_ids = torch.tensor(list(color_map.values()),
                                      dtype=torch.long,
                                      device=device)

        # 生成模糊映射表
        self._generate_fuzzy_mapping(color_map, delta)

        # 统计信息
        self.undefined_colors = set()
        self.ambiguous_mappings = defaultdict(int)
        self.total_unmatched = 0

    def _generate_fuzzy_mapping(self, color_map, delta):
        """预生成颜色模糊映射表"""
        # 先填充精确匹配
        for color, class_id in color_map.items():
            r, g, b = color
            self.color_tensor[r, g, b] = class_id

        if delta == 0:
            return

        # 生成模糊区域（三维立方体）
        delta_range = range(-delta, delta + 1)
        for (r, g, b), class_id in color_map.items():
            for dr in delta_range:
                for dg in delta_range:
                    for db in delta_range:
                        nr = r + dr
                        ng = g + dg
                        nb = b + db
                        if 0 <= nr < 256 and 0 <= ng < 256 and 0 <= nb < 256:
                            current_val = self.color_tensor[nr, ng, nb].item()
                            if current_val == 0:
                                self.color_tensor[nr, ng, nb] = class_id
                            elif current_val != class_id:
                                self.ambiguous_mappings[(nr, ng, nb)] += 1
                                # pass

    @autocast()
    def convert(self, img_tensor):
        """转换图像张量到类别矩阵（支持模糊匹配）"""
        if img_tensor.dtype != torch.uint8:
            raise ValueError("输入张量必须是uint8类型")

        original_shape = img_tensor.shape
        img_flat = img_tensor.view(-1, 3)

        # 直接查找映射表
        r = img_flat[:, 0].long()
        g = img_flat[:, 1].long()
        b = img_flat[:, 2].long()
        classes = self.color_tensor[r, g, b]

        # 处理未匹配像素（当delta>0时）
        if self.delta > 0:
            unmatched_mask = (classes == 0)
            self.total_unmatched += unmatched_mask.sum().item()

            if unmatched_mask.any():
                # 计算最近邻
                unmatched_colors = img_flat[unmatched_mask].float()
                distances = torch.cdist(unmatched_colors, self.base_colors)
                min_dist, min_idx = torch.min(distances, dim=1)

                # 应用距离阈值（三维欧氏距离）
                max_distance = self.delta * np.sqrt(3)
                valid_mask = (min_dist <= max_distance)

                # 更新类别
                classes[unmatched_mask] = torch.where(
                    valid_mask,
                    self.class_ids[min_idx],
                    torch.tensor(0, device=device)
                )

                # 统计未匹配颜色
                if not valid_mask.all():
                    invalid_colors = unmatched_colors[~valid_mask].byte().cpu().numpy()
                    for c in set(map(tuple, invalid_colors)):
                        self.undefined_colors.add(tuple(c))

        return classes.view(original_shape[:-1])


# ==================== 进度跟踪器 ====================
class ProgressTracker:
    def __init__(self, total_files):
        self.start_time = time.time()
        self.total_files = total_files
        self.processed = 0
        self.errors = 0
        self.error_messages = []

    def update(self, success=True, count=1, message=""):
        self.processed += count
        if not success:
            self.errors += count
            if message:
                self.error_messages.append(message)

    def get_progress(self):
        elapsed = time.time() - self.start_time
        progress = self.processed / self.total_files * 100
        remaining = (elapsed / max(self.processed, 1)) * (self.total_files - self.processed)
        return {
            'percentage': min(progress, 100),
            'processed': self.processed,
            'remaining': remaining,
            'errors': self.errors,
            'total': self.total_files
        }


# ==================== 指标计算核心 ====================
class MetricCalculator:
    @staticmethod
    @autocast()
    def pixel_accuracy(y_true, y_pred):
        return torch.mean((y_true == y_pred).float())

    @staticmethod
    @autocast()
    def calculate_iou(y_true, y_pred, class_stats):
        unique_classes = torch.unique(torch.cat([y_true.flatten(), y_pred.flatten()]))
        for c in unique_classes:
            c = c.item()
            true_mask = (y_true == c)
            pred_mask = (y_pred == c)

            intersection = torch.logical_and(true_mask, pred_mask).sum().item()
            union = torch.logical_or(true_mask, pred_mask).sum().item()

            if union == 0:
                continue

            if c not in class_stats:
                class_stats[c] = {'intersection': 0, 'union': 0}

            class_stats[c]['intersection'] += intersection
            class_stats[c]['union'] += union

    @staticmethod
    @autocast()
    def boundary_f1(y_true, y_pred):
        def get_boundary(mask):
            h, w = mask.shape
            horizontal = (mask[:, :-1] != mask[:, 1:])
            vertical = (mask[:-1, :] != mask[1:, :])

            boundary = torch.zeros((h, w), dtype=torch.bool, device=device)
            boundary[:, :-1] |= horizontal
            boundary[:, 1:] |= horizontal
            boundary[:-1, :] |= vertical
            boundary[1:, :] |= vertical
            return boundary

        true_boundary = get_boundary(y_true)
        pred_boundary = get_boundary(y_pred)

        tp = torch.logical_and(pred_boundary, true_boundary).sum().item()
        fp = torch.logical_and(pred_boundary, ~true_boundary).sum().item()
        fn = torch.logical_and(~pred_boundary, true_boundary).sum().item()

        return tp, fp, fn


# ==================== 主评估函数 ====================
def evaluate_segmentation(annot_dir, pred_dir, color_map, batch_size=8, color_delta=0):
    # 初始化组件
    color_mapper = ColorMapperGPU(color_map, delta=color_delta)
    progress = ProgressTracker(len(os.listdir(annot_dir)))
    metric_calculator = MetricCalculator()

    # 统计量存储
    class_stats = {}
    total_pa, total_tp, total_fp, total_fn = 0.0, 0, 0, 0
    valid_count = 0

    # 文件列表
    png_files = sorted([f for f in os.listdir(annot_dir) if f.endswith('.png')])

    print(f"\n开始评估 {len(png_files)} 张图像 (批量大小: {batch_size})...")
    print(f"颜色容差设置: ±{color_delta}（各通道）")

    # 批量处理循环
    for batch_start in range(0, len(png_files), batch_size):
        batch_files = png_files[batch_start:batch_start + batch_size]
        batch_annot, batch_pred = [], []
        valid_files = []

        try:
            # 加载并预处理图像
            for filename in batch_files:
                annot_path = os.path.join(annot_dir, filename)
                pred_path = os.path.join(pred_dir, filename)

                # 跳过缺失文件
                if not os.path.exists(pred_path):
                    progress.update(success=False, message=f"文件缺失: {pred_path}")
                    continue

                # 加载图像
                try:
                    with Image.open(annot_path) as annot_img:
                        annot = torch.from_numpy(np.array(annot_img.convert('RGB'))).to(device)
                    with Image.open(pred_path) as pred_img:
                        pred = torch.from_numpy(np.array(pred_img.convert('RGB'))).to(device)
                except Exception as e:
                    progress.update(success=False, message=f"图像加载失败: {filename} ({str(e)})")
                    continue

                # 颜色转换
                try:
                    annot = color_mapper.convert(annot)
                    pred = color_mapper.convert(pred)
                except Exception as e:
                    progress.update(success=False, message=f"颜色转换失败: {filename} ({str(e)})")
                    continue

                # 尺寸验证
                if annot.shape != pred.shape:
                    progress.update(success=False, message=f"尺寸不匹配: {filename}")
                    continue

                batch_annot.append(annot)
                batch_pred.append(pred)
                valid_files.append(filename)

            if not batch_annot:
                continue

            # 堆叠批量数据
            annot_batch = torch.stack(batch_annot)
            pred_batch = torch.stack(batch_pred)

            # 计算指标
            batch_pa = metric_calculator.pixel_accuracy(annot_batch, pred_batch)
            total_pa += batch_pa.item() * len(valid_files)

            # 交并比统计
            for a, p in zip(annot_batch, pred_batch):
                metric_calculator.calculate_iou(a, p, class_stats)

            # 边界F1统计
            for a, p in zip(annot_batch, pred_batch):
                tp, fp, fn = metric_calculator.boundary_f1(a, p)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            valid_count += len(valid_files)
            progress.update(success=True, count=len(valid_files))

            # 显存清理
            del annot_batch, pred_batch
            torch.cuda.empty_cache()

        except Exception as e:
            error_msg = f"处理批次 {batch_start // batch_size} 时出错: {str(e)}"
            progress.update(success=False, count=batch_size, message=error_msg)

        # 更新进度
        print_progress(progress.get_progress())

    # ========== 最终指标计算 ==========
    if valid_count == 0:
        print("\n错误：没有有效文件可处理")
        return

    # 颜色映射统计
    print("\n\n====== 颜色映射统计 ======")
    print(f"预设颜色数量: {len(color_map)}")
    print(f"容差范围: ±{color_delta}（各通道）")
    print(f"模糊映射冲突次数: {sum(color_mapper.ambiguous_mappings.values())}")
    print(f"未匹配像素总数: {color_mapper.total_unmatched}")
    print(f"最终未定义颜色数量: {len(color_mapper.undefined_colors)}")
    if color_mapper.undefined_colors:
        print("示例未定义颜色（最多10个）:", list(color_mapper.undefined_colors)[:10])

    # 性能指标计算
    mean_pa = total_pa / valid_count

    iou_list = []
    class_iou = {}
    for c, stats in class_stats.items():
        if stats['union'] > 0:
            iou = stats['intersection'] / stats['union']
            iou_list.append(iou)
            class_iou[c] = iou
    mean_iou = np.mean(iou_list) if iou_list else 0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 结果展示
    print("\n====== 评估结果 ======")
    print(f"有效文件数: {valid_count}/{len(png_files)}")
    print(f"平均像素准确率 (PA): {mean_pa:.4f}")
    print(f"平均交并比 (mIoU): {mean_iou:.4f}")
    print(f"边界F1分数: {f1:.4f}")
    print(f"边界精确率: {precision:.4f}")
    print(f"边界召回率: {recall:.4f}")

    print("\n各类别IoU:")
    for c in sorted(class_iou.keys()):
        print(f"类别 {c}: {class_iou[c]:.4f}")

    if progress.error_messages:
        print("\n错误汇总（最多显示10条）:")
        for msg in progress.error_messages[:10]:
            print(f"- {msg}")
    print("======================")


def print_progress(progress):
    bar_length = 30
    filled = int(round(bar_length * progress['percentage'] / 100))
    bar = '█' * filled + '-' * (bar_length - filled)

    message = (f"\r进度: |{bar}| {progress['percentage']:.1f}% "
               f"[已处理: {progress['processed']}/{progress['total']}] "
               f"剩余时间: {progress['remaining']:.1f}s "
               f"错误: {progress['errors']}")
    print(message, end='', flush=True)


if __name__ == "__main__":
    # ========== 配置区域 ==========
    # COLOR_MAP = {
    #     (158, 218, 229): 0,  # 背景
    #     (31, 119, 179): 1,  # 类别1
    #     (152, 223, 128): 2,  # 类别2
    #     (214, 39, 40): 3,  # 类别3
    #     (247, 182, 210): 4  # 类别4
    # }

    COLOR_MAP = {
        (255, 124, 14): 0,
        (44, 160, 44): 1,
        (199, 199, 199): 2,
        (197, 176, 213): 3,
        (152, 233, 138): 4,
        (255, 187, 120): 5,
        (174, 199, 232): 6,
        (255, 152, 150): 7,
        (140, 86, 75): 8,
        (188, 189, 34): 9,
        (148, 103, 189): 10,
        (157, 216, 227): 11,

    }

    ANNOTATIONS_DIR = "./datas/mask_images/yaogan"  # 标注图像目录
    PREDICTIONS_DIR = "./deeplabv3plus/yaogan"  # 预测图像目录
    BATCH_SIZE = 2  # 根据GPU显存调整
    COLOR_DELTA = 5  # 颜色容差参数（0为精确匹配）
    # =============================

    evaluate_segmentation(
        annot_dir=ANNOTATIONS_DIR,
        pred_dir=PREDICTIONS_DIR,
        color_map=COLOR_MAP,
        batch_size=BATCH_SIZE,
        color_delta=COLOR_DELTA
    )