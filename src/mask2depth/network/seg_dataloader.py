import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_util

class CocoSegmentationDataset(Dataset):
    """
    自定义 COCO 格式实例分割数据集.
    支持加载图像和分割标注 (mask).
    """

    def __init__(self, root: str, ann_file: str, category_mapping: dict, valid_categories: list, transforms=None, min_area=1):
        """
        初始化数据集.

        参数:
            root (str): 图像文件所在目录的路径.
            ann_file (str): COCO 格式的注释文件路径 (JSON).
            category_mapping (dict): 类别映射字典 {id: name}.
            valid_categories (list): 有效类别 ID 列表.
            transforms (callable, optional): 数据增强或预处理.
            min_area (int): 最小有效掩码面积阈值，默认为10.
        """
        self.root = root
        self.coco = COCO(ann_file)  # 加载 COCO 注释
        self.ids = list(self.coco.imgs.keys())  # 获取所有图像的 ID
        self.transforms = transforms
        self.min_area = min_area
        self.category_mapping = category_mapping
        self.valid_categories = valid_categories

    def __len__(self):
        """返回数据集的大小."""
        return len(self.ids)

    def __getitem__(self, index):
        """
        获取数据集中的一个样本.

        参数:
            index (int): 样本索引.

        返回:
            tuple: (图像, 标注字典)，其中标注字典包含 'boxes', 'labels', 'masks', 'class_names' 等信息.
        """
        # 打印当前样本的信息

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)  # 获取该图像对应的标注 ID
        anns = self.coco.loadAnns(ann_ids)  # 加载标注
        img_info = self.coco.loadImgs(img_id)[0]  # 加载图像信息

        # 加载图像
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # 提取标注信息
        boxes = []
        labels = []
        masks = []
        class_names = []
        for ann in anns:
            if 'bbox' in ann and 'segmentation' in ann and ann['category_id'] is not None:
                try:
                    # 检查类别是否合法
                    if ann['category_id'] not in self.valid_categories:
                        continue

                    # 转换为二值掩码
                    binary_mask = self._convert_segmentation_to_mask(ann, img_info['height'], img_info['width'])

                    # 验证掩码有效性
                    if binary_mask.sum() > self.min_area:
                        bbox = ann['bbox']
                        x_min, y_min, width, height = bbox
                        x_max, y_max = x_min + width, y_min + height
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(ann['category_id'])  # 类别 ID
                        masks.append(binary_mask)
                        class_names.append(self.category_mapping[ann['category_id']])  # 映射类别名称
                except Exception as e:
                    print(f"Error processing annotation ID {ann['id']} for image ID {img_id}: {e}")

        # 如果没有有效的掩码，打印调试信息并返回 None
        if len(masks) == 0:
            return None

        # print(f"Loading sample {index}: Image ID = {img_id}, Valid Masks = {len(masks)}")

        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.from_numpy(m).to(torch.uint8) for m in masks])

        # 构建目标字典
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "class_names": class_names,
            "image_id": torch.tensor([img_id]),
        }

        # 应用数据增强或预处理
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def _convert_segmentation_to_mask(self, ann, height, width):
        """
        将 segmentation 转换为二值掩码.

        参数:
            ann (dict): 单个标注信息.
            height (int): 图像高度.
            width (int): 图像宽度.

        返回:
            numpy.ndarray: 二值掩码.
        """
        segmentation = ann['segmentation']
        if isinstance(segmentation, list):  # 多边形格式
            rle = mask_util.frPyObjects(segmentation, height, width)
            binary_mask = mask_util.decode(rle)
            if len(binary_mask.shape) > 2:  # 多个对象的掩码需要合并
                binary_mask = binary_mask.any(axis=2)
        elif isinstance(segmentation, dict):  # RLE 格式
            binary_mask = mask_util.decode(segmentation)
        else:
            binary_mask = torch.zeros((height, width), dtype=torch.uint8)
        return binary_mask

