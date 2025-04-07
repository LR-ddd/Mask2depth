from pycocotools.coco import COCO
import numpy as np

# 配置参数
annFile = '/root/autodl-tmp/codes/ml-depth-pro-main/data/cotton_datas/labels/train.json'
background_id = 11  # 背景类索引

# 初始化COCO API
coco = COCO(annFile)

# 创建类别字典 (背景+COCO类别)
categories = coco.loadCats(coco.getCatIds())
class_dict = {background_id: {'name': 'background', 'count': 0}}
class_dict.update({cat['id']: {'name': cat['name'], 'count': 0} for cat in categories})

# 遍历所有图片统计像素
total_pixels = 0
for img_id in coco.getImgIds():
    img = coco.loadImgs(img_id)[0]
    h, w = img['height'], img['width']
    total_pixels += h * w  # 累计总像素
    
    # 统计前景像素
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        mask = coco.annToMask(ann)
        class_dict[ann['category_id']]['count'] += np.sum(mask)

# 计算背景像素（总像素 - 所有前景像素）
foreground_total = sum(v['count'] for k,v in class_dict.items() if k != background_id)
class_dict[background_id]['count'] = total_pixels - foreground_total

# 计算平滑后的频率
smooth_factor = 1e3  # 可调整的平滑因子
num_classes = len(class_dict)
total_pixels_smoothed = total_pixels + smooth_factor * num_classes
freq_dict = {
    cls: (info['count'] + smooth_factor) / total_pixels_smoothed
    for cls, info in class_dict.items()
}

# 计算原始权重
median = np.median(list(freq_dict.values()))
raw_weights = {cls: median / freq for cls, freq in freq_dict.items()}

# 对数缩放 + 线性归一化到[1,5]
valid_weights = [w for w in raw_weights.values() if w > 0]
log_weights = np.log(valid_weights)
max_log = np.max(log_weights)
min_log = np.min(log_weights)

weight_dict = {}
for cls in class_dict:
    if raw_weights[cls] <= 0:
        weight_dict[cls] = 0.0
        continue
    # 对数变换
    log_w = np.log(raw_weights[cls])
    # 线性映射到1-5
    scaled_w = 1 + (log_w - min_log) / (max_log - min_log) * 4
    weight_dict[cls] = np.clip(scaled_w, 1.0, 5.0)  # 确保严格在范围内

# 打印结果
print(f"{'ID':<4} {'Class':<15} {'Pixels':<12} {'Weight':<8}")
print('-' * 35)
for cls in sorted(class_dict.keys()):
    info = class_dict[cls]
    print(f"{cls:<4} {info['name']:<15} {info['count']:<12} {weight_dict[cls]:.2f}")