import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import yaml
from mask2depth import create_model_and_transforms, DepthProConfig
from network.seg_dataloader import CocoSegmentationDataset
from tqdm import tqdm
import numpy as np


# 载入配置文件
def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# 从 YAML 文件中加载类别映射为字典
def load_classes_from_yaml(yaml_file_path):
    """从 YAML 文件中加载类别映射为字典"""
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    classes = config["project"]["classes"]
    return {item["id"]: item["name"] for item in classes}


def load_model(checkpoint_path, model, device):
    """加载训练好的模型"""
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    return model


def infer(model, image, transform, device):
    """进行图像推理并返回分割结果"""
    with torch.no_grad():
        # 图像预处理
        input_tensor = transform(image).unsqueeze(0).to(device)  # 增加批次维度并转移到设备
        outputs = model(input_tensor, mode="infer")  # 传递 mode="infer" 给模型进行推理

        # 从输出字典中提取推理结果
        segmentation_infer = outputs.get("segmentation_infer", {})
        pred_labels = segmentation_infer.get("pred_labels", None)
        color_maps = segmentation_infer.get("color_maps", None)

        # 返回推理的结果（标签图和彩色分割图）
        return pred_labels, color_maps


def display_segmentation_result(pred_labels, color_maps, label_map, image, save_path=None):
    """显示并保存分割结果"""
    # 转换为 numpy 数组并去除批次维度
    pred_labels = pred_labels.squeeze(0).cpu().numpy().astype(np.uint8)
    
    # 自定义颜色映射（示例使用 COCO 标准颜色）
    CLASS_COLORS = [
        # (31, 119, 179),    # 类别0: 红色
        # (152, 223, 128),    # 类别1: 绿色
        # (214, 39, 40),    # 类别2: 蓝色
        # (247,182,210),    # 类别2: 蓝色
        # (158, 218, 229)   # 类别2: 蓝色
        # 添加更多颜色...
        (255,124,14),
        (44,160,44),
        (199,199,199),
        (197,176,213),
        (152,233,138),
        (255,187,120),
        (174,199,232),
        (255,152,150),
        (140,86,75),
        (188,189,34),
        (148,103,189),
        (157,216,227)
    ]
    
    # 生成 RGB 图像
    h, w = pred_labels.shape
    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个类别填充颜色
    for class_id in np.unique(pred_labels):
        mask = (pred_labels == class_id)
        if class_id < len(CLASS_COLORS):
            rgb_array[mask] = CLASS_COLORS[class_id]
        else:
            # 处理未定义颜色的类别
            rgb_array[mask] = (128, 128, 128)  # 默认灰色
    
    # 显示结果
    # plt.imshow(rgb_array)
    # plt.axis('off')
    # plt.show()
    
    # 保存为24位RGB图像
    if save_path:
        result_image = Image.fromarray(rgb_array)
        result_image.save(save_path)
        print(f"保存24位分割图到：{save_path}")

        # 保存color_map（如果存在）
        if color_maps is not None:
            color_map = color_maps[0].squeeze(0).cpu().numpy()
            color_map = (color_map * 255).astype(np.uint8)
            Image.fromarray(color_map).save(save_path.replace('.png', 'result.png'))



def main():
    # 配置路径
    config_path = "./cfg/config.yaml"
    checkpoint_path = "./checkpoints/05539Yaogan.pt"
    input_folder = "./deploy/origin"  # 新增：输入文件夹路径
    output_folder = "./deploy/result"  # 新增：输出文件夹路径

    # 加载配置和类别映射
    cfg = load_config(config_path)
    category_mapping = load_classes_from_yaml(config_path)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型配置（只需初始化一次）
    model_config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=None,
        decoder_features=256,
        use_fov_head=False,
        fov_encoder_preset=None,
    )
    model, transform = create_model_and_transforms(model_config, device=device)
    model = load_model(checkpoint_path, model, device)

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图像文件（支持常见格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
        and os.path.isfile(os.path.join(input_folder, f))
    ]

    if not image_files:
        print(f"错误：输入目录中没有找到支持的图像文件（支持的格式：{image_extensions}）")
        return

    print(f"找到 {len(image_files)} 张待处理图像，开始批量推理...")

    # 批量处理循环
    processed_count = 0
    for filename in image_files:
        try:
            # 构建完整路径
            image_path = os.path.join(input_folder, filename)
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')  # 确保RGB格式
            
            # 执行推理
            pred_labels, color_maps = infer(model, image, transform, device)
            
            # 生成保存路径
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(output_folder, f"{base_name}_result.png")
            
            # 保存结果
            display_segmentation_result(
                pred_labels, 
                color_maps,
                category_mapping,
                image,
                save_path=save_path
            )
            
            processed_count += 1
            print(f"已处理：{filename} -> {save_path}")

        except Exception as e:
            print(f"处理 {filename} 时发生错误：{str(e)}")
            continue

    print(f"处理完成！成功处理 {processed_count}/{len(image_files)} 张图像")

if __name__ == "__main__":
    main()


