import os
import sys

# ==================== 内存优化配置 ====================
# 缓解 CUDA 内存碎片化，必须在 import torch 之前设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmseg.apis import init_model
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

# ==================== 可配置参数 ====================
# 配置文件路径（相对于项目根目录 mysegmentationpackage）
CONFIG_PATH = '../output/train/T_88_3lr_swin_dice3&ce1_0.5bg_LS10_hcms2018crop512_epoch50_1x/my_upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
# 模型权重路径
CHECKPOINT_PATH = '../../pth/best_mDice_epoch_27.pth'
# OCT 测试图像路径
IMAGE_PATH = './example_figure/Subject_01_seg1_1.png'
# 结果输出目录（保存在本脚本同目录下）
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'retina_results')
# GradCAM 输入尺寸（减小以节省显存）
GradCAM_INPUT_SIZE = 512
# 掩码二值化阈值
MASK_THRESHOLD = 0.5

# 视网膜层类别名称（10 类, 来自 OCTDuke2015Dataset）
CLASS_NAMES = [
    'background1',   # 0
    'RNFL',          # 1 - 视网膜神经纤维层
    'GCIP',          # 2 - 节细胞层+内丛状层
    'INL',           # 3 - 内核层
    'OPL',           # 4 - 外丛状层
    'ONL',           # 5 - 外核层
    'IS',            # 6 - 感光细胞内节
    'OS-RPE',        # 7 - 感光细胞外节+色素上皮
    'background2',   # 8
    'Fluid',         # 9 - 积液
]
# 需要生成热力图的类别索引（排除两个背景类）
TARGET_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 9]
# ==================================================


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def print_gpu_memory(prefix=''):
    """打印当前 GPU 显存使用情况（仅用于调试）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'[GPU Memory {prefix}] allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB')
        torch.cuda.synchronize()


def overlay_mask_on_image(image_rgb, mask, color=(0, 255, 0), alpha=0.4):
    """将二值掩码以半透明颜色叠加到图像上"""
    overlay = image_rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.float32)
    return overlay


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if torch.cuda.is_available():
        print_gpu_memory('before loading model')

    # 1. 加载模型
    model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device=device)
    model.eval()
    print('Model loaded successfully')
    if torch.cuda.is_available():
        print_gpu_memory('after loading model')

    # 2. 读取图像（保留原始尺寸用于最终可视化叠加）
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {IMAGE_PATH}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    original_rgb = np.float32(image_rgb) / 255

    # 3. 缩小输入尺寸以节省显存
    image_resized = cv2.resize(image_rgb, (GradCAM_INPUT_SIZE, GradCAM_INPUT_SIZE))
    rgb_img = np.float32(image_resized) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # 4. 推理生成分割掩码（使用 torch.no_grad() 节省显存）
    with torch.no_grad():
        output = model(input_tensor)
    logits = output['out'] if isinstance(output, dict) else output
    probs_all = F.softmax(logits, dim=1).cpu()[0].detach().numpy()  # [C, H, W]
    pred_mask = np.argmax(probs_all, axis=0)  # [H, W]
    print(f'Predicted mask shape: {pred_mask.shape}, unique classes: {np.unique(pred_mask)}')

    # 释放中间变量
    del output, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('after inference + cache cleared')

    # 5. 为每个目标类别生成 GradCAM 热力图
    # 使用全 1 掩码（而非预测掩码），确保对所有类别（包括模型未 argmax 的类）都生成热力图
    target_layers = [model.backbone.fam_layers[-1]]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取模型输出分辨率，用于创建全 1 掩码
    h_model, w_model = pred_mask.shape[:2]
    all_ones_mask = np.ones((h_model, w_model), dtype=np.float32)

    for cat_idx in TARGET_CATEGORIES:
        class_name = CLASS_NAMES[cat_idx]
        print(f'\n--- Processing class {cat_idx}: {class_name} ---')

        # 创建 GradCAM target（全 1 掩码 = 对整个类别通道做梯度）
        targets = [SemanticSegmentationTarget(cat_idx, all_ones_mask)]

        # GradCAM 前向传播（FP16 混合精度节省显存）
        with GradCAM(model=model, target_layers=target_layers) as cam:
            with torch.cuda.amp.autocast():
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        print(f'  Heatmap range: min={grayscale_cam.min():.6f}, max={grayscale_cam.max():.6f}')

        # 将 heatmap 放大回原始尺寸
        grayscale_cam = cv2.resize(grayscale_cam, (original_w, original_h))

        # 生成叠加图并保存
        cam_image = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)
        output_path = os.path.join(OUTPUT_DIR, f'gradcam_{class_name}.png')
        cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        print(f'  Saved: {output_path}')

        # 每处理一个类别清理一次显存
        del targets, grayscale_cam, cam_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. 保存合并对比图
    print('\n--- Saving combined visualization ---')

    # 创建合并图（2行 x 4列）
    num_cats = len(TARGET_CATEGORIES)
    cols = 4
    rows = (num_cats + cols - 1) // cols + 1  # 加一行用于原图+分割结果
    fig_height = rows * 300
    fig_width = cols * 300

    # 检查是否有 pred_mask 信息，创建分割结果可视化图
    seg_vis = np.zeros_like(original_rgb)
    # 用不同颜色绘制各层分割结果
    layer_colors = [
        (255, 0, 0),     # RNFL - 红
        (0, 255, 0),     # GCIP - 绿
        (0, 0, 255),     # INL - 蓝
        (255, 255, 0),   # OPL - 青
        (255, 0, 255),   # ONL - 紫
        (0, 255, 255),   # IS - 黄
        (128, 0, 128),   # OS-RPE - 深紫
        (255, 128, 0),   # Fluid - 橙
    ]
    for i, cat_idx in enumerate(TARGET_CATEGORIES):
        mask = (pred_mask == cat_idx)
        if mask.any():
            # 调整尺寸到原始图像大小
            mask_resized = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            color = layer_colors[i % len(layer_colors)]
            for c in range(3):
                seg_vis[:, :, c] = np.where(mask_resized > 0, color[c] / 255.0, seg_vis[:, :, c])

    # 保存原图对比
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'original_image.png'),
                cv2.cvtColor(np.uint8(original_rgb * 255), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'segmentation_overlay.png'),
                cv2.cvtColor(np.uint8(seg_vis * 255), cv2.COLOR_RGB2BGR))

    print(f'\nAll results saved to: {OUTPUT_DIR}')
    print(f'  - original_image.png (original input)')
    print(f'  - segmentation_overlay.png (predicted segmentation)')
    for cat_idx in TARGET_CATEGORIES:
        print(f'  - gradcam_{CLASS_NAMES[cat_idx]}.png (GradCAM heatmap)')

    # 释放显存
    del model, input_tensor, pred_mask, probs_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('final cleanup')


if __name__ == '__main__':
    main()
