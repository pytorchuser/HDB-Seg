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
CONFIG_PATH = '../../configs/swin/my_upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
# 模型权重路径
CHECKPOINT_PATH = '../../pth/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'
# OCT 测试图像路径
IMAGE_PATH = '../../data/Needle1/cropped/images/testing/1_29w_431.png'
# 结果输出文件名（保存在本脚本同目录下）
OUTPUT_FILENAME = 'sim_swin_plus_gradcam_result.png'
# 目标类别索引（0=背景，1=needle）
TARGET_CATEGORY = 1
# GradCAM 输入尺寸（减小以节省显存，如 256）
GradCAM_INPUT_SIZE = 256
# 掩码二值化阈值
MASK_THRESHOLD = 0.5
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if torch.cuda.is_available():
        print_gpu_memory('before loading model')

    # 1. 加载模型（使用 mmseg.apis.init_model，自动处理注册和权重加载）
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

    # 3. 缩小输入尺寸以节省显存（方案 A）
    image_resized = cv2.resize(image_rgb, (GradCAM_INPUT_SIZE, GradCAM_INPUT_SIZE))
    rgb_img = np.float32(image_resized) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # 4. 推理生成 needle 掩码（使用 torch.no_grad() 节省显存，因为此步不需要梯度）
    with torch.no_grad():
        output = model(input_tensor)
    logits = output['out'] if isinstance(output, dict) else output
    probs = F.softmax(logits, dim=1).cpu()[0, TARGET_CATEGORY, :, :].detach().numpy()
    mask_binary = np.float32(probs > MASK_THRESHOLD)
    print(f'Needle prob range: min={probs.min():.4f}, max={probs.max():.4f}')

    # 释放不再需要的中间变量，缓解显存压力
    del output, logits, probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('after inference + cache cleared')

    # 5. GradCAM（需要梯度，不能使用 no_grad；使用混合精度 FP16 节省显存）
    target_layers = [model.backbone.fam_layers[-1]]
    targets = [SemanticSegmentationTarget(TARGET_CATEGORY, mask_binary)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        with torch.cuda.amp.autocast():
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    print(f'Heatmap range: min={grayscale_cam.min():.6f}, max={grayscale_cam.max():.6f}')

    # 6. 保存结果（将 heatmap 放大回原始尺寸再叠加到原图上）
    grayscale_cam = cv2.resize(grayscale_cam, (original_w, original_h))
    cam_image = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME)
    cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f'Result saved to {output_path}')

    # 释放显存
    del model, input_tensor, grayscale_cam, cam_image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('final cleanup')


if __name__ == '__main__':
    main()
