"""
SIMSwinTransformerPlus 视网膜层分割 GradCAM 可视化
目标层: backbone.fam_layers[i].w_conv (FamLayer 内部的 3×3 ConvModule)
       相较于使用整个 fam_layers[i], w_conv 是双线性池化后的直接卷积层,
       梯度信号更干净, 热力图更精确。

所有关键参数均可通过下方配置区修改, 无需改动核心逻辑。
"""

import os
import sys

# ==================== 内存优化配置 ====================
# 缓解 CUDA 内存碎片化, 必须在 import torch 之前设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import gc
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmseg.apis import init_model
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM  # 可替换为 GradCAMPlusPlus / ScoreCAM / LayerCAM 等


# ====================================================================
#                       可配置参数 (按需修改)
# ====================================================================

# --- 模型配置 ---
# 配置文件路径 (相对于 mysegmentationpackage 目录)
CONFIG_PATH = './duke2015/my_upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
# 模型权重路径
CHECKPOINT_PATH = './duke2015/best_mDice_epoch_36.pth'

# --- 输入图像 ---
IMAGE_PATH = './example_figure/Subject_01_seg1_1.png'

# --- 输出配置 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'result', 'duke2015', 'LayerCAM_retina_results_w_conv')

# --- GradCAM 参数 ---
GradCAM_INPUT_SIZE = 512          # GradCAM 输入尺寸 (缩小以节省显存)
MASK_THRESHOLD = 0.5              # 预测掩码二值化阈值 (仅 USE_PRED_MASK=True 时生效)
USE_PRED_MASK = False             # True: 用预测掩码限制 GradCAM 区域; False: 全1掩码

# --- 预处理参数 ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --- 硬件配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = False                    # 是否使用 FP16 混合精度
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False     # 强制 CUDA 确定性计算; 让CUDA cuDNN 以确定性运行

# --- 目标层配置 ---
# fam_layers 索引: 0=Stage1(128²,96ch), 1=Stage2(64²,192ch), 2=Stage3(32²,384ch), 3=Stage4(16²,768ch)
# 深层语义强适合粗定位, 浅层空间细节多适合薄层边界
TARGET_LAYER_INDICES = [1, 2, 3]   # 可单选 [3] 或多选 [0,1,2,3]
TARGET_SUB_MODULE = 'w_conv'       # FamLayer 内部子模块: 'w_conv' | 'residual' | 'res_conv'

# --- 类别配置 ---
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
TARGET_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 9]  # 排除两个背景类

# --- GradCAM 方法 ---
# 可选: GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM, EigenCAM, EigenGradCAM, XGradCAM, HiResCAM, AblationCAM
GradCAM_METHOD = 'LayerCAM'

# ====================================================================


class SemanticSegmentationTarget:
    """语义分割 GradCAM 目标: 对指定类别的输出通道加权求和"""

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def print_gpu_memory(prefix=''):
    """打印当前 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'[GPU Memory {prefix}] allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB')


def get_gradcam_class(method_name):
    """根据名称获取 GradCAM 类"""
    import pytorch_grad_cam as pgc
    method_map = {
        'GradCAM': pgc.GradCAM,
        'GradCAMPlusPlus': pgc.GradCAMPlusPlus,
        'ScoreCAM': pgc.ScoreCAM,
        'LayerCAM': pgc.LayerCAM,
        'EigenCAM': pgc.EigenCAM,
        'EigenGradCAM': pgc.EigenGradCAM,
        'XGradCAM': pgc.XGradCAM,
        'HiResCAM': pgc.HiResCAM,
        'AblationCAM': pgc.AblationCAM,
    }
    if method_name not in method_map:
        raise ValueError(f"Unknown GradCAM method: {method_name}. "
                         f"Available: {list(method_map.keys())}")
    return method_map[method_name]


def get_target_layer(model, layer_idx, sub_module_name):
    """获取目标层: model.backbone.fam_layers[layer_idx].<sub_module_name>

    Args:
        model: 完整的 mmseg 模型
        layer_idx: fam_layers 索引 (0-3)
        sub_module_name: FamLayer 内部子模块名称
            - 'w_conv':   ConvModule (3×3 conv), 双线性池化后的直接卷积层 [推荐]
            - 'residual': Residual 融合块
            - 'res_conv': FamLayer 入口 1×1 卷积 (对齐 ResNet 通道)
    Returns:
        target_layer (nn.Module)
    """
    fam_layer = model.backbone.fam_layers[layer_idx]
    sub_module = getattr(fam_layer, sub_module_name)
    return sub_module


def get_target_layer_display_name(layer_idx, sub_module_name):
    """生成人类可读的目标层名称, 用于日志和输出文件名"""
    stage_name = f'Stage{layer_idx + 1}'
    resolutions = {0: 128, 1: 64, 2: 32, 3: 16}
    channels = {0: 96, 1: 192, 2: 384, 3: 768}
    res = resolutions.get(layer_idx, '?')
    ch = channels.get(layer_idx, '?')
    sub_display = {
        'w_conv': 'WConv',
        'residual': 'Residual',
        'res_conv': 'ResConv',
    }.get(sub_module_name, sub_module_name)
    return f'{stage_name}_{sub_display}_{res}x{res}_{ch}ch'


def main():
    # ---- 打印配置 ----
    print('=' * 70)
    print('  SIMSwinTransformerPlus GradCAM — w_conv 目标层')
    print('=' * 70)
    print(f'  Config:        {CONFIG_PATH}')
    print(f'  Checkpoint:    {CHECKPOINT_PATH}')
    print(f'  Image:         {IMAGE_PATH}')
    print(f'  Output:        {OUTPUT_DIR}')
    print(f'  Device:        {DEVICE}')
    print(f'  Input size:    {GradCAM_INPUT_SIZE}')
    print(f'  Target layers: fam_layers[{TARGET_LAYER_INDICES}].{TARGET_SUB_MODULE}')
    print(f'  GradCAM method:{GradCAM_METHOD}')
    print(f'  Categories:    {[CLASS_NAMES[c] for c in TARGET_CATEGORIES]}')
    print(f'  Use pred mask: {USE_PRED_MASK}')
    print(f'  AMP:           {USE_AMP}')
    print('=' * 70)

    # ---- 获取 GradCAM 类 ----
    GradCAMClass = get_gradcam_class(GradCAM_METHOD)

    device = torch.device(DEVICE)

    if torch.cuda.is_available():
        print_gpu_memory('before loading model')

    # ---- 1. 加载模型 ----
    model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device=device)
    model.eval()
    print('Model loaded successfully')
    if torch.cuda.is_available():
        print_gpu_memory('after loading model')

    # ---- 2. 读取并预处理图像 ----
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {IMAGE_PATH}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    original_rgb = np.float32(image_rgb) / 255
    # 预计算 uint8 BGR 原图 (在 GradCAM 循环前, 避免后续内存碎片化导致分配失败)
    original_bgr_uint8 = image_bgr.copy()  # 直接保存 BGR 原图

    # 缩小输入尺寸以节省显存
    image_resized = cv2.resize(image_rgb, (GradCAM_INPUT_SIZE, GradCAM_INPUT_SIZE))
    rgb_img = np.float32(image_resized) / 255
    input_tensor = preprocess_image(rgb_img, mean=MEAN, std=STD)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # ---- 3. 推理生成分割掩码 ----
    with torch.no_grad():
        output = model(input_tensor)
    logits = output['out'] if isinstance(output, dict) else output
    probs_all = F.softmax(logits, dim=1).cpu()[0].detach().numpy()
    pred_mask = np.argmax(probs_all, axis=0)
    print(f'Predicted mask shape: {pred_mask.shape}, unique classes: {np.unique(pred_mask)}')

    del output, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('after inference + cache cleared')

    # ---- 4. 准备掩码 ----
    h_model, w_model = pred_mask.shape[:2]
    if USE_PRED_MASK:
        print('Using prediction-based masks for GradCAM targets')
    else:
        print('Using all-ones masks for GradCAM targets')

    def make_mask(cat_idx):
        """为指定类别创建掩码"""
        if USE_PRED_MASK:
            mask = np.float32(pred_mask == cat_idx)
            if mask.sum() < 10:  # 预测区域太小则回退到全1掩码
                print(f'  Warning: class {cat_idx} has <10 pixels in pred, falling back to all-ones')
                mask = np.ones((h_model, w_model), dtype=np.float32)
        else:
            mask = np.ones((h_model, w_model), dtype=np.float32)
        return mask

    # ---- 5. 创建输出目录 ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 6. 对每个目标层 + 每个类别生成热力图 ----
    total_images = 0

    for layer_idx in TARGET_LAYER_INDICES:
        target_layer = get_target_layer(model, layer_idx, TARGET_SUB_MODULE)
        layer_display_name = get_target_layer_display_name(layer_idx, TARGET_SUB_MODULE)
        target_layers = [target_layer]

        print(f'\n{"=" * 50}')
        print(f'  Layer {layer_idx}: {layer_display_name}')
        print(f'  Module: {TARGET_SUB_MODULE}')
        print(f'{"=" * 50}')

        for cat_idx in TARGET_CATEGORIES:
            class_name = CLASS_NAMES[cat_idx]
            print(f'  [{layer_display_name}] class {cat_idx}: {class_name} ...', end=' ', flush=True)

            mask = make_mask(cat_idx)
            targets = [SemanticSegmentationTarget(cat_idx, mask)]

            try:
                with GradCAMClass(model=model, target_layers=target_layers) as cam:
                    if USE_AMP and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                    else:
                        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            except Exception as e:
                print(f'FAILED: {e}')
                del targets, mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            print(f'OK (range: {grayscale_cam.min():.4f} ~ {grayscale_cam.max():.4f})')

            # resize 到原始尺寸并叠加到原图
            grayscale_cam = cv2.resize(grayscale_cam, (original_w, original_h))
            cam_image = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)

            # 保存: 文件名包含层名和类别名
            filename = f'{layer_display_name}_{class_name}.png'
            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            total_images += 1

            del targets, mask, grayscale_cam, cam_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- 7. 保存原图和分割结果 ----
    # 强制 GC 回收 GradCAM 循环中的碎片内存
    gc.collect()

    # 直接用预存的 BGR uint8 原图, 避免 float32 乘法分配
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'original_image.png'), original_bgr_uint8)

    # 分割结果可视化 (用 uint8 直接构造, 避免 float32 中间量)
    seg_vis_uint8 = np.zeros_like(image_rgb)  # uint8, 无碎片风险
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
        m = (pred_mask == cat_idx)
        if m.any():
            m_resized = cv2.resize(m.astype(np.uint8), (original_w, original_h),
                                   interpolation=cv2.INTER_NEAREST)
            color = layer_colors[i % len(layer_colors)]
            for c in range(3):
                seg_vis_uint8[:, :, c] = np.where(m_resized > 0, color[c], seg_vis_uint8[:, :, c])
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'segmentation_overlay.png'),
                cv2.cvtColor(seg_vis_uint8, cv2.COLOR_RGB2BGR))

    # ---- 8. 打印汇总 ----
    print(f'\n{"=" * 70}')
    print(f'  Done!')
    print(f'  Output directory: {OUTPUT_DIR}')
    print(f'  Total images:     {total_images} '
          f'({len(TARGET_LAYER_INDICES)} layers × {len(TARGET_CATEGORIES)} classes)')
    print(f'  Files:')
    print(f'    - original_image.png')
    print(f'    - segmentation_overlay.png')
    for layer_idx in TARGET_LAYER_INDICES:
        name = get_target_layer_display_name(layer_idx, TARGET_SUB_MODULE)
        for cat_idx in TARGET_CATEGORIES:
            print(f'    - {name}_{CLASS_NAMES[cat_idx]}.png')
    print(f'{"=" * 70}')

    # 释放
    del model, input_tensor, pred_mask, probs_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('final cleanup')


if __name__ == '__main__':
    main()
