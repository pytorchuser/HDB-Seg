import os

# ==================== 内存优化配置 ====================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from mmseg.apis import init_model
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

# ==================== 可配置参数 ====================
CONFIG_PATH = '../output/train/T_88_3lr_swin_dice3&ce1_0.5bg_LS10_hcms2018crop512_epoch50_1x/my_upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
CHECKPOINT_PATH = '../../pth/best_mDice_epoch_27.pth'
IMAGE_PATH = './example_figure/Subject_01_seg1_1.png'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'retina_results_multi')
GradCAM_INPUT_SIZE = 512

CLASS_NAMES = [
    'background1',   # 0
    'RNFL',          # 1
    'GCIP',          # 2
    'INL',           # 3
    'OPL',           # 4
    'ONL',           # 5
    'IS',            # 6
    'OS-RPE',        # 7
    'background2',   # 8
    'Fluid',         # 9
]
TARGET_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 9]

# 三个推荐的目标层索引 (对应 Stage 2, 3, 4)
TARGET_LAYER_NAMES = ['Stage2', 'Stage3', 'Stage4']
TARGET_LAYER_INDICES = [1, 2, 3]
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
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'[GPU Memory {prefix}] allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB')
        torch.cuda.synchronize()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print_gpu_memory('before loading model')

    # 1. 加载模型
    model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device=device)
    model.eval()
    print('Model loaded successfully')
    print_gpu_memory('after loading model')

    # 2. 读取图像
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {IMAGE_PATH}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    original_rgb = np.float32(image_rgb) / 255

    # 3. 预处理输入
    image_resized = cv2.resize(image_rgb, (GradCAM_INPUT_SIZE, GradCAM_INPUT_SIZE))
    rgb_img = np.float32(image_resized) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # 4. 推理生成分割掩码（仅用于信息输出，不做 GradCAM 限制）
    with torch.no_grad():
        output = model(input_tensor)
    logits = output['out'] if isinstance(output, dict) else output
    probs_all = F.softmax(logits, dim=1).cpu()[0].detach().numpy()
    pred_mask = np.argmax(probs_all, axis=0)
    print(f'Predicted classes: {np.unique(pred_mask)}')
    del output, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('after inference')

    # 获取模型输出分辨率
    h_model, w_model = pred_mask.shape[:2]
    all_ones_mask_np = np.ones((h_model, w_model), dtype=np.float32)

    # 5. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 6. 对每个目标层 + 每个类别生成 GradCAM
    for layer_idx, layer_name in zip(TARGET_LAYER_INDICES, TARGET_LAYER_NAMES):
        target_layers = [model.backbone.fam_layers[layer_idx]]
        print(f'\n========== Processing {layer_name} (fam_layers[{layer_idx}]) ==========')

        for cat_idx in TARGET_CATEGORIES:
            class_name = CLASS_NAMES[cat_idx]
            print(f'  [{layer_name}] class {cat_idx}: {class_name} ...', end=' ')

            # GradCAM target (全 1 掩码)
            targets = [SemanticSegmentationTarget(cat_idx, all_ones_mask_np)]

            try:
                with GradCAM(model=model, target_layers=target_layers) as cam:
                    with torch.cuda.amp.autocast():
                        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            except Exception as e:
                print(f'FAILED: {e}')
                del targets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # resize 到原始尺寸并叠加到原图
            grayscale_cam = cv2.resize(grayscale_cam, (original_w, original_h))
            cam_image = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)

            # 保存
            filename = f'{layer_name}_{class_name}.png'
            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            print(f'saved')

            # 清理
            del targets, grayscale_cam, cam_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f'\n========== All done ==========')
    print(f'Results saved to: {OUTPUT_DIR}')
    print(f'Total: {len(TARGET_LAYER_INDICES)} layers x {len(TARGET_CATEGORIES)} classes = {len(TARGET_LAYER_INDICES) * len(TARGET_CATEGORIES)} images')

    # 释放模型
    del model, input_tensor, pred_mask, probs_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory('final cleanup')


if __name__ == '__main__':
    main()
