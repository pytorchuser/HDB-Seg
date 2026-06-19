import torch
import cv2
import numpy as np
import timm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ===== 1. 创建模型（使用 timm 的 ImageNet 预训练权重） =====
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
model.eval()
if torch.cuda.is_available():
    model.cuda()

# ===== 2. 指定 Grad-CAM 目标层 =====
# timm 的 SwinTransformer 用 layers (Sequential) 存储 4 个 Stage
# Stage 4 输出 7×7 patches, 768 channels
target_layers = [model.layers[-1].blocks[-1].norm2]

# ===== 3. 定义 reshape_transform：将 3D [B, N, C] 转为 4D [B, C, H, W] =====
# Stage 4: N=49=7×7 patches
def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2).contiguous()
    return result

# ===== 4. 初始化 GradCAM =====
cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform,
)

# ===== 5. 加载图像 =====
image_path = 'C:/Users/cy/anaconda3/envs/swin_trans/mysegmentationpackage/data/Needle1/cropped/images/testing/1_29w_431.png'

img_bgr = cv2.imread(image_path, 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype(np.float32) / 255.0

# ===== 6. 预处理（timm swin_tiny 输入为 224×224） =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img_rgb).unsqueeze(0)

if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()

# ===== 7. 生成热力图 =====
# 分类模型用 targets=None，自动取 argmax 确定目标类别
grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=None
)

grayscale_cam = grayscale_cam[0, :]

# 打印热力图数值范围检查
print(f"热力图数值范围: min={grayscale_cam.min():.6f}, max={grayscale_cam.max():.6f}, mean={grayscale_cam.mean():.6f}")

# ===== 8. 插值到原图大小并叠加 =====
cam_resized = cv2.resize(grayscale_cam, (224, 224),
                          interpolation=cv2.INTER_LINEAR)

img_resized = cv2.resize(img_normalized, (224, 224))

visualization = show_cam_on_image(
    img_resized,
    cam_resized,
    use_rgb=True
)

# ===== 9. 保存结果 =====
import os
output_path = os.path.join(os.path.dirname(__file__), 'gradcam_result.png')
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"热力图已保存到 {output_path}")