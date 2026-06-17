import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from mmseg.models.backbones import SwinTransformer

# ===== 1. 创建模型（必须传参初始化） =====
model = SwinTransformer(
    pretrain_img_size=512,
    embed_dims=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=7,
    out_indices=(0, 1, 2, 3)
)
model.init_weights()

original_forward = model.forward
model.forward = lambda x: original_forward(x)[-1]
model.eval()
if torch.cuda.is_available():
    model.cuda()

# ===== 2. 指定 Grad-CAM 目标层 =====
# MMSegmentation 的 SwinTransformer 用 self.stages (ModuleList) 存储 4 个 Stage
target_layers = [model.stages[-1].blocks[-1].norm2]

# ===== 3. 初始化 GradCAM =====
cam = GradCAM(
    model=model,
    target_layers=target_layers,
)

# ===== 4. 加载图像 =====
image_path = 'C:/Users/cy/anaconda3/envs/swin_trans/mysegmentationpackage/data/Needle1/cropped/images/testing/1_29w_431.png'

img_bgr = cv2.imread(image_path, 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype(np.float32) / 255.0

# ===== 5. 预处理 =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img_rgb).unsqueeze(0)

if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()

# ===== 6. 生成热力图 =====
grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=[lambda t: t.sum()]
)

grayscale_cam = grayscale_cam[0, :]

# ===== 7. 插值到原图大小并叠加 =====
cam_resized = cv2.resize(grayscale_cam, (512, 512),
                          interpolation=cv2.INTER_LINEAR)

img_resized = cv2.resize(img_normalized, (512, 512))

visualization = show_cam_on_image(
    img_resized,
    cam_resized,
    use_rgb=True
)

# ===== 8. 保存结果 =====
cv2.imwrite('gradcam_result.png',
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print("热力图已保存到 gradcam_result.png")

