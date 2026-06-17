from torchinfo import summary
# from mmseg.models.backbones import SIMSwinTransformerPlus
from mmseg.models.backbones import SwinTransformer

# 假设你的 OCT 分割模型
# model = SIMSwinTransformerPlus()
model = SwinTransformer()

# OCT B-scan 通常是单通道灰度图
summary(
    model,
    input_size=(1, 3, 512, 512),       # (batch, channels, height, width)
    col_names=["input_size", "output_size", "num_params", "params_percent"],
    depth=6,
    device='cuda'
)