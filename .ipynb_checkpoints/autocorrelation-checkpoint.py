import torch
import numpy as np
import torch
import torch.fft as fft
import cv2
import numpy as np
from PIL import Image



def compute_speckle_ac(img, save_path="speckle_ac.png"):
    # 1. 读取图像并转换为张量 [1, 1, H, W]
    # 遥感或物理成像建议读入原始位数，这里以灰度图为例
    
    # 转换为 float32 并搬到 GPU (如果有)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(img).float().to(device)
    
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    print(x.shape)
    
    x = (x / 40.0).clamp_(0.0, 1.0)
    
    # 2. 计算自相关 (Autocorrelation)
    # 根据公式: AC = IFFT(|FFT(I)|^2)
    # 步骤 A: 二维傅里叶变换
    X_fft = fft.fft2(x)
    
    # 步骤 B: 计算功率谱 (取模的平方)
    X_pow = torch.abs(X_fft) ** 2
    
    # 步骤 C: 逆傅里叶变换
    ac = fft.ifft2(X_pow)
    
    # 步骤 D: 取实部并进行 fftshift，让自相关峰值居中
    ac = torch.real(ac)
    ac = fft.fftshift(ac)
    
    # 3. 结果后处理与保存
    # 线性归一化到 0-255 用于保存图片
    ac_min = ac.min()
    ac_max = ac.max()
    ac_norm = (ac - ac_min) / (ac_max - ac_min + 1e-8) * 255.0
    ac_np = ac_norm.cpu().numpy().astype(np.uint8)
    
    ac_np = ac_np[0, 0]
    
    # 保存结果
    cv2.imwrite(save_path, ac_np)
    print(f"AC result saved to {save_path}")
    
    return ac_np

# 使用示例
# compute_speckle_ac("your_speckle_image.png")

pat = np.load('data/4f_twophases_pi/eval/pat/pat_19000.npy')

ac_np = compute_speckle_ac(pat)