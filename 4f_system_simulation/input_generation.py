import numpy as np
import cv2
import os
import scipy.ndimage as ndimage
from tqdm import tqdm

# =====================================================================
# 核心生成模块 (无硬编码，由外部参数驱动)
# =====================================================================

def generate_solid_psf_probes(save_dir, size, block_sizes, stride, intensities):
    """生成分立亮度的纯色空间点阵探针"""
    count = 0
    # 先计算总数用于进度条
    total_iters = sum([len(range(0, size - bs + 1, stride)) * len(range(0, size - bs + 1, stride)) for bs in block_sizes]) * len(intensities)
    
    with tqdm(total=total_iters, desc="[1/6] Solid PSF Probes", unit="img") as pbar:
        for bs in block_sizes:
            for intensity in intensities:
                for y in range(0, size - bs + 1, stride):
                    for x in range(0, size - bs + 1, stride):
                        img = np.zeros((size, size), dtype=np.uint8)
                        img[y:y+bs, x:x+bs] = intensity  # 放置纯色方块点源
                        
                        filename = f"psf_solid_bs{bs}_int{intensity}_{count:05d}.png"
                        cv2.imwrite(os.path.join(save_dir, filename), img)
                        count += 1
                        pbar.update(1)
    return count

def generate_noisy_psf_probes(save_dir, size, block_sizes, stride, num_variations):
    """生成带有局域噪声分布的空间点阵探针"""
    count = 0
    total_iters = sum([len(range(0, size - bs + 1, stride)) * len(range(0, size - bs + 1, stride)) for bs in block_sizes]) * num_variations
    
    with tqdm(total=total_iters, desc="[2/6] Noisy PSF Probes", unit="img") as pbar:
        for bs in block_sizes:
            for y in range(0, size - bs + 1, stride):
                for x in range(0, size - bs + 1, stride):
                    for v in range(num_variations):
                        img = np.zeros((size, size), dtype=np.uint8)
                        
                        # 生成局域的随机噪声方块 (0-255)，每次循环由于np.random的状态不同，生成的必定不同
                        noise_block = np.random.randint(0, 256, size=(bs, bs), dtype=np.uint8)
                        img[y:y+bs, x:x+bs] = noise_block
                        
                        filename = f"psf_noisy_bs{bs}_v{v}_{count:05d}.png"
                        cv2.imwrite(os.path.join(save_dir, filename), img)
                        count += 1
                        pbar.update(1)
    return count

def generate_gratings(save_dir, size, angles, periods, phases):
    """生成正弦光栅测试集"""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    count = 0
    total_iters = len(angles) * len(periods) * len(phases)
    
    with tqdm(total=total_iters, desc="[3/6] Gratings        ", unit="img") as pbar:
        for angle in angles:
            theta = np.deg2rad(angle)
            for p in periods:
                k = 2 * np.pi / p
                for phase in phases:
                    # 核心公式: 1/2 * [1 + cos(kx*x + ky*y + phase)]
                    grating = 0.5 * (1 + np.cos(k * (x * np.cos(theta) + y * np.sin(theta)) - phase))
                    img = (grating * 255).astype(np.uint8)
                    
                    filename = f"grating_ang{angle}_p{p}_{count:05d}.png"
                    cv2.imwrite(os.path.join(save_dir, filename), img)
                    count += 1
                    pbar.update(1)
    return count

def generate_random_shapes(save_dir, size, num_images, min_shapes, max_shapes, min_intensity, max_intensity):
    """生成随机纯几何图元组合 (不含噪声)"""
    count = 0
    for i in tqdm(range(num_images), desc="[4/6] Geometric Shapes", unit="img"):
        img = np.zeros((size, size), dtype=np.uint8)
        num_shapes = np.random.randint(min_shapes, max_shapes + 1)
        
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rect', 'line'])
            intensity = int(np.random.randint(min_intensity, max_intensity + 1))
            
            if shape_type == 'circle':
                center = (np.random.randint(0, size), np.random.randint(0, size))
                radius = np.random.randint(2, max(3, size // 3))
                thickness = -1 if np.random.rand() > 0.5 else np.random.randint(1, 4)
                cv2.circle(img, center, radius, intensity, thickness)
                
            elif shape_type == 'rect':
                pt1 = (np.random.randint(0, size), np.random.randint(0, size))
                pt2 = (np.random.randint(0, size), np.random.randint(0, size))
                thickness = -1 if np.random.rand() > 0.5 else np.random.randint(1, 4)
                cv2.rectangle(img, pt1, pt2, intensity, thickness)
                
            elif shape_type == 'line':
                pt1 = (np.random.randint(0, size), np.random.randint(0, size))
                pt2 = (np.random.randint(0, size), np.random.randint(0, size))
                thickness = np.random.randint(1, 5)
                cv2.line(img, pt1, pt2, intensity, thickness)
                
        filename = f"shape_{count:05d}.png"
        cv2.imwrite(os.path.join(save_dir, filename), img)
        count += 1
    return count

def generate_pure_continuous_noise(save_dir, size, num_images, min_sigma, max_sigma):
    """生成纯粹的连续带限分形噪声"""
    count = 0
    for i in tqdm(range(num_images), desc="[5/6] Continuous Noise", unit="img"):
        raw_noise = np.random.rand(size, size).astype(np.float32)
        sigma = np.random.uniform(min_sigma, max_sigma)
        
        smoothed_noise = ndimage.gaussian_filter(raw_noise, sigma=sigma)
        
        # 重新归一化以拉伸对比度
        n_min = smoothed_noise.min()
        n_max = smoothed_noise.max()
        if n_max > n_min:
            normalized_noise = (smoothed_noise - n_min) / (n_max - n_min)
        else:
            normalized_noise = smoothed_noise
            
        img = (normalized_noise * 255).astype(np.uint8)
        
        filename = f"noise_cont_{count:05d}.png"
        cv2.imwrite(os.path.join(save_dir, filename), img)
        count += 1
    return count

def generate_standard_random_noise(save_dir, size, num_images, blur_sigma):
    """生成经典的全局随机噪声 (支持高斯平滑)"""
    count = 0
    for i in tqdm(range(num_images), desc="[6/6] Std Random Noise", unit="img"):
        raw_data = np.random.rand(size, size).astype(np.float32)
        
        if blur_sigma > 0:
            ksize = np.random.choice([3,5,7])
            if ksize % 2 == 0: ksize += 1
            raw_data = cv2.GaussianBlur(raw_data, (ksize, ksize), blur_sigma)
            
            # 防止平滑后对比度降低
            if raw_data.max() > raw_data.min():
                raw_data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())
                
        img = (raw_data * 255).astype(np.uint8)
        
        filename = f"noise_std_sigma{blur_sigma}_{count:05d}.png"
        cv2.imwrite(os.path.join(save_dir, filename), img)
        count += 1
    return count

# =====================================================================
# 主程序：所有可变参数集中配置区
# =====================================================================
if __name__ == "__main__":
    
    # --- 1. 全局基础配置 ---
    IMAGE_SIZE = 50
    OUTPUT_DIR_NAME = "input_random_20k_sigma_0_5_k_5"
    RANDOM_SEED = 42

    # --- 2. 探针参数配置 (PSF Probes) ---
    PSF_BLOCK_SIZES = [4, 6, 8, 10, 12]           
    PSF_STRIDE = 2                                
    PSF_INTENSITIES = [192, 255]                  # [核心产出: 4,860 张] 抛弃低信噪比，专注高强度扫描
    PSF_NOISY_VARIATIONS = 5                      # [核心产出: 12,150 张] 提升为 5 种，强化局部高频扰动的解耦能力

    # --- 3. 光栅参数配置 (Gratings) ---
    GRATING_ANGLES = [0, 30, 60, 90, 120, 150]    
    GRATING_PERIODS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32] 
    GRATING_PHASES = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
                                                  # [核心产出: 480 张]

    # --- 4. 几何图元参数配置 (Random Shapes) ---
    SHAPES_NUM_IMAGES = 0                      # [核心产出: 6,000 张] 大幅提升，帮助网络建立边缘衍射直觉
    SHAPES_MIN_PER_IMG = 1                        
    SHAPES_MAX_PER_IMG = 6                        
    SHAPES_MIN_INTENSITY = 50                     
    SHAPES_MAX_INTENSITY = 255                    

    # --- 5. 连续噪声参数配置 (Continuous Noise) ---
    NOISE_CONT_NUM_IMAGES = 0                  # [核心产出: 5,510 张] 大幅提升，提供极其丰富的低频相干背景
    NOISE_CONT_MIN_SIGMA = 1.5                    
    NOISE_CONT_MAX_SIGMA = 5.0                    

    # --- 6. 经典随机噪声参数配置 (Standard Random Noise) ---
    NOISE_STD_NUM_IMAGES = 20000                   # [核心产出: 1,000 张] 缩减比例，避免网络浪费算力记忆纯白噪声
    NOISE_STD_BLUR_SIGMA = 0.5

    # --- 执行生成逻辑 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(current_dir, OUTPUT_DIR_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.random.seed(RANDOM_SEED)
    
    print(f"Target Directory: {OUTPUT_DIR}")
    print("Starting comprehensive dataset generation...\n")

    total_generated = 0
    
    # [1] 纯色分立探针
#     c_psf_solid = generate_solid_psf_probes(
#         save_dir=OUTPUT_DIR, size=IMAGE_SIZE, 
#         block_sizes=PSF_BLOCK_SIZES, stride=PSF_STRIDE, intensities=PSF_INTENSITIES
#     )
#     total_generated += c_psf_solid
    
    # [2] 局域噪声探针
#     c_psf_noisy = generate_noisy_psf_probes(
#         save_dir=OUTPUT_DIR, size=IMAGE_SIZE, 
#         block_sizes=PSF_BLOCK_SIZES, stride=PSF_STRIDE, num_variations=PSF_NOISY_VARIATIONS
#     )
#     total_generated += c_psf_noisy
    
    # [3] 正弦光栅
    # c_grat = generate_gratings(
    #     save_dir=OUTPUT_DIR, size=IMAGE_SIZE, 
    #     angles=GRATING_ANGLES, periods=GRATING_PERIODS, phases=GRATING_PHASES
    # )
    # total_generated += c_grat
    
    # [4] 几何图形
    # c_shape = generate_random_shapes(
    #     save_dir=OUTPUT_DIR, size=IMAGE_SIZE, num_images=SHAPES_NUM_IMAGES,
    #     min_shapes=SHAPES_MIN_PER_IMG, max_shapes=SHAPES_MAX_PER_IMG,
    #     min_intensity=SHAPES_MIN_INTENSITY, max_intensity=SHAPES_MAX_INTENSITY
    # )
    # total_generated += c_shape
    
    # [5] 连续分形噪声
    # c_noise_cont = generate_pure_continuous_noise(
    #     save_dir=OUTPUT_DIR, size=IMAGE_SIZE, num_images=NOISE_CONT_NUM_IMAGES,
    #     min_sigma=NOISE_CONT_MIN_SIGMA, max_sigma=NOISE_CONT_MAX_SIGMA
    # )
    # total_generated += c_noise_cont
    
    # [6] 经典全局随机噪声
    c_noise_std = generate_standard_random_noise(
        save_dir=OUTPUT_DIR, size=IMAGE_SIZE, num_images=NOISE_STD_NUM_IMAGES,
        blur_sigma=NOISE_STD_BLUR_SIGMA
    )
    total_generated += c_noise_std
    
    print("\n" + "="*45)
    print(f"[Success] Dataset Generation Complete!")
    # print(f" - Solid PSF Probes:   {c_psf_solid:4d} images")
    # print(f" - Noisy PSF Probes:   {c_psf_noisy:4d} images")
    # print(f" - Gratings:           {c_grat:4d} images")
    # print(f" - Geometric Shapes:   {c_shape:4d} images")
    # print(f" - Continuous Noise:   {c_noise_cont:4d} images")
    print(f" - Standard Noise:     {c_noise_std:4d} images")
    print(f" - TOTAL:              {total_generated:4d} images")
    print("="*45)