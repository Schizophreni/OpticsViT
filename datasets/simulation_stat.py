import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def fast_analyze():
    pat1 = '../data/4f_random_perlin_20k/train/pat'
    pat2 = '../data/4f_random_perlin_20k/eval/pat'
    imgs = glob.glob(os.path.join(pat1, "*.npy")) + glob.glob(os.path.join(pat2, "*.npy"))
    
    num_imgs = len(imgs)
    if num_imgs == 0: return
    
    # 预读一张图获取尺寸
    sample = np.load(imgs[0])
    h, w = sample.shape[-2:], sample.shape
    
    # 用于计算 mean 和 std 的累加器 (使用 float64 防止溢出)
    sum_x = np.zeros(h, dtype=np.float64)
    sum_x2 = np.zeros(h, dtype=np.float64)
    intens = []
    
    print(f"Analyzing {num_imgs} images...")

    # 使用线程池并行读取磁盘，CPU 负责累加
    # 这里的 max_workers 取决于你的磁盘 IOPS，SSD 建议 8-16
    with ThreadPoolExecutor(max_workers=12) as executor:
        # 使用 map 保持顺序或使用 tqdm 配合 submit
        results = list(tqdm(executor.map(np.load, imgs), total=num_imgs))

    for img in tqdm(results, desc="Calculating stats"):
        # 确保是 2D 形状 (h, w)
        data = img.squeeze() 
        
        # 统计 Intensity
        intens.append(data.max())
        
        # 增量累加用于均值和标准差
        sum_x += data
        sum_x2 += data ** 2

    # 计算统计量
    intens = np.array(intens)
    max_inten = intens.max()
    min_inten = intens.min()

    # 计算 Mean: E[X]
    mean_img = sum_x / num_imgs
    # 计算 Std: sqrt(E[X^2] - (E[X])^2)
    # 增加 epsilon 防止数值不稳定导致的负值
    var_img = (sum_x2 / num_imgs) - (mean_img ** 2)
    std_img = np.sqrt(np.maximum(var_img, 1e-10))

    # 归一化并保存
    mean_final = mean_img / (mean_img.max() + 1e-10)
    std_final = std_img / (std_img.max() + 1e-10)

    torchvision.utils.save_image(torch.from_numpy(std_final)[None], "std_75.png")
    torchvision.utils.save_image(torch.from_numpy(mean_final)[None], "mean_75.png")

    print(f"Max intensity: {max_inten}")
    print(f"Min intensity: {min_inten}")

    # 绘制直方图 (保持原样)
    plt.figure(figsize=(10, 6))
    plt.hist(intens, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Data Distribution (n={num_imgs})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('histogram_75.png', dpi=300)

if __name__ == "__main__":
    fast_analyze()