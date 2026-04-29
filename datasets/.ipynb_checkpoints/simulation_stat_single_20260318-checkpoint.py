import numpy as np
import os

def analyze_npy(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'。请检查路径是否正确。")
        return
        
    try:
        data = np.load(file_path)
        print(f"文件路径: {file_path}")
        print("-" * 40)
        print(f"数据维度 (Shape): {data.shape}")
        print(f"数据类型 (Dtype): {data.dtype}")
        print(f"最大值 (Max):     {data.max():.6f}")
        print(f"最小值 (Min):     {data.min():.6f}")
        print(f"平均值 (Mean):    {data.mean():.6f}")
        print("-" * 40)
    except Exception as e:
        print(f"读取或分析文件时发生错误：{e}")

if __name__ == "__main__":
    # 请在此处修改为你想要分析的 .npy 文件路径
    FILE_PATH = '/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/AIOptics/data/4f_twophases_2pi_experiment/eval/pat/pat_18001.npy' 
    
    analyze_npy(FILE_PATH)