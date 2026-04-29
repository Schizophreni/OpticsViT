import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_single_image(img_path, phase_folder, pat_folder, is_train=True):
    """
    单个任务处理函数：同时处理 phase 和 pat，减少重复读取和路径解析
    """
    try:
        # 1. 路径解析
        base_name = os.path.basename(img_path).replace(".npy", "").replace("_sensor", "") # .replace('slm_', '')
        # final_result/SLM_1_noise_std_sigma0.5_19995_10000us_test.bmp.npy
        # base_name = os.path.basename(img_path).split("SLM_1_")[-1].split("_10000us")[0]
        # parent_dir = os.path.dirname(img_path)
        # 根据你的逻辑，phase 图片在结果文件夹的上一级
        # amp_folder = os.path.dirname(parent_dir)
        # amp_folder = os.path.dirname(amp_folder)
        # amp_folder = os.path.dirname(amp_folder)
        phase_dir = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260421_data/"
        phase_path = os.path.join(phase_dir, "input_perlin_20k", base_name + ".png")

        # 2. 处理 Phase 数据
        if os.path.exists(phase_path):
            phase = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
            if phase is not None:
                phase = phase.astype('f4') / 255.0
                np.save(os.path.join(phase_folder, f'phase_{base_name}.npy'), phase)
        else:
            print(img_path, phase_path)
            exit()

        # 3. 处理 Pat 数据 (包含 CenterCrop)
        # pat = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pat = np.load(img_path)
        if pat.max() < 10:
            print(img_path)
            exit()
        if pat is not None:
            # pat = pat.astype('f4') / 255.0
            # 使用 numpy 代替 torch 进行 CenterCrop (假设输入比 200x200 大)
            # 这样避免了 torch 内存搬运的开销
            h, w = pat.shape
            th, tw = 400, 400
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            pat_cropped = pat[i:i+th, j:j+tw]
            
            # 如果一定要保持 4D (1, 1, 200, 200)
            pat_final = pat_cropped[np.newaxis, np.newaxis, ...]
            np.save(os.path.join(pat_folder, f'pat_{base_name}.npy'), pat_final)
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    # amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260323_ASM_experimentaldata/SLMinput"
    # amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/public/AI_optics/Data_store/20260327/main/Pmode_input_random_20k_sigma_0_5_k_5"
    amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260421_data/output_random_perlin_4f_144.44"
    # amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/public/AI_optics/Data_store/20260428/final_result"
    
    # 快速搜集所有文件
    # folders = glob.glob(os.path.join(amp_folder, "SLMinput_Group_*"))
    # files = []
    # for fol in folders:
    #     sub_fols = glob.glob(os.path.join(fol, "results_*"))
    #     if sub_fols:
    #         imgs = glob.glob(os.path.join(sub_fols[0], "*.bmp"))
    #         files.extend(imgs)
    files = glob.glob(os.path.join(amp_folder, "*.npy"))

    train_split_idx = len(files) - 2000
    train_files = files[:train_split_idx]
    eval_files = files[train_split_idx:]

    # 准备目录
    dirs = {
        "train_phase": "../data/4f_random_perlin_20k/train/phase",
        "eval_phase": "../data/4f_random_perlin_20k/eval/phase",
        "train_pat": "../data/4f_random_perlin_20k/train/pat",
        "eval_pat": "../data/4f_random_perlin_20k/eval/pat"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 使用多进程池加速
    # max_workers 建议设置为 CPU 核心数的 1.5 倍左右，或者是磁盘写入能力的上限
    num_workers = min(os.cpu_count(), 16) 

    print(f"Starting Multi-processing with {num_workers} workers...")

    # 处理训练集
    print("Processing Train Set...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(
            partial(process_single_image, phase_folder=dirs["train_phase"], pat_folder=dirs["train_pat"]),
            train_files
        ), total=len(train_files)))

    # 处理验证集
    print("Processing Eval Set...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(
            partial(process_single_image, phase_folder=dirs["eval_phase"], pat_folder=dirs["eval_pat"]),
            eval_files
        ), total=len(eval_files)))

if __name__ == "__main__":
    main()