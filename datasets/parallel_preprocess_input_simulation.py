import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse


def process_single_image(img_path, raw_input_folder, save_input_folder, save_target_folder):
    """
    单个任务处理函数：同时处理 phase 和 pat，减少重复读取和路径解析
    """
    try:
        # 1. 路径解析
        base_name = os.path.basename(img_path).replace(".npy", "").replace("_sensor", "") # .replace('slm_', '')
        base_name = base_name.split("SLM_1_")[-1].split("_10000us")[0]
        phase_path = os.path.join(raw_input_folder, base_name + ".png")

        # 2. 处理 Phase 数据
        if os.path.exists(phase_path):
            phase = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
            if phase is not None:
                phase = phase.astype('f4') / 255.0
                np.save(os.path.join(save_input_folder, f'phase_{base_name}.npy'), phase)
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
            np.save(os.path.join(save_target_folder, f'pat_{base_name}.npy'), pat_final)
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def get_args():
    parser = argparse.ArgumentParser(description="Optics ViT Training Script")
    parser.add_argument("--raw_target_folder", type=str, help="folder to save target data")
    parser.add_argument("--raw_input_folder", type=str, help="raw output folder")
    parser.add_argument("--save_folder", type=str, help="folder to save processed data")
    parser.add_argument("--input-mode", type=str, default="phase", choices=["phase", "amp"], help="type of input data to process")
    return parser.parse_args()

def main():
    args = get_args()
    amp_folder = args.raw_target_folder
    files = glob.glob(os.path.join(amp_folder, "*.npy"))

    train_split_idx = len(files) - 2000  # 2000 for validation
    train_files = files[:train_split_idx]
    eval_files = files[train_split_idx:]

    # 准备目录
    dirs = {
        "train_input": f"{args.save_folder}/train/{args.input_mode}",
        "eval_input": f"{args.save_folder}/eval/{args.input_mode}",
        "train_pat": f"{args.save_folder}/train/pat",
        "eval_pat": f"{args.save_folder}/eval/pat"
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
            partial(process_single_image, raw_input_folder=args.raw_input_folder, save_input_folder=dirs["train_input"], save_target_folder=dirs["train_pat"]),
            train_files
        ), total=len(train_files)))

    # 处理验证集
    print("Processing Eval Set...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(
            partial(process_single_image, raw_input_folder=args.raw_input_folder, save_input_folder=dirs["eval_input"], save_target_folder=dirs["eval_pat"]),
            eval_files
        ), total=len(eval_files)))

if __name__ == "__main__":
    main()