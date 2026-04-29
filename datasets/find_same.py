import os
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

img_folder = "../data/4f_random_20k_sigma_0_5_k_5_0401_Pmode/train/pat/"
threshold = 1 / 255.0
output_txt = "match_result.txt"

imgs = sorted(glob.glob(os.path.join(img_folder, "*.npy")))
eval_imgs = sorted(glob.glob(os.path.join(img_folder.replace("train", "eval"), "*.npy")))
imgs = imgs + eval_imgs
remaining = set(imgs)

max_workers = min(16, (os.cpu_count() or 1) * 2)


def check_one(args):
    img_path, src_img, threshold = args
    try:
        img_val = np.load(img_path, mmap_mode="r")
        img_val = img_val[0, 0, :200, :200]
        src_img = src_img[0, 0, :200, :200]
        diff = np.abs(img_val - src_img).mean()
        if diff < threshold:
            return img_path
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return None


with open(output_txt, "w", encoding="utf-8") as f:
    for i, src_path in enumerate(tqdm(imgs, ncols=80, desc="Processing src")):
        if src_path not in remaining:
            continue

        src_img = np.load(src_path, mmap_mode="r")

        candidates = [
            img_path
            for img_path in imgs[i + 1 :]
            if img_path in remaining
        ]

        if not candidates:
            continue

        tasks = [(img_path, src_img, threshold) for img_path in candidates]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(check_one, tasks))

        matched = [x for x in results if x is not None]

        if matched:
            src_name = os.path.basename(src_path)
            for match_path in matched:
                match_name = os.path.basename(match_path)
                f.write(f"{src_name}, {match_name}\n")
                remaining.discard(match_path)
            f.flush()

print(f"Done. Result saved to: {output_txt}")
