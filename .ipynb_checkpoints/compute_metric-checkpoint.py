import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from utils import PearsonLoss
import os


def metric(folder_pred, folder_gt, criterion):
    results = []
    totensor = transforms.ToTensor()
    imgs = os.listdir(folder_pred)
    imgs = [item for item in imgs if "png" in item or "jpg" in item]
    imgs = sorted(imgs)
    print(f"Process {len(imgs)} imgs ...")
    for img in imgs:
        pred = Image.open(os.path.join(folder_pred, img))
        gt = Image.open(os.path.join(folder_gt, img))
        pred, gt = totensor(pred), totensor(gt)
        score = 1 - criterion(pred, gt)
        results.append(100*score)
    results = np.array(results)
    return results.mean()


folder_pred = "newton_results/pat_150_experiment_6000us_iter1_pearson"
folder_gt = "newton_results/preds_150_experiment_6000us_iter1"
criterion = PearsonLoss()

score = metric(folder_pred, folder_gt, criterion)
print("Mean score: ", score)
    