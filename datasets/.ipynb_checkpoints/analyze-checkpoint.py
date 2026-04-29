from dataset import OpticsDataset
import torch
import torchvision.utils as tv_utils
from tqdm import tqdm
import numpy as np


train_dataset = OpticsDataset(train=True, root_dir="../data")
val_dataset = OpticsDataset(train=False, root_dir="../data")


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)

background = 0.0
cnt = 0

for _, pat, _ in tqdm(train_loader, ncols=60):
    background += pat.sum(dim=0)
    cnt += pat.shape[0]

for _, pat, _ in tqdm(val_loader, ncols=60):
    background += pat.sum(dim=0)
    cnt += pat.shape[0]

background = background / cnt
tv_utils.save_image(background, "back.png")

np.save("back.npy", background.numpy())


pat = np.load("../data/train/pat/amp_{:04d}_phase_{:04d}.npy".format(0, 10))
diff = (pat - background.numpy())
print(diff)
print(np.abs(diff).mean())
tv_utils.save_image(torch.from_numpy(diff), 'diff.png')
