import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision.transforms as transforms

crop_transform = transforms.CenterCrop((200, 200))

train_split = 18000
eval_split = 2000

amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/public/AI_optics/Data_store/20260116DMD/output_npy"
pat_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/public/AI_optics/Data_store/20260124DMD/final_result"

train_amp_folder = os.path.join("../data/amp_data/train/amp")
eval_amp_folder = os.path.join("../data/amp_data/eval/amp")

train_pat_folder = os.path.join("../data/amp_data/train/pat")
eval_pat_folder = os.path.join("../data/amp_data/eval/pat")

os.makedirs(train_amp_folder, exist_ok=True)
os.makedirs(eval_amp_folder, exist_ok=True)
os.makedirs(train_pat_folder, exist_ok=True)
os.makedirs(eval_pat_folder, exist_ok=True)


# process train
for i in tqdm(range(train_split)):
    amp_name = "matrix_{:05d}.npy".format(i)
    amp = np.load(os.path.join(amp_folder, amp_name)).astype("f4") / 255.0
    amp = torch.from_numpy(amp).unsqueeze(0)
    amp = amp.numpy().astype('f4')
    np.save(os.path.join(train_amp_folder, amp_name), amp)

# process eval
for i in tqdm(range(train_split, train_split+eval_split)):
    amp_name = "matrix_{:05d}.npy".format(i)
    amp = np.load(os.path.join(amp_folder, amp_name)).astype("f4") / 255.0
    amp = torch.from_numpy(amp).unsqueeze(0)
    amp = amp.numpy().astype('f4')
    np.save(os.path.join(eval_amp_folder, amp_name), amp)


# process output 
for i in tqdm(range(train_split)):
    # for j in range(train_split + eval_split):
    pat_name = 'DMD,matrix_{:05d}_SLM,flat0_51000us_test.npy'.format(i)
    pat = np.load(os.path.join(pat_folder, pat_name))
    pat = pat / 65535.0 # [0, 1] normalize
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    
    pat = crop_transform(pat)
    pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)

    pat = pat.numpy().astype('f4')

    pat_name = 'amp_{:05d}.npy'.format(i)
    np.save(os.path.join(train_pat_folder, pat_name), pat)



# process output 
for i in tqdm(range(train_split, train_split+eval_split)):
    pat_name = 'DMD,matrix_{:05d}_SLM,flat0_51000us_test.npy'.format(i)
    pat = np.load(os.path.join(pat_folder, pat_name))
    pat = pat / 65535.0 # [0, 1] normalize
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    
    pat = crop_transform(pat)
    pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)

    pat = pat.numpy().astype('f4')

    pat_name = 'amp_{:05d}.npy'.format(i)
    np.save(os.path.join(eval_pat_folder, pat_name), pat)