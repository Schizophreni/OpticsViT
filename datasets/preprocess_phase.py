import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision.transforms as transforms


crop_transform = transforms.CenterCrop((200, 200))

start_idx = 0
train_split = 38000
eval_split = 2000

phase_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/public/AI_optics/Data_store/20251213SLM/output_npy"
pat_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/public/AI_optics/Data_store/final_result/"


train_phase_folder = os.path.join("../data/data_6000us_iter1/train/phase")
eval_phase_folder = os.path.join("../data/data_6000us_iter1/eval/phase")

train_pat_folder = os.path.join("../data/data_6000us_iter1/train/pat")
eval_pat_folder = os.path.join("../data/data_6000us_iter1/eval/pat")


os.makedirs(train_phase_folder, exist_ok=True)
os.makedirs(eval_phase_folder, exist_ok=True)
os.makedirs(train_pat_folder, exist_ok=True)
os.makedirs(eval_pat_folder, exist_ok=True)



# process train
# for i in tqdm(range(start_idx, start_idx+train_split)):
#     phase_name = "matrix_{:05d}.npy".format(i)
#     phase = np.load(os.path.join(phase_folder, phase_name)).astype("f4") / 1023.0
#     phase = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)
#     phase = torch.clamp(phase, min=0.0, max=1.0).squeeze(0)
    
#     phase = 2*torch.pi*phase

#     phase = phase.numpy().astype('f4')

#     np.save(os.path.join(train_phase_folder, phase_name), phase)

# # process eval
# for i in tqdm(range(train_split+start_idx, start_idx+train_split+eval_split)):
#     phase_name = "matrix_{:05d}.npy".format(i)
#     phase = np.load(os.path.join(phase_folder, phase_name)).astype("f4") / 1023.0
#     phase = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)
#     phase = torch.clamp(phase, min=0.0, max=1.0).squeeze(0)
    
#     phase = 2*torch.pi*phase

#     phase = phase.numpy().astype('f4')

#     np.save(os.path.join(eval_phase_folder, phase_name), phase)


# process output 
for i in tqdm(range(start_idx, start_idx+train_split)):
    pat_name = 'SLM_mat_emnist_{:05d}_6000us_test.bmp.npy'.format(i)
    pat = np.load(os.path.join(pat_folder, pat_name))
    pat = pat / 65535.0 # [0, 1] normalize
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    # pat = F.interpolate(pat, (50, 50), mode='bilinear')
    pat = crop_transform(pat)
    pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)

    pat = pat.numpy().astype('f4')
    # pat = torch.sqrt(pat).numpy().astype('f4')

    pat_name = 'emnist_{:05d}.npy'.format(i)
    np.save(os.path.join(train_pat_folder, pat_name), pat)


# process output 
for i in tqdm(range(start_idx+train_split, start_idx+train_split+eval_split)):
    pat_name = 'SLM_mat_emnist_{:05d}_6000us_test.bmp.npy'.format(i)
    pat = np.load(os.path.join(pat_folder, pat_name))
    pat = pat / 65535.0 # [0, 1] normalize
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    # pat = F.interpolate(pat, (50, 50), mode='bilinear')
    pat = crop_transform(pat)
    pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)

    pat = pat.numpy().astype('f4')
    # pat = torch.sqrt(pat).numpy().astype('f4')

    pat_name = 'emnist_{:05d}.npy'.format(i)
    np.save(os.path.join(eval_pat_folder, pat_name), pat)