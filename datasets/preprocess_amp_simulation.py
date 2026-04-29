import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2


crop_transform = transforms.CenterCrop((200, 200))

start_idx = 1
train_split = 18000
eval_split = 2000

amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260204_ASM_experimentaldata/input"
pat_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260204_ASM_experimentaldata/20260208/A/main/Amode/"


train_phase_folder = os.path.join("../data/amp_data_experiment/train/amp")
eval_phase_folder = os.path.join("../data/amp_data_experiment/eval/amp")

train_pat_folder = os.path.join("../data/amp_data_experiment/train/pat")
eval_pat_folder = os.path.join("../data/amp_data_experiment/eval/pat")


os.makedirs(train_phase_folder, exist_ok=True)
os.makedirs(eval_phase_folder, exist_ok=True)
os.makedirs(train_pat_folder, exist_ok=True)
os.makedirs(eval_pat_folder, exist_ok=True)


# process train
for i in tqdm(range(start_idx, start_idx+train_split)):
    phase_name = "input_{:05d}.png".format(i)
    phase = cv2.imread(os.path.join(amp_folder, phase_name), cv2.IMREAD_GRAYSCALE) # np.load(os.path.join(phase_folder, phase_name)).astype("f4") / 1023.0
    phase = np.array(phase).astype('f4') / 255.0
    # phase = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)
    # phase = torch.clamp(phase, min=0.0, max=1.0).squeeze(0)
    
    # phase = 2*torch.pi*phase

    # phase = phase.numpy().astype('f4')

    np.save(os.path.join(train_phase_folder, phase_name.replace('.png', '.npy')), phase)

# process eval
for i in tqdm(range(train_split+start_idx, start_idx+train_split+eval_split)):
    phase_name = "input_{:05d}.png".format(i)
    phase = cv2.imread(os.path.join(amp_folder, phase_name), cv2.IMREAD_GRAYSCALE) # np.load(os.path.join(phase_folder, phase_name)).astype("f4") / 1023.0
    phase = np.array(phase).astype('f4') / 255.0
    # phase = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)
    # phase = torch.clamp(phase, min=0.0, max=1.0).squeeze(0)
    
    # phase = 2*torch.pi*phase

    # phase = phase.numpy().astype('f4')

    np.save(os.path.join(eval_phase_folder, phase_name.replace('.png', '.npy')), phase)


# process output 
for i in tqdm(range(start_idx, start_idx+train_split)):
    # pat_name = 'input_{:05d}_sensor.npy'.format(i)
    pat_name = 'slm_input_{:05d}.bmp'.format(i)
    # pat = np.load(os.path.join(pat_folder, pat_name))
    # pat = pat / 65535.0 # [0, 1] normalize
    pat = cv2.imread(os.path.join(pat_folder, pat_name), cv2.IMREAD_GRAYSCALE)
    pat = np.array(pat).astype('f4')/255.0
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    # pat = F.interpolate(pat, (50, 50), mode='bilinear')
    pat = crop_transform(pat)
    # pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)
    pat.squeeze(0)

    pat = pat.numpy().astype('f4')
    # pat = torch.sqrt(pat).numpy().astype('f4')

    pat_name = 'pat_{:05d}.npy'.format(i)
    np.save(os.path.join(train_pat_folder, pat_name), pat)


# process output 
for i in tqdm(range(start_idx+train_split, start_idx+train_split+eval_split)):
    # pat_name = 'input_{:05d}_sensor.npy'.format(i)
    pat_name = 'slm_input_{:05d}.bmp'.format(i)
    # pat = np.load(os.path.join(pat_folder, pat_name))
    # pat = pat / 65535.0 # [0, 1] normalize
    pat = cv2.imread(os.path.join(pat_folder, pat_name), cv2.IMREAD_GRAYSCALE)
    pat = np.array(pat).astype('f4')/255.0
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    # pat = F.interpolate(pat, (50, 50), mode='bilinear')
    pat = crop_transform(pat)
    # pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)
    pat.squeeze(0)

    pat = pat.numpy().astype('f4')
    # pat = torch.sqrt(pat).numpy().astype('f4')

    pat_name = 'pat_{:05d}.npy'.format(i)
    np.save(os.path.join(eval_pat_folder, pat_name), pat)