import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2
import glob


crop_transform = transforms.CenterCrop((200, 200))


# # amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20251226_ASM_test/input_random_20k_sigma_0_5_k_5"
# amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/AIOptics/newton_results/pat_150_experiment_6000us_emnist_all/"
# amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260204_ASM_experimentaldata/input"
# pat_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20251226_ASM_test/output_random_20k_sigma_0_5_k_5"
# pat_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260204_ASM_experimentaldata/20260208/P/main/Pmode2pi"

# files = os.listdir(pat_folder)
# files = [item for item in files if "png" in item]

amp_folder = "/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260323_ASM_experimentaldata/SLMinput"
files = []

folders = glob.glob(os.path.join(amp_folder, "SLMinput_Group_*"))
for fol in folders:
    print(fol)
    sub_fol = glob.glob(os.path.join(fol, "results_*"))[0]
    imgs = glob.glob(os.path.join(sub_fol, "*.bmp"))
    files.extend(imgs)


start_idx = 0
train_split = len(files) - 2000
eval_split = 2000

train_phase_folder = os.path.join("../data/4f_comb_phases_20k_experiment/train/phase")
eval_phase_folder = os.path.join("../data/4f_comb_phases_20k_experiment/eval/phase")

train_pat_folder = os.path.join("../data/4f_comb_phases_20k_experiment/train/pat")
eval_pat_folder = os.path.join("../data/4f_comb_phases_20k_experiment/eval/pat")


os.makedirs(train_phase_folder, exist_ok=True)
os.makedirs(eval_phase_folder, exist_ok=True)
os.makedirs(train_pat_folder, exist_ok=True)
os.makedirs(eval_pat_folder, exist_ok=True)


# process train
for i in tqdm(range(start_idx, start_idx+train_split)):
    # phase_name = "input_{:05d}.png".format(i)
    # phase_name = files[i].replace("slm_", "").replace(".bmp", ".png")
    # phase_name = files[i].replace("_sensor.png", ".png")
    img = files[i]
    amp_folder = img.split("results_")[0]
    phase_name = img.split("/")[-1].replace(".bmp", ".png")
    phase = cv2.imread(os.path.join(amp_folder, phase_name), cv2.IMREAD_GRAYSCALE) # np.load(os.path.join(phase_folder, phase_name)).astype("f4") / 1023.0
    phase = np.array(phase).astype('f4') / 255.0
    # phase = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)
    # phase = torch.clamp(phase, min=0.0, max=1.0).squeeze(0)
    
    # phase = 2*torch.pi*phase

    # phase = phase.numpy().astype('f4')

    # np.save(os.path.join(train_phase_folder, phase_name.replace('.png', '.npy')), phase)
    np.save(os.path.join(train_phase_folder, 'phase_{}.npy'.format(phase_name.split(".png")[0])), phase)

# process eval
for i in tqdm(range(train_split+start_idx, start_idx+train_split+eval_split)):
    # phase_name = "input_{:05d}.png".format(i)
    # phase_name = files[i].replace("_sensor.png", ".png")
    # phase_name = files[i].replace("slm_", "").replace(".bmp", ".png")
    img = files[i]
    amp_folder = img.split("results_")[0]
    phase_name = img.split("/")[-1].replace(".bmp", ".png")
    phase = cv2.imread(os.path.join(amp_folder, phase_name), cv2.IMREAD_GRAYSCALE) # np.load(os.path.join(phase_folder, phase_name)).astype("f4") / 1023.0
    phase = np.array(phase).astype('f4') / 255.0
    # phase = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)
    # phase = torch.clamp(phase, min=0.0, max=1.0).squeeze(0)
    
    # phase = 2*torch.pi*phase

    # phase = phase.numpy().astype('f4')

    # np.save(os.path.join(eval_phase_folder, phase_name.replace('.png', '.npy')), phase)
    np.save(os.path.join(eval_phase_folder, 'phase_{}.npy'.format(phase_name.split(".png")[0])), phase)


# # # process output 
for i in tqdm(range(start_idx, start_idx+train_split)):
    # pat_name = 'input_{:05d}_sensor.npy'.format(i)
    # pat_name = files[i].replace("png", "npy")
    # pat_name = 'slm_input_{:05d}.bmp'.format(i)
    img = files[i]
    pat = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # pat = np.load(os.path.join(pat_folder, pat_name))
    # pat = cv2.imread(os.path.join(pat_folder, pat_name), cv2.IMREAD_GRAYSCALE)
    pat = np.array(pat).astype('f4')/255.0
    # # pat = pat / 65535.0 # [0, 1] normalize
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    # pat = F.interpolate(pat, (50, 50), mode='bilinear')
    pat = crop_transform(pat)
    # pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)
    # pat = pat[:, :, 170:370, 510:710]
    pat.squeeze(0)

    pat = pat.numpy().astype('f4')
    # pat = torch.sqrt(pat).numpy().astype('f4')

    # pat_name = 'pat_{}.npy'.format(pat_name.split("_sensor")[0])
    pat_name = 'pat_{}.npy'.format(pat_name.split("/")[-1].split(".bmp")[0])
    np.save(os.path.join(train_pat_folder, pat_name), pat)


# process output 
for i in tqdm(range(start_idx+train_split, start_idx+train_split+eval_split)):
    # pat_name = 'input_{:05d}_sensor.npy'.format(i)
    # pat_name = 'slm_input_{:05d}.bmp'.format(i)
    # pat_name = files[i].replace("png", "npy")
    img = files[i]
    # pat = np.load(os.path.join(pat_folder, pat_name))
    # pat = cv2.imread(os.path.join(pat_folder, pat_name), cv2.IMREAD_GRAYSCALE)
    # pat = np.array(pat).astype('f4')/255.0
    pat = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    pat = np.array(pat).astype('f4')/255.0
    pat = torch.from_numpy(pat).unsqueeze(0).unsqueeze(0)
    # pat = F.interpolate(pat, (50, 50), mode='bilinear')
    pat = crop_transform(pat)
    # pat = torch.clamp(pat, min=0.0, max=1.0).squeeze(0)
    # pat = pat[:, :, 170:370, 510:710]
    pat.squeeze(0)

    pat = pat.numpy().astype('f4')
    # pat = torch.sqrt(pat).numpy().astype('f4')

    # pat_name = 'pat_{}.npy'.format(pat_name.split("_sensor")[0])
    pat_name = 'pat_{}.npy'.format(pat_name.split("/")[-1].split(".bmp")[0])
    np.save(os.path.join(eval_pat_folder, pat_name), pat)