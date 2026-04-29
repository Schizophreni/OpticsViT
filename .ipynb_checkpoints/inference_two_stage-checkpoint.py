import torchvision.utils as tv_utils
import torch
import torch.nn.functional as F
import os
import numpy as np
from layers.vit_size import OpticsViT
from layers.vit_inr import OpticsViTINR
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision
from utils import PearsonLoss, pad_image
from PIL import Image

PHASE_SIZE = 50
PAT_SIZE = 200
MODE = '4f_phase_pi'

save_pat_dir = f'two_stage_results/pat_{PAT_SIZE}_{MODE}'
save_pred_dir = f'two_stage_results/preds_{PAT_SIZE}_{MODE}'
save_phase_dir = f'two_stage_results/phase_{PAT_SIZE}_{MODE}'

os.makedirs(save_pat_dir, exist_ok=True)
os.makedirs(save_pred_dir, exist_ok = True)
os.makedirs(save_phase_dir, exist_ok=True)


EMNIST_dataset = torchvision.datasets.EMNIST(root='data', download=False, split='letters')

img_transform = transforms.Compose([
    transforms.Resize((PAT_SIZE, PAT_SIZE)),
    transforms.ToTensor()
    ]
)

img_crop = transforms.Compose([
    transforms.CenterCrop((PAT_SIZE, PAT_SIZE)),
    transforms.ToTensor()
]
)


# inp2pat = OpticsViTINR(image_size=PHASE_SIZE, patch_size=5, enc_depth=4, dec_depth=4, heads=8, dim_head=32, dim=256, 
#                       mlp_dim=int(256*8/3), in_channels=2, out_channels=1, act=torch.nn.Sigmoid, out_dim=384, use_learnable_pos=False, num_reg=0, drop_path_rate=0.0).cuda()

pat2inp = OpticsViT(input_size=PAT_SIZE, input_patch_size=10, enc_depth=4, output_size=PHASE_SIZE, output_patch_size=5, dec_depth=4, 
                      dim=384, heads=8, dim_head=48, mlp_dim=int(384*8/3), in_channels=1, out_channels=2, act=torch.nn.Tanh, out_norm=True, out_dim=256, drop_path_rate=0.0).cuda()

# ckp = torch.load(f'checkpoints/experiment_{PAT_SIZE}_6000us_stage1/stage1/best.pth', weights_only=False)
# inp2pat.load_state_dict(ckp)

ckp = torch.load(f'checkpoints/4f_twophases_pi_200/stage2_enc48_4_dec32_4_8heads_patch10/best.pth', weights_only=False)
pat2inp.load_state_dict(ckp)

# inp2pat.eval()
# inp2pat.freeze()

pat2inp.eval()
pat2inp.freeze()

# phase_img = Image.open('newton_results/phase/emnist_0.png').convert('L')
# phase_img = img_transform(phase_img).unsqueeze(0).to('cuda')*2*torch.pi
# print(phase_img)

for idx in tqdm(range(20000), ncols=80): 
    img = EMNIST_dataset[idx][0]
    img = img_transform(img).unsqueeze(0)
    
    # img = Image.open('objects/537.png').convert('L')
    # img = img_crop(img).unsqueeze(0)
    # # img = pad_image(img, 100, 100)
    # img = img * MEAN_INTEN / img.sum(dim=[1,2,3], keepdims=True)
    img = img.clamp(0, 1.0)
    img = img.cuda()
    
    phase, _ = pat2inp(img)
    # _, pat_recon = inp2pat(phase, scale=PAT_SIZE/PHASE_SIZE)
    

    phase_radi = torch.atan2(phase[:,[1]], phase[:,[0]]+1e-7)
    phase_radi = torch.remainder(phase_radi, torch.pi) / torch.pi
    # phase_radi = torch.remainder(phase_radi, 2 * torch.pi) / (2*torch.pi)
    
    # tv_utils.save_image(pat_recon.detach().cpu(), f"{save_pred_dir}/emnist_{idx}.png")
    tv_utils.save_image(phase_radi, f"{save_phase_dir}/emnist_{idx}.png")
    # phase = (phase_radi * 1023)[0,0].detach().cpu().numpy().astype('uint16')
    np.save(f"{save_phase_dir}/emnist_{idx}.npy", phase_radi[0, 0].cpu().numpy())
    # tv_utils.save_image(img.detach().cpu(), f"{save_pat_dir}/emnist_{idx}.png")


