import torchvision.utils as tv_utils
import torch
import torch.nn.functional as F
import os
import numpy as np
from layers.vit import OpticsViT
from layers.vit_inr import OpticsViTINR
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision
from utils import PearsonLoss, pad_image, ssim
from PIL import Image

PHASE_SIZE = 50
PAT_SIZE = 100
MODE = '4f_random_sigma_0_5_k_5_20k'

pearson_criterion = PearsonLoss()

save_pat_dir = f'newton_results/pat_{PAT_SIZE}_{MODE}'
save_pred_dir = f'newton_results/preds_{PAT_SIZE}_{MODE}'
save_phase_dir = f'newton_results/phase_{PAT_SIZE}_{MODE}'

os.makedirs(save_pat_dir, exist_ok=True)
os.makedirs(save_pred_dir, exist_ok = True)
os.makedirs(save_phase_dir, exist_ok=True)

EMNIST_dataset = torchvision.datasets.EMNIST(root='data', download=False, split='letters')

img_transform = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor()
    ]
)

img_crop = transforms.Compose([
    transforms.CenterCrop((PAT_SIZE, PAT_SIZE)),
    transforms.ToTensor()
]
)


inp2pat = OpticsViTINR(image_size=PHASE_SIZE, patch_size=5, enc_depth=4, dec_depth=4, heads=8, dim_head=32, dim=256, 
                      mlp_dim=int(256*8/3), in_channels=2, out_channels=1, act=torch.nn.Sigmoid, out_dim=384, pat_size=PAT_SIZE).cuda()

ckp = torch.load('checkpoints/4f_random_20k_sigma_0_5_k_5_0421_simulation/stage1/best.pth', weights_only=False)
inp2pat.load_state_dict(ckp)

inp2pat.eval()
inp2pat.freeze()

def weight_l1(x, y, w):
    # x, y, w: [b, 1-3, h, w]
    diff = ((x - y).square())*w
    loss = diff.sum(dim=[1,2,3])/(w.sum(dim=[1,2,3])+1e-5)
    loss = loss.mean()
    return loss

# phase_img = Image.open('../datasets/20251226_ASM_test/input/generated_batch_20000samples_sigma0.5/input_11111.png').convert('L')
# phase_img = img_transform(phase_img).unsqueeze(0).to('cuda')*1.25*torch.pi
# # print(phase_img)

for idx in range(20000): 
    img = EMNIST_dataset[idx][0]
    img = img_transform(img).unsqueeze(0)
    
    # img = Image.open('preds/phase/gt_1.png').convert('L')
    # img = img_crop(img).unsqueeze(0)
    img = pad_image(img, PAT_SIZE, PAT_SIZE)
    # img = img * MEAN_INTEN / img.sum(dim=[1,2,3], keepdims=True)
    img = img.clamp(0, 1.0)
    img = img.cuda()
    # img = torch.sqrt(img)
    
    phase_param = torch.nn.Parameter(torch.randn(1, 1, PHASE_SIZE, PHASE_SIZE, device='cuda'))
    optimizer = torch.optim.AdamW([phase_param], lr=0.5)
    mask = (img > 0.01).float().detach()
    for step in tqdm(range(200), ncols=80):
        optimizer.zero_grad()
        # phase = F.normalize(phase_param, dim=1)
        phase = torch.sigmoid(phase_param)
        phase = phase * torch.pi * 2
        _, pred_img = inp2pat(phase, scale=PAT_SIZE/PHASE_SIZE) # Enforce 0-1
        
        loss = weight_l1(pred_img, img, mask) + weight_l1(pred_img, img, 1-mask) #  + ((1-mask)*pred_img).mean() # + F.mse_loss(pred_img, img)
        # loss = F.l1_loss(pred_img, img)
        # loss = pearson_criterion(pred_img, img)
        # loss = pearson_criterion(pred_img, img) # + F.mse_loss(pred_img, img)
        # loss = F.mse_loss(pred_img, img)
        # loss =  huber_criterion(pred_img, img)
        loss.backward()
        optimizer.step()
        # print(step, loss.item())
    phase_param = phase_param.mean(dim=0, keepdim=True)
    # phase_pse = F.normalize(phase_param, dim=1)
    # phase_pse = torch.sigmoid(phase_param)
    phase_pse = torch.sigmoid(phase_param) * torch.pi * 2

    # phase_radi = torch.atan2(phase_pse[:,[1]], phase_pse[:,[0]]+1e-7)
    # phase_radi = torch.remainder(phase_radi, 2 * torch.pi) / (torch.pi)
    # phase_radi = phase_radi.clamp(0, 1.0)
    phase_radi = torch.sigmoid(phase_param)
    # phase_radi = (phase_radi + torch.pi) / (2*torch.pi)
    # print(phase_radi)
    # print(phase_param)
    _, pat_recon = inp2pat(phase_pse, scale=PAT_SIZE/PHASE_SIZE)
    # phase_radi = phase_pse
    # pat_recon = torch.square(pat_recon).sum(dim=1, keepdims=True)

    tv_utils.save_image(pat_recon.detach().cpu(), f"{save_pred_dir}/emnist_{idx}.png")
    tv_utils.save_image(phase_radi, f"{save_phase_dir}/emnist_{idx}.png")
    phase = phase_radi[0, 0].detach().cpu() # (phase_radi * 1023)[0,0].detach().cpu().numpy().astype('uint16')
    np.save(f"{save_phase_dir}/emnist_{idx}.npy", phase)
    tv_utils.save_image(img.detach().cpu(), f"{save_pat_dir}/emnist_{idx}.png")


