import torchvision.utils as tv_utils
import torch
import torch.nn.functional as F
import os
import numpy as np
from layers.vit import OpticsViT
# from layers.mlp import MLP
from layers.vit_inr import OpticsViTINR
from PIL import Image
import torchvision.transforms as transforms

img_transform = transforms.Compose([
    # transforms.CenterCrop((96, 96)),
    transforms.Resize((200, 200)),
    transforms.ToTensor()
    ]
)

os.makedirs('debug', exist_ok=True)

pat2inp = OpticsViT(image_size=50, patch_size=5, depth=12, heads=8, dim_head=64, dim=512, 
                      mlp_dim=2048, depth_pred=12, mask_size=10, channels=1, out_channels=2)
# inp2pat = OpticsViT(image_size=50, patch_size=5, depth=6, heads=8, dim_head=64, dim=512, 
                      # mlp_dim=2048, depth_pred=6, mask_size=10, channels=2, out_channels=1)
inp2pat = OpticsViTINR(image_size=50, patch_size=5, enc_depth=6, dec_depth=6, heads=8, dim_head=64, dim=512, 
                      mlp_dim=2048, in_channels=2, out_channels=1, act=torch.nn.Sigmoid).cuda()
# inp2pat = MLP(input_size=50, in_channels=2, out_channels=1, mode='amp', output_size=200) 
# pat2inp = MLP(input_size=200, in_channels=1, out_channels=2, mode='real', output_size=50)

ckp = torch.load('checkpoints/inr/stage1/best.pth', weights_only=False)
inp2pat.load_state_dict(ckp)

ckp = torch.load('checkpoints/phase/stage2/best.pth', weights_only=False)
pat2inp.load_state_dict(ckp)

pat2inp.eval()
inp2pat.eval()
pat2inp.freeze()
inp2pat.freeze()

img_name = "preds/stage2/emnist_5.png"
# img_name = "MNIST/MINST_3261.png"

# img_name = "preds/phase/gt_1.png"

img = Image.open(img_name)

img = img_transform(img).unsqueeze(0)[:,[0],...].cuda()

# img = img * torch.rand_like(img)

# inp_pred = pat2inp(img)

# inp_pred_norm = torch.sqrt(torch.square(inp_pred).sum(dim=1, keepdim=True))

# inp_pred_norm /= inp_pred_norm.max()

# pat_recon = inp2pat(inp_pred)
# pat_recon = torch.sqrt(torch.square(pat_recon).sum(dim=1, keepdim=True))

# phase = torch.arctan(inp_pred[:,[1],...] / inp_pred[:,[0],...]) + torch.pi
# phase = phase / (2*torch.pi)

# print(inp_pred)

# phase = torch.arctan(inp_pred[:,[1],:,:] / (inp_pred[:,[0],:,:]+1e-5)) + torch.pi
# phase = phase / (2*torch.pi)
# # phase = inp_pred

# # inten = torch.square(inp_pred).sum(dim=1, keepdim=True)

# # print(inten)
# print(phase)
# # inten = inten / inten.max()

# tv_utils.save_image(phase, 'debug/phase_pred_{}.png'.format(img_name.split("/")[-1].split(".")[0]))
# tv_utils.save_image(pat_recon, f'debug/pat_recon.png')
# tv_utils.save_image(img, f'debug/pat.png')
# tv_utils.save_image(inten, 'debug/inten_pred_{}.png'.format(img_name.split("/")[-1].split(".")[0]))
# tv_utils.save_image(inp_pred_norm, f'debug/inp_pred_norm.png')

# phase = (phase * 1023)[0,0].detach().numpy().astype('uint16')
# inten = (inten * 65535)[0, 0].detach().numpy().astype('uint16')
# np.save(f"debug/npy/phase_{img_name.split("/")[-1].split(".")[0]}.npy", phase)
# np.save(f"debug/npy/inten_{img_name.split("/")[-1].split(".")[0]}.npy", inten)

phase_param = torch.nn.Parameter(torch.randn(1, 2, 50, 50, device='cuda'))
inp2pat.freeze()
optimizer = torch.optim.AdamW([phase_param], lr=0.2)
for step in range(100):
    optimizer.zero_grad()
    # phase = torch.tanh(phase_param) # *2*torch.pi
    phase = F.normalize(phase_param, dim=1)
    # phase = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
    _, pred_img = inp2pat(phase, scale=4) # Enforce 0-1
    loss = F.mse_loss(pred_img, img)
    loss.backward()
    optimizer.step()
    print(step, loss.item())
# phase_pse = torch.tanh(phase_param)
phase_pse = F.normalize(phase_param, dim=1)

phase_radi = (torch.atan2(phase_pse[:,[1]], phase_pse[:,[0]]+1e-7) + torch.pi) / (2*torch.pi)
print(phase_radi.min(), phase_radi.max())
_, pat_recon = inp2pat(phase_pse, scale=4)

tv_utils.save_image(pat_recon.detach().cpu(), f"debug/recon_silver_{img_name.split("/")[-1].split(".")[0]}.png")
# phase_pse = torch.sigmoid(phase_param)
tv_utils.save_image(phase_radi, f"debug/phase_silver_{img_name.split("/")[-1].split(".")[0]}.png")
phase = (phase_radi * 1023)[0,0].detach().cpu().numpy().astype('uint16')
np.save(f"debug/npy/phase_{img_name.split("/")[-1].split(".")[0]}.npy", phase)


