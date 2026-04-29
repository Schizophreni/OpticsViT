"""
Stage1: mimicking the physical system of transforming the image to optical pattern
Use torch mixed precision training to accelerate the training process
Ref: https://www.cnblogs.com/jimchen1218/p/14315008.html
"""
import os
import torch
import torch.nn as nn
import torch.fft as fft
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from datasets.simulation_dataset import OpticsDataset
# from datasets.Amp_dataset import OpticsDataset
from utils import psnr, ssim  # metrics
from tqdm import tqdm
import numpy as np
import torchvision
from scheduler import WarmupCosineLR
from layers.vit_inr import OpticsViTINR
import torch.nn.functional as F
import pdb
from PIL import Image
import torchvision.transforms as transforms
import shutil
import wandb
from utils import SSIM


def comp_inten(x):
    return torch.square(x).sum(dim=1, keepdims=True)

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss() # 通常 L1 比 MSE 在频域效果更稳健

    def forward(self, pred, target):
        """
        pred, target: [B, C, H, W] 的张量
        """
        # 1. 计算二维快速傅里叶变换
        # 使用 rfft2 如果输入是实数，效率更高；这里用 fft2 更通用
        pred_fft = fft.fft2(pred, norm='ortho')
        target_fft = fft.fft2(target, norm='ortho')

        # 2. 提取幅度谱 (Amplitude Spectrum)
        # 散斑的本质信息隐藏在幅度中（即功率谱的平方根）
        pred_abs = torch.abs(pred_fft)
        target_abs = torch.abs(target_fft)

        # 3. 计算频域损失
        fft_loss = self.criterion(pred_abs, target_abs)

        return self.loss_weight * fft_loss


def get_radial_decay_mask(h, w, device, eps=0.1, power=1.0):
    """
    生成一个从中心向边缘按 1/(R^power) 衰减的 Mask
    h, w: 图像尺寸
    eps: 防止除以 0 的平滑项，值越小中心权重越陡峭
    power: 衰减幂次，1.0 即为 1/R
    """
    # 生成归一化坐标 [-1, 1]
    y = torch.linspace(-1, 1, h, device=device) * h
    x = torch.linspace(-1, 1, w, device=device) * w
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # 计算径向距离 R (中心为 0, 边缘最大为 sqrt(2))
    r = torch.sqrt(grid_x**2 + grid_y**2)
    
    # 计算衰减权重 W = 1 / (R + eps)
    # 也可以归一化一下，让中心最大权重为 1
    mask = 1.0 / (r + eps)
    coeff = h*w / (mask.sum())
    mask = mask * coeff
    
    # 如果你想严格限制在 SLM 的圆形有效区域内，可以加一个圆剪裁
    # circular_aperture = (r <= 1.0).float()
    # mask = mask * circular_aperture
    
    return mask # [H, W]

class SLMWeightedL1Loss(torch.nn.Module):
    def __init__(self, eps=0.1, power=1.0):
        super().__init__()
        self.eps = eps
        self.power = power
        self.mask = None

    def forward(self, pred, target):
        if self.mask is None or self.mask.shape != pred.shape[-2:]:
            self.mask = get_radial_decay_mask(
                pred.shape[-2], pred.shape[-1], 
                pred.device, self.eps, self.power
            )
        
        # 计算加权 L1
        loss = torch.abs(pred - target) * self.mask
        return loss.mean()


# 使用示例
# criterion = FocalAreaLoss(alpha=100.0, dilation_kernel=11)
# loss = criterion(prediction, ground_truth)

    
pixel_criterion = torch.nn.L1Loss()  # pixel criterion
l2_criterion = torch.nn.MSELoss()
fft_criterion = FFTLoss(0.2)
ssim_criterion = SSIM()
focal_loss = SLMWeightedL1Loss()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./data", help="path to the data")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--gpu_ids", type=str, default="0", help="device")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="path to save the model")
    parser.add_argument("--resume_path", type=str, default="", help="path to load the model")
    parser.add_argument("--print_interval", type=int, default=10, help="print interval")
    parser.add_argument("--phase_size", type=int, default=224, help="resize size for images and patterns")
    parser.add_argument("--pat_size", type=int, default=224, help="resize size for images and patterns")
    parser.add_argument("--exp_name", type=str, default="vit_inr")
    args = parser.parse_args()
    return args

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, scaler,
                    device, epoch, print_interval=10, wandb_writer=None, args=None):
    model.train()
    running_loss = 0.0
    running_loss_inr = 0.0
    max_scale = args.pat_size / args.phase_size
    for i, (signal, pat, _, _) in enumerate(dataloader):
        lr_scheduler.step()
        # print(signal.min(), signal.max())
        signal, pat = signal.to(device), pat.to(device)
        # singal_shift = signal_shift.to(device)
        # print(signal.shape, pat.shape)
        # print(signal.shape, pat.shape, signal.max(), signal.mean(), pat.max(), pat.min(), pat.mean())
        # scale = np.random.rand()*(max_scale -1) + 1
        # resize_size = int(scale * signal.shape[-1])
        # pat_resize = F.interpolate(pat, (resize_size, resize_size), mode='bilinear').clamp(0, 1)
        # pat_0 = F.interpolate(pat, (signal.shape[-1], signal.shape[-1]), mode='bilinear').clamp(0, 1)
        # print(pat.shape)
        phase_shift = (torch.rand(signal.shape[0], 1, 1, 1) * torch.pi/2) % (2*torch.pi)
        phase_shift = phase_shift.to(signal.device)
        # pat = torch.sqrt(pat)
        
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            _, out_pred = model(signal, scale=max_scale)
            # print(out_pred.shape, pat.shape)
            # out_norm = comp_inten(out_pred)
            # out_real, out_imag = torch.chunk(out_pred, chunks=2, dim=1)
            
#             _, out_shift = model(signal + phase_shift, scale=max_scale)
#             out_real_shift, out_imag_shift = torch.chunk(out_shift, chunks=2, dim=1)
            
#             out_real_phys = torch.cos(phase_shift)*out_real - torch.sin(phase_shift)*out_imag
#             out_imag_phys = torch.sin(phase_shift)*out_real + torch.cos(phase_shift)*out_imag
            
            # print(signal.max(), signal.min())
        # with torch.amp.autocast("cuda"):
            # signal_shift = signal_shift.to(signal)
            # _, out_pred_shift = model(signal_shift, scale=max_scale)
            # out_0, out_inr = model(signal, scale=scale)
            
            # fft_loss = fft_criterion(out_pred.float(), pat.float())
            # ssim_loss = 1 - ssim_criterion(out_pred, pat)
            mean_loss = F.l1_loss(out_pred.mean(dim=[1,2,3]), pat.mean(dim=[1,2,3]))
            ## without mean loss, reverse better
            loss = F.l1_loss(out_pred, pat) + mean_loss
            # loss_phys = F.mse_loss(out_real_shift, out_real_phys) + F.mse_loss(out_imag_shift, out_imag_phys)
            
            # loss = F.mse_loss(torch.log(out_pred+1), torch.log(1+pat))
            # loss_inr = l2_criterion(out_inr, pat_resize) + l2_criterion(out_0, pat_0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        running_loss_inr += mean_loss.item()
        # pdb.set_trace()
        if (1+i) % print_interval == 0:
            print(f"[epoch: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_interval:.5f}, loss_inr: {running_loss_inr / print_interval:.5f}")
            wandb_writer.log(
                {
                    "epoch": epoch + 1,
                    "iter": i + 1,
                    "train/loss": running_loss / print_interval,
                    "train/loss_inr": running_loss_inr / print_interval,
                }
            )
            running_loss = 0.0
            running_loss_inr = 0.0

@torch.no_grad()
def validate(model, val_loader, device, args, wandb_writer=None):
    model.eval()
    psnrs, ssims = [], []
    max_scale = args.pat_size / args.phase_size
    with torch.no_grad():
        for idx, (signal, pat, pat_name, _) in tqdm(enumerate(val_loader), ncols=60, desc="Validating"):
            signal, pat = signal.to(device), pat.to(device)
            print(signal.min(), signal.max())
            # pat = torch.sqrt(pat)
            _, out_pred = model(signal, scale=max_scale)
            out_pred_norm = out_pred # comp_inten(out_pred)
            # out_pred_norm = torch.sqrt(torch.square(out_pred).sum(dim=1, keepdim=True))
            # print(l2_criterion(out_pred_norm, pat).item())
            out_pred_norm.clamp_(0.0, 1.0)
            if idx <= 8:
                torchvision.utils.save_image(signal[[0]].cpu()/(2*torch.pi), os.path.join("preds/phase", "inp_{}.png".format(idx+1)))
                torchvision.utils.save_image(pat[[0],...], os.path.join("preds/phase", "gt_{}.png".format(idx+1)))
                torchvision.utils.save_image(out_pred_norm[[0],...].cpu(), os.path.join("preds/phase", "pred_{}.png".format(idx+1)))
            if idx == 0:
                tmp = signal / signal.max()
                inputs = tmp[:4].cpu()
                preds = out_pred[:4].cpu()
                gts = pat[:4].cpu()

                # 获取三者中最大的 H 和 W
                max_h = max(inputs.shape[2], preds.shape[2], gts.shape[2])
                max_w = max(inputs.shape[3], preds.shape[3], gts.shape[3])

                def align_tensor(t, h, w):
                    if t.shape[2] != h or t.shape[3] != w:
                        return F.interpolate(t, size=(h, w), mode='nearest') # 仿真数据有时用 nearest 更好
                    return t

                inputs = align_tensor(inputs, max_h, max_w)
                preds = align_tensor(preds, max_h, max_w)
                gts = align_tensor(gts, max_h, max_w)

                combined = torch.cat([inputs, preds, gts], dim=3)
                grid = torchvision.utils.make_grid(combined, nrow=1, padding=2) # 纵向排列 4 个对比组

                # 3. 直接 log 图像
                wandb.log({
                    "val/predictions_grid": wandb.Image(grid, caption="Left: Input | Middle: Pred | Right: GT")
                })
            psnrs.append(psnr(out_pred_norm*255, pat*255).item())
    psnr_mean = np.mean(np.array(psnrs))
    return psnr_mean

def load_checkpoint(model, optimizer, lr_scheduler, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    # scaler.load_state_dict(checkpoint["scaler"])
    best_psnr = checkpoint["best_psnr"]
    return checkpoint["epoch"], best_psnr

def main(args):
    # set devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda") if len(args.gpu_ids) > 0 else torch.device("cpu")
    save_path = os.path.join(args.save_path, "stage1")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # initialize wandb
    wandb_log = wandb.init(
        project="AIOptics",
        entity="wurancs-int",
        name=args.exp_name,
        config=args # resume="allow"
    )
    # use wandb to log files
    artifact = wandb.Artifact(name="code", type='code')
    artifact.add_file("layers/vit_inr.py")
    artifact.add_file("datasets/simulation_dataset.py")
    artifact.add_file("main_stage1_simulation.py")
    artifact.add_file("run.sh")
    wandb_log.log_artifact(artifact)
    
    # build model
    model = OpticsViTINR(image_size=args.phase_size, patch_size=5, enc_depth=4, dec_depth=4, heads=8, dim_head=32, dim=256, 
                      mlp_dim=int(256*8/3), in_channels=2, out_channels=1, act=torch.nn.Sigmoid, out_dim=384, use_learnable_pos=False, num_reg=0, drop_path_rate=0.1, input_pad=0, pat_size=args.pat_size)
    model = model.to(device)
    # print(model)
    # build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupCosineLR(optimizer, warmup_iters=1500, total_iters=args.num_epochs*18000//args.batch_size)
    scaler = GradScaler()
    # resume from checkpoint
    if os.path.exists(args.resume_path):
        print("Resume from: ", args.resume_path)
        start_epoch, best_psnr = load_checkpoint(model, optimizer, lr_scheduler, scaler, args.resume_path)
        start_epoch, best_psnr = 0, 0
    else:
        start_epoch, best_psnr = 0, 0.0
    # build dataloader
    train_dataset = OpticsDataset(train=True, root_dir=args.root_dir)
    val_dataset = OpticsDataset(train=False, root_dir=args.root_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
    # train
    for epoch in range(start_epoch, args.num_epochs):
        # validate(model, val_loader, device)
        train_one_epoch(model, train_loader, optimizer, lr_scheduler, scaler, device, epoch, print_interval=args.print_interval,
                        wandb_writer=wandb_log, args=args)
        # save checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr
        }, os.path.join(save_path, "latest.pth"))
        # validate
        psnr_mean = validate(model, val_loader, device, args=args, wandb_writer=wandb_log)
        wandb_log.log(
            {
            "epoch": epoch + 1,
            "val/psnr": psnr_mean
            }
        )
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))


if __name__ == "__main__":
    args = get_args()
    main(args)