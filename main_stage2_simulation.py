"""
Stage1: mimicking the physical system of transforming the image to optical pattern
Use torch mixed precision training to accelerate the training process
Ref: https://www.cnblogs.com/jimchen1218/p/14315008.html
"""
import os
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from datasets.simulation_dataset import OpticsDataset
from utils import psnr, ssim  # metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import torchvision
from layers.vit_size import OpticsViT
from layers.vit_inr import OpticsViTINR
from utils import mask_loss
from scheduler import WarmupCosineLR
from layers.mlp import MLP
from utils import PearsonLoss
import pdb
from utils import contractive_loss as contractive_criterion
import torch.nn.functional as F
from utils import CosLoss, corr, BetaSampling 
import shutil


pixel_criterion = torch.nn.L1Loss()  # pixel criterion
l2_criterion = torch.nn.MSELoss()
pearson_criterion = PearsonLoss()
cos_criterion = CosLoss

def weight_l1(x, y, w):
    # x, y, w: [b, 1-3, h, w]
    diff = ((x - y).square())*w
    loss = diff.sum(dim=[1,2,3])/(w.sum(dim=[1,2,3])+1e-5)
    loss = loss.mean()
    return loss

def mix_pat_noise(noisy_pat, emnist):
    # scale = np.random.rand()*0.75 + 0.25 # [1/4, 4]
    scale = 10 / 15
    h, w = noisy_pat.shape[-2], noisy_pat.shape[-1]
    resize_h, resize_w = int(h*scale), int(w*scale)
    resize_emnist = F.interpolate(emnist, (resize_h, resize_w), mode='bilinear')
    start_h, start_w = int((h - resize_h)//2), int((w - resize_w)//2)
    comb = torch.zeros_like(noisy_pat)
    comb[:, :, start_h:start_h+resize_h, start_w:start_w+resize_w] = resize_emnist
    # alpha = 0.8 + 0.2 * torch.rand(noisy_pat.shape[0], 1, 1, 1).to(noisy_pat.device)
# BetaSampling(batch_size=noisy_pat.shape[0], device=noisy_pat.device, alpha=1.0)
    # print(scale)
    # print(alpha)
    # comb = emnist * alpha + (1-alpha) * noisy_pat
    # comb = (emnist * alpha + (1-alpha) * noisy_pat).detach()
    return comb.contiguous()

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
    args = parser.parse_args()
    return args

def train_one_epoch(model_forward, model, dataloader, optimizer, lr_scheduler, scaler,
                    device, epoch, print_interval=10, writer=None, log_file=None):
    model.train()
    running_loss = 0.0
    running_loss_cons = 0.0
    running_loss_mid = 0.0
    running_loss_emnist = 0.0
    model_forward.eval()
    for i, (signal, pat, _, emnist) in enumerate(dataloader):
        lr_scheduler.step()
        signal, pat = signal.to(device), pat.to(device)
        emnist = emnist.to(device)
        # emnist = scale_speckle(emnist).clamp(0,1)
        # print(signal.shape, pat.shape, signal.max(), signal.mean(), pat.max(), pat.min(), pat.mean())
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            comb_emnist = mix_pat_noise(pat, emnist)
            emnist_inp_pred, z_emnist = model(emnist)
            # comb_inp_pred, z_comb = model(comb_emnist)
            # print(emnist_inp_pred.shape)
            if i == 0:
                torchvision.utils.save_image(comb_emnist[:16], os.path.join("preds/stage2", "comb_emnist.png"))
            _, emnist_recon = model_forward(emnist_inp_pred, scale=4)
        mask = (emnist > 0.01).float().detach()
        # loss_emnist = F.l1_loss(emnist_recon, emnist)
        loss_emnist = weight_l1(emnist_recon, emnist, mask) + weight_l1(emnist_recon, emnist, 1-mask)
            # _, comb_recon = model_forward(comb_inp_pred, scale=2) 
            # loss_emnist = pearson_criterion(emnist_recon, emnist) # + 0.1*F.mse_loss(z_emnist, z_comb.detach())
            # loss_energy = F.mse_loss(emnist.std(dim=[1,2,3]), emnist_recon.std(dim=[1,2,3]))
        # loss_mid = cos_criterion(signal, signal_pred)
        # print(out_pred_norm.min(), out_pred_norm.max(), out_pred_norm.shape, pat.min(), pat.max())
        # loss_cons = l2_criterion(signal_pred.detach(), signal_pred_phi) + l2_criterion(signal_pred, signal_pred_phi.detach())
        # loss_contractive = contractive_criterion(model, pat)
        # scaler.scale(loss_mid + 0.1 * loss_cons).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # pdb.set_trace()
            # (loss_emnist).backward()
            # optimizer.step()
        scaler.scale(loss_emnist).backward()
        scaler.step(optimizer)
        scaler.update()
        # exit()
        
        # optimizer.step()
        # running_loss += loss.item()
        # running_loss_cons += loss_energy.item()
        # running_loss_mid += loss_mid.item()
        running_loss_emnist += loss_emnist.item()
        # running_loss_con += loss_contractive.item()
        if i % print_interval == 0:
            print(f"[epoch: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_interval:.5f}, loss_cons: {running_loss_cons / print_interval:.5f}, loss_mid: {running_loss_mid / print_interval:.5f}, loss_emnist: {running_loss_emnist / print_interval:.5f}")
            writer.add_scalar("train loss", running_loss / print_interval, epoch * len(dataloader) + i)
            writer.add_scalar("train loss consistency", running_loss_cons / print_interval, epoch * len(dataloader) + i)
            with open(log_file, "a") as f:
                f.write(f"[epoch: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_interval:.5f}\n, loss_cons: {running_loss_cons / print_interval:.5f}, loss_mid: {running_loss_mid / print_interval:.5f}, loss_emnist: {running_loss_emnist / print_interval:.5f}")
            running_loss = 0.0
            running_loss_cons = 0.0
            running_loss_mid = 0.0
            running_loss_inv = 0.0
            running_loss_emnist = 0.0

@torch.no_grad()
def validate(model_forward, model, val_loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for idx, (signal, pat, pat_name, emnist) in tqdm(enumerate(val_loader), ncols=60, desc="Validating"):
            signal, pat = signal.to(device), pat.to(device)
            emnist = emnist.to(device)
            # emnist = scale_speckle(emnist)
            signal_pred, _ = model(pat)
            # signal_pred_amp = torch.sqrt(torch.square(signal_pred).sum(dim=1, keepdim=True))
            # signal_pred_norm = F.normalize(signal_pred, dim=1)
            _, pat_pred = model_forward(signal_pred, scale=2)
            emnist_inp_pred, _ = model(emnist)
            # emnist_inp_pred = F.normalize(emnist_inp_pred, dim=1)
            _, emnist_recon = model_forward(emnist_inp_pred, scale=2)
            # emnist_recon = torch.sqrt(torch.square(emnist_recon).sum(dim=1, keepdim=True))
            
            if idx < 10:
                torchvision.utils.save_image(signal.cpu() / (2*torch.pi), os.path.join("preds/stage2", "inp_{}.png".format(idx+1)))
                torchvision.utils.save_image(pat[[0],...].cpu(), os.path.join("preds/stage2", "pat_{}.png".format(idx+1)))
                # torchvision.utils.save_image(signal[[0],[1],:,:].cpu(), os.path.join("preds", "imag_{}.png".format(idx+1)))
                torchvision.utils.save_image(tensor2phase(signal_pred).cpu(), os.path.join("preds/stage2", "inp_pred_{}.png".format(idx+1)))
                torchvision.utils.save_image(pat_pred[[0],...], os.path.join("preds/stage2", "pat_recon_{}.png".format(idx+1)))
                torchvision.utils.save_image(emnist_recon[[0],...], os.path.join("preds/stage2", "emnist_recon_{}.png".format(idx+1)))
                torchvision.utils.save_image(tensor2phase(emnist_inp_pred).cpu(), os.path.join("preds/stage2", "emnist_phase_pred_{}.png".format(idx+1)))
                torchvision.utils.save_image(emnist[[0],...], os.path.join("preds/stage2", "emnist_{}.png".format(idx+1)))
            psnrs.append(100 - 100*pearson_criterion(emnist, emnist_recon).item())
            # psnrs.append(psnr(emnist*255, emnist_recon*255).item())
            # psnrs.append(psnr(tensor2phase(signal)*255, tensor2phase(signal_pred)*255).item())
            # print(emnist.mean(), emnist.min(), emnist.max(), emnist_recon.mean(), emnist_recon.min(), emnist_recon.max())
    psnr_mean = np.mean(np.array(psnrs))
    return psnr_mean

def tensor2phase(x):
    phase = torch.atan2(x[:,[1],...], x[:,[0],...])
    # phase = (phase + torch.pi) / (2*torch.pi)
    # phase = torch.clamp(phase, 0, torch.pi) / torch.pi
    phase = torch.remainder(phase, 2*torch.pi) / (2*torch.pi)
    return phase

def load_checkpoint(model, optimizer, lr_scheduler, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    best_psnr = checkpoint["best_psnr"]
    return checkpoint["epoch"], best_psnr

def main(args):
    # set devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda") if len(args.gpu_ids) > 0 else torch.device("cpu")
    save_path = os.path.join(args.save_path, "stage2_enc48_4_dec32_4_8heads_patch10")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = os.path.join(save_path, "log.txt")
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    # save key files
    shutil.copy("layers/vit_size.py", os.path.join(save_path, "model.py"))
    shutil.copy("datasets/simulation_dataset.py", os.path.join(save_path, "dataset.py"))
    shutil.copy("main_stage2_simulation.py", os.path.join(save_path, "train.py"))
    shutil.copy("run.sh", os.path.join(save_path, "run.sh"))
    # build model
    model_forward = OpticsViTINR(image_size=args.phase_size, patch_size=5, enc_depth=4, dec_depth=4, heads=8, dim_head=32, dim=256, 
                      mlp_dim=int(256*8/3), in_channels=2, out_channels=1, act=torch.nn.Sigmoid, out_dim=384, use_learnable_pos=False, num_reg=0, drop_path_rate=0.0, pat_size=args.pat_size)
    model = OpticsViT(input_size=args.pat_size, input_patch_size=10, enc_depth=4, output_size=args.phase_size, output_patch_size=5, dec_depth=4, 
                      dim=384, heads=8, dim_head=48, mlp_dim=int(384*8/3), in_channels=1, out_channels=2, act=torch.nn.Tanh, out_norm=True, out_dim=256, drop_path_rate=0.0)

    model.to(device)
    # build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupCosineLR(optimizer, warmup_iters=1000, total_iters=args.num_epochs*18000//args.batch_size)
    scaler = GradScaler()
    # load forward model
    stage1_ckpt = torch.load(os.path.join(args.save_path, "stage1", "best.pth"), weights_only=False)
    model_forward.load_state_dict(stage1_ckpt)
    model_forward.freeze()
    model_forward.to(device)
    
    # resume from checkpoint
    if os.path.exists(args.resume_path):
        start_epoch, best_psnr = load_checkpoint(model, optimizer, lr_scheduler, scaler, args.resume_path)
    else:
        start_epoch, best_psnr = 0, 0.0
    # build dataloader
    train_dataset = OpticsDataset(train=True, root_dir=args.root_dir)
    val_dataset = OpticsDataset(train=False, root_dir=args.root_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=192, shuffle=False, num_workers=16)
    # train
    for epoch in range(start_epoch, args.num_epochs):
        # validate(model, val_loader, device)
        train_one_epoch(model_forward, model, train_loader, optimizer, lr_scheduler, scaler, device, epoch, print_interval=args.print_interval,
                        writer=writer, log_file=log_file)
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
        psnr_mean = validate(model_forward, model, val_loader, device)
        writer.add_scalar("val PSNR", psnr_mean, epoch + 1)
        print(f"[{epoch + 1}] val PSNR: {psnr_mean:.3f}\n")
        with open(log_file, "a") as f:
            f.write(f"[{epoch + 1}] val PSNR: {psnr_mean:.3f}\n")
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
    writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)