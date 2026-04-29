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
# from datasets.Phase_dataset import OpticsDataset
from datasets.Amp_dataset import OpticsDataset
from utils import psnr, ssim  # metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import torchvision
# from layers.vit import OpticsViT
from utils import mask_loss
from scheduler import WarmupCosineLR
# from layers.mlp import MLP
from layers.vit_inr import OpticsViTINR
import torch.nn.functional as F
import pdb
import shutil


pixel_criterion = torch.nn.L1Loss()  # pixel criterion
l2_criterion = torch.nn.MSELoss()

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

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, scaler,
                    device, epoch, print_interval=10, writer=None, log_file=None):
    model.train()
    running_loss = 0.0
    running_loss_inr = 0.0
    for i, (signal, pat, _, _) in enumerate(dataloader):
        lr_scheduler.step()
        # print(signal.min(), signal.max())
        signal, pat = signal.to(device), pat.to(device)
        # print(signal.shape, pat.shape)
        # print(signal.shape, pat.shape, signal.max(), signal.mean(), pat.max(), pat.min(), pat.mean())
        # scale = np.random.rand() + 1 # [1, 2]
        # resize_size = int(scale * signal.shape[-1])
        # pat_resize = F.interpolate(pat, (resize_size, resize_size), mode='bilinear').clamp(0, 1)
        # pat_0 = F.interpolate(pat, (signal.shape[-1], signal.shape[-1]), mode='bilinear').clamp(0, 1)
        # print(pat.shape)
        
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            _, out_pred = model(signal, scale=3.0)
        # out_0, out_inr = model(signal, scale=scale)

            loss = F.l1_loss(out_pred, pat)
            # loss_inr = l2_criterion(out_inr, pat_resize) + l2_criterion(out_0, pat_0)
            
            # print(out_pred.min().item(), out_pred.max().item(), pat.min(), pat.max().item())
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        running_loss += loss.item()
        # running_loss_inr += loss_inr.item()
        # pdb.set_trace()
        if i % print_interval == 0:
            print(f"[epoch: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_interval:.5f}, loss_inr: {running_loss_inr / print_interval:.5f}")
            writer.add_scalar("train loss", running_loss / print_interval, epoch * len(dataloader) + i)
            writer.add_scalar("train loss inr", running_loss_inr / print_interval, epoch * len(dataloader) + i)
            with open(log_file, "a") as f:
                f.write(f"[epoch: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_interval:.5f}\n, loss_inr: {running_loss_inr / print_interval:.5f}")
            running_loss = 0.0
            running_loss_inr = 0.0

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for idx, (signal, pat, pat_name, _) in tqdm(enumerate(val_loader), ncols=60, desc="Validating"):
            signal, pat = signal.to(device), pat.to(device)
            print(signal.min(), signal.max())
            _, out_pred = model(signal, scale=3.0)
            out_pred_norm = out_pred
            # out_pred_norm = torch.sqrt(torch.square(out_pred).sum(dim=1, keepdim=True))
            # print(l2_criterion(out_pred_norm, pat).item())
            out_pred_norm.clamp_(0.0, 1.0)
            if idx < 10:
                # tmp = (torch.atan2(signal[[0],[1]], signal[[0],[0]]) + torch.pi)/(2*torch.pi)
                torchvision.utils.save_image(signal.cpu(), os.path.join("preds/phase", "inp_{}.png".format(idx+1)))
                # torchvision.utils.save_image(signal[[0],[1],:,:].cpu(), os.path.join("preds", "imag_{}.png".format(idx+1)))
                torchvision.utils.save_image(out_pred_norm[[0],...].cpu(), os.path.join("preds/phase", "pred_{}.png".format(idx+1)))
                torchvision.utils.save_image(pat[[0],...], os.path.join("preds/phase", "gt_{}.png".format(idx+1)))
            psnrs.append(psnr(out_pred_norm*255, pat*255).item())
    psnr_mean = np.mean(np.array(psnrs))
    return psnr_mean

def load_checkpoint(model, optimizer, lr_scheduler, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    best_psnr = checkpoint["best_psnr"]
    epoch = checkpoint['epoch']
    # epoch, best_psnr = 0, 0
    return epoch, best_psnr # checkpoint["epoch"], best_psnr

def main(args):
    # set devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda") if len(args.gpu_ids) > 0 else torch.device("cpu")
    save_path = os.path.join(args.save_path, "stage1")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save key files
    shutil.copy("layers/vit_inr.py", os.path.join(save_path, "model.py"))
    shutil.copy("datasets/Amp_dataset.py", os.path.join(save_path, "dataset.py"))
    shutil.copy("main_stage1.py", os.path.join(save_path, "train.py"))
    shutil.copy("run.sh", os.path.join(save_path, "run.sh"))
    
    log_file = os.path.join(save_path, "log.txt")
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    # build model
    model = OpticsViTINR(image_size=args.phase_size, patch_size=5, enc_depth=4, dec_depth=4, heads=8, dim_head=32, dim=256, 
                      mlp_dim=int(256*8/3), in_channels=1, out_channels=1, act=torch.nn.Sigmoid, out_dim=384, use_learnable_pos=False, num_reg=0, drop_path_rate=0.1)
    model = model.to(device)
    print(model)
    # build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupCosineLR(optimizer, warmup_iters=1500, total_iters=args.num_epochs*18000//args.batch_size)
    scaler = GradScaler()
    # resume from checkpoint
    if os.path.exists(args.resume_path):
        print("=== resume from: ", args.resume_path)
        start_epoch, best_psnr = load_checkpoint(model, optimizer, lr_scheduler, scaler, args.resume_path)
    else:
        start_epoch, best_psnr = 0, 0.0
    # build dataloader
    train_dataset = OpticsDataset(train=True, root_dir=args.root_dir, input_size=args.pat_size)
    val_dataset = OpticsDataset(train=False, root_dir=args.root_dir, input_size=args.pat_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
    # train
    for epoch in range(start_epoch, args.num_epochs):
        # validate(model, val_loader, device)
        train_one_epoch(model, train_loader, optimizer, lr_scheduler, scaler, device, epoch, print_interval=args.print_interval,
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
        psnr_mean = validate(model, val_loader, device)
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