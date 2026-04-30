"""
Stage1: mimicking the physical system of transforming the image to optical pattern
Use torch mixed precision training to accelerate the training process
Ref: https://www.cnblogs.com/jimchen1218/p/14315008.html
"""
import os
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
import argparse
# PROJECT_ROOT = Path(__file__).resolve().parent
# DATASETS_DIR = PROJECT_ROOT / "datasets"
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# if str(DATASETS_DIR) not in sys.path:
#     sys.path.insert(0, str(DATASETS_DIR))
# try:
#     from .datasets.dataset import OpticsDataset
# except ImportError:
from datasets.dataset import OpticsDataset
# from datasets.Amp_dataset import OpticsDataset
from utils import psnr, ssim  # metrics
from tqdm import tqdm
import numpy as np
import torchvision
from scheduler import WarmupCosineLR
from layers.vit_inr import OpticsViTINR
import torch.nn.functional as F
import wandb
from utils import SSIM

    
pixel_criterion = torch.nn.L1Loss()  # pixel criterion
l2_criterion = torch.nn.MSELoss()
ssim_criterion = SSIM()

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
    parser.add_argument("--input_size", type=int, default=224, help="resize size for images and patterns")
    parser.add_argument("--pat_size", type=int, default=224, help="resize size for images and patterns")
    parser.add_argument("--exp_name", type=str, default="vit_inr")
    parser.add_argument("--input_scale", type=str, default="2_pi", help="scale factor for input images")
    parser.add_argument("--clip_speckle", type=float, default=1.0, help="clip value for speckle patterns")
    parser.add_argument("--input_channels", type=int, default=2, help="number of input channels for the model")
    args = parser.parse_args()
    return args

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, scaler,
                    device, epoch, print_interval=10, wandb_writer=None, args=None):
    model.train()
    running_loss = 0.0
    running_loss_inr = 0.0
    max_scale = args.pat_size / args.input_size
    for i, (signal, pat, _, _) in enumerate(dataloader):
        lr_scheduler.step()
        signal, pat = signal.to(device), pat.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            _, out_pred = model(signal, scale=max_scale)
            mean_loss = F.l1_loss(out_pred.mean(dim=[1,2,3]), pat.mean(dim=[1,2,3]))
            loss = F.l1_loss(out_pred, pat) + mean_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        running_loss_inr += mean_loss.item()
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
    max_scale = args.pat_size / args.input_size
    with torch.no_grad():
        for idx, (signal, pat, _, _) in tqdm(enumerate(val_loader), ncols=60, desc="Validating"):
            signal, pat = signal.to(device), pat.to(device)
            _, out_pred = model(signal, scale=max_scale)
            out_pred_norm = out_pred 
            out_pred_norm.clamp_(0.0, 1.0)
            # save first 8 samples for visualization
            if idx <= 8:
                torchvision.utils.save_image(signal[[0]].cpu()/(2*torch.pi), os.path.join("preds/phase", "inp_{}.png".format(idx+1)))
                torchvision.utils.save_image(pat[[0],...], os.path.join("preds/phase", "gt_{}.png".format(idx+1)))
                torchvision.utils.save_image(out_pred_norm[[0],...].cpu(), os.path.join("preds/phase", "pred_{}.png".format(idx+1)))
            # save first four samples of the first batch for wandb visualization
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
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
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

    # build dataloader
    train_dataset = OpticsDataset(train=True, root_dir=args.root_dir)
    val_dataset = OpticsDataset(train=False, root_dir=args.root_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
    
    # build model
    model = OpticsViTINR(image_size=args.input_size, patch_size=5, enc_depth=4, dec_depth=4, heads=8, dim_head=32, dim=256, 
                      mlp_dim=int(256*8/3), in_channels=args.input_channels, out_channels=1, act=torch.nn.Sigmoid, out_dim=384, use_learnable_pos=False, num_reg=0, drop_path_rate=0.1, input_pad=0, pat_size=args.pat_size)
    model = model.to(device)
    # print(model)
    # build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupCosineLR(optimizer, warmup_iters=1500, total_iters=args.num_epochs*len(train_dataset)//args.batch_size)
    scaler = GradScaler()
    # resume from checkpoint
    if os.path.exists(args.resume_path):
        print("Resume from: ", args.resume_path)
        start_epoch, best_psnr = load_checkpoint(model, optimizer, lr_scheduler, scaler, args.resume_path)
    else:
        start_epoch, best_psnr = 0, 0.0
    # train
    for epoch in range(start_epoch, args.num_epochs):
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
    # parse input scale
    if "pi" in args.input_scale.lower():
        s = args.input_scale.lower().split("_")[0]
        args.input_scale = float(s)*torch.pi
    else:
        args.input_scale = float(args.input_scale)
    main(args)
