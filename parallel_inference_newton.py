import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as tv_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from layers.vit_inr import OpticsViTINR
from utils import pad_image


INPUT_SIZE = 50
PAT_SIZE = 100
MODE = "4f_random_sigma_0_5_k_5_20k"
DEFAULT_CHECKPOINT = "checkpoints/4f_random_20k_sigma_0_5_k_5_0421_simulation/stage1/best.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-optimized Newton inference with async result saving."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--pat-size", type=int, default=PAT_SIZE)
    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-workers", type=int, default=4)
    parser.add_argument("--max-pending-saves", type=int, default=128)
    parser.add_argument("--mask-threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--input-channels", type=int, default=2, help="Number of input channels for the model (default: 2)")
    parser.add_argument("--input-mode", type=str, default="amp", choices=["amp", "phase"])
    parser.add_argument("--input_scale", type=float, default=1.0, help="Maximum phase value for normalization (default: 1.0)")
    return parser.parse_args()


def weight_l1(x, y, w):
    diff = (x - y).square() * w
    loss = diff.sum(dim=[1, 2, 3]) / (w.sum(dim=[1, 2, 3]) + 1e-5)
    return loss.mean()


class IndexedSubset(Dataset):
    def __init__(self, dataset, start_idx=0, num_samples=None):
        end_idx = len(dataset) if num_samples is None else min(len(dataset), start_idx + num_samples)
        self.dataset = dataset
        self.indices = list(range(start_idx, end_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, _ = self.dataset[real_idx]
        return img, real_idx


def build_loader(batch_size, pat_size, num_workers, start_idx, num_samples):
    img_transform = transforms.Compose(
        [
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
        ]
    )

    base_dataset = torchvision.datasets.EMNIST(
        root="data",
        download=False,
        split="letters",
        transform=img_transform,
    )
    subset = IndexedSubset(base_dataset, start_idx=start_idx, num_samples=num_samples)

    def collate_fn(batch):
        imgs = []
        indices = []
        for img, sample_idx in batch:
            img = pad_image(img.unsqueeze(0), pat_size, pat_size).squeeze(0)
            imgs.append(img.clamp(0, 1.0))
            indices.append(sample_idx)
        return torch.stack(imgs, dim=0), torch.tensor(indices, dtype=torch.long)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def build_model(input_size, pat_size, input_channels, checkpoint_path, device):
    model = OpticsViTINR(
        image_size=input_size,
        patch_size=5,
        enc_depth=4,
        dec_depth=4,
        heads=8,
        dim_head=32,
        dim=256,
        mlp_dim=int(256 * 8 / 3),
        in_channels=input_channels,
        out_channels=1,
        act=torch.nn.Sigmoid,
        out_dim=384,
        pat_size=pat_size,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class AsyncResultSaver:
    def __init__(self, save_pat_dir, save_pred_dir, save_input_dir, max_workers=4, max_pending=128):
        self.save_pat_dir = Path(save_pat_dir)
        self.save_pred_dir = Path(save_pred_dir)
        self.save_input_dir = Path(save_input_dir)
        self.max_pending = max_pending
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending = set()

    def _drain_if_needed(self):
        while len(self.pending) >= self.max_pending:
            done, self.pending = wait(self.pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()

    def submit(self, img, pred, input, sample_idx):
        self._drain_if_needed()
        future = self.executor.submit(
            self._save_single,
            img.contiguous(),
            pred.contiguous(),
            input.contiguous(),
            int(sample_idx),
        )
        self.pending.add(future)

    def _save_single(self, img, pred, input, sample_idx):
        sample_name = f"emnist_{sample_idx}"
        tv_utils.save_image(pred, self.save_pred_dir / f"{sample_name}.png")
        tv_utils.save_image(input, self.save_input_dir / f"{sample_name}.png")
        # np.save(self.save_input_dir / f"{sample_name}.npy", input[0].numpy())
        tv_utils.save_image(img, self.save_pat_dir / f"{sample_name}.png")

    def close(self):
        if self.pending:
            done, _ = wait(self.pending)
            for future in done:
                future.result()
        self.executor.shutdown(wait=True)


def optimize_batch(model, img_batch, num_steps, lr, input_size, pat_size, mask_threshold, use_amp, input_scale):
    batch_size = img_batch.shape[0]
    input_param = torch.nn.Parameter(
        torch.randn(batch_size, 1, input_size, input_size, device=img_batch.device)
    )
    optimizer = torch.optim.AdamW([input_param], lr=lr)
    mask = (img_batch > mask_threshold).float().detach()
    amp_enabled = use_amp and img_batch.device.type == "cuda"
    if amp_enabled and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    last_loss = None

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=img_batch.device.type, dtype=torch.float16, enabled=amp_enabled):
            input = torch.sigmoid(input_param) * input_scale
            _, pred_img = model(input, scale=pat_size / input_size)
            loss = weight_l1(pred_img, img_batch, mask) + weight_l1(pred_img, img_batch, 1 - mask)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        last_loss = loss.detach().item()

    with torch.no_grad():
        input_radi = torch.sigmoid(input_param)
        input_pse = input_radi * input_scale
        _, pat_recon = model(input_pse, scale=pat_size / input_size)

    return input_radi.detach(), pat_recon.detach(), last_loss


def main():
    args = parse_args()
    use_amp = args.use_amp or (torch.cuda.is_available() and not args.disable_amp)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        use_amp = False

    save_pat_dir = Path(f"newton_results/pat_{args.pat_size}_{args.mode}")
    save_pred_dir = Path(f"newton_results/preds_{args.pat_size}_{args.mode}")
    save_input_dir = Path(f"newton_results/{args.input_mode}_{args.pat_size}_{args.mode}")
    os.makedirs(save_pat_dir, exist_ok=True)
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_input_dir, exist_ok=True)

    loader = build_loader(
        batch_size=args.batch_size,
        pat_size=args.pat_size,
        num_workers=args.num_workers,
        start_idx=args.start_idx,
        num_samples=args.num_samples,
    )
    model = build_model(args.input_size, args.pat_size, args.input_channels, args.checkpoint, device)
    saver = AsyncResultSaver(
        save_pat_dir=save_pat_dir,
        save_pred_dir=save_pred_dir,
        save_input_dir=save_input_dir,
        max_workers=args.save_workers,
        max_pending=args.max_pending_saves,
    )

    progress = tqdm(loader, total=len(loader), ncols=100, desc="Batch inference")
    try:
        for img_batch, sample_indices in progress:
            img_batch = img_batch.to(device, non_blocking=True)
            input_radi, pat_recon, loss_value = optimize_batch(
                model=model,
                img_batch=img_batch,
                num_steps=args.num_steps,
                lr=args.lr,
                input_size=args.input_size,
                pat_size=args.pat_size,
                mask_threshold=args.mask_threshold,
                use_amp=use_amp,
                input_scale=args.input_scale,
            )

            img_cpu = img_batch.detach().cpu()
            pred_cpu = pat_recon.detach().cpu()
            input_cpu = input_radi.detach().cpu()
            for img, pred, input, sample_idx in zip(img_cpu, pred_cpu, input_cpu, sample_indices.tolist()):
                saver.submit(img, pred, input, sample_idx)

            progress.set_postfix(loss=f"{loss_value:.6f}")
    finally:
        saver.close()


if __name__ == "__main__":
    main()
