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
from layers.vit_size import OpticsViT


INPUT_SIZE = 50
PAT_SIZE = 200
MODE = "4f_phase_pi"
DEFAULT_STAGE2_CHECKPOINT = (
    "checkpoints/4f_twophases_pi_200/stage2_enc48_4_dec32_4_8heads_patch10/best.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch two-stage inference with optional async reconstruction saving."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", "--phase-size", dest="input_size", type=int, default=INPUT_SIZE)
    parser.add_argument("--pat-size", type=int, default=PAT_SIZE)
    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--stage2-checkpoint", type=str, default=DEFAULT_STAGE2_CHECKPOINT)
    parser.add_argument("--recon-checkpoint", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-workers", type=int, default=4)
    parser.add_argument("--max-pending-saves", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--save-pat", action="store_true")
    parser.add_argument("--save-pred", action="store_true")
    parser.add_argument("--input-mode", type=str, default="phase", choices=["phase", "amp"])
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--input-scale", type=float, default=torch.pi)
    return parser.parse_args()


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
            transforms.Resize((pat_size, pat_size)),
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


def build_inverse_model(pat_size, input_size, checkpoint_path, device):
    model = OpticsViT(
        input_size=pat_size,
        input_patch_size=10,
        enc_depth=4,
        output_size=input_size,
        output_patch_size=5,
        dec_depth=4,
        dim=384,
        heads=8,
        dim_head=48,
        mlp_dim=int(384 * 8 / 3),
        in_channels=1,
        out_channels=2,
        act=torch.nn.Tanh,
        out_norm=True,
        out_dim=256,
        drop_path_rate=0.0,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_forward_model(input_size, pat_size, input_channels, checkpoint_path, device):
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
    def __init__(
        self,
        save_input_dir,
        save_pat_dir=None,
        save_pred_dir=None,
        max_workers=4,
        max_pending=128,
    ):
        self.save_input_dir = Path(save_input_dir)
        self.save_pat_dir = Path(save_pat_dir) if save_pat_dir is not None else None
        self.save_pred_dir = Path(save_pred_dir) if save_pred_dir is not None else None
        self.max_pending = max_pending
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending = set()

    def _drain_if_needed(self):
        while len(self.pending) >= self.max_pending:
            done, self.pending = wait(self.pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()

    def submit(self, input_tensor, sample_idx, img=None, pred=None):
        self._drain_if_needed()
        future = self.executor.submit(
            self._save_single,
            input_tensor.contiguous(),
            int(sample_idx),
            None if img is None else img.contiguous(),
            None if pred is None else pred.contiguous(),
        )
        self.pending.add(future)

    def _save_single(self, input_tensor, sample_idx, img=None, pred=None):
        sample_name = f"emnist_{sample_idx}"
        tv_utils.save_image(input_tensor, self.save_input_dir / f"{sample_name}.png")
        np.save(self.save_input_dir / f"{sample_name}.npy", input_tensor[0].numpy())
        if self.save_pat_dir is not None and img is not None:
            tv_utils.save_image(img, self.save_pat_dir / f"{sample_name}.png")
        if self.save_pred_dir is not None and pred is not None:
            tv_utils.save_image(pred, self.save_pred_dir / f"{sample_name}.png")

    def close(self):
        if self.pending:
            done, _ = wait(self.pending)
            for future in done:
                future.result()
        self.executor.shutdown(wait=True)


def predict_input_batch(model, img_batch, input_mode, input_scale, use_amp):
    amp_enabled = use_amp and img_batch.device.type == "cuda"
    with torch.autocast(device_type=img_batch.device.type, dtype=torch.float16, enabled=amp_enabled):
        model_output, _ = model(img_batch)

    if input_mode == "phase":
        input_tensor = torch.atan2(model_output[:, [1]], model_output[:, [0]] + 1e-7)
        input_tensor = torch.remainder(input_tensor, input_scale) / input_scale
    else:
        input_tensor = torch.sqrt(torch.square(model_output).sum(dim=1, keepdim=True)).clamp(0.0, 1.0)

    return model_output.detach(), input_tensor.detach()


def reconstruct_pattern_batch(model, input_tensor, input_mode, input_size, pat_size, use_amp):
    amp_enabled = use_amp and input_tensor.device.type == "cuda"
    if input_mode == "phase":
        model_input = input_tensor * (2 * torch.pi)
    else:
        model_input = input_tensor
    with torch.autocast(device_type=model_input.device.type, dtype=torch.float16, enabled=amp_enabled):
        _, pred_batch = model(model_input, scale=pat_size / input_size)
    return pred_batch.detach()


def main():
    args = parse_args()
    if args.save_pred and args.recon_checkpoint is None:
        raise ValueError("--save-pred requires --recon-checkpoint.")

    use_amp = args.use_amp or (torch.cuda.is_available() and not args.disable_amp)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        use_amp = False
    input_channels = args.input_channels
    if input_channels is None:
        input_channels = 2 if args.input_mode == "phase" else 1

    save_pat_dir = Path(f"two_stage_results/pat_{args.pat_size}_{args.mode}")
    save_pred_dir = Path(f"two_stage_results/preds_{args.pat_size}_{args.mode}")
    save_input_dir = Path(f"two_stage_results/{args.input_mode}_{args.pat_size}_{args.mode}")
    os.makedirs(save_input_dir, exist_ok=True)
    if args.save_pat:
        os.makedirs(save_pat_dir, exist_ok=True)
    if args.save_pred:
        os.makedirs(save_pred_dir, exist_ok=True)

    loader = build_loader(
        batch_size=args.batch_size,
        pat_size=args.pat_size,
        num_workers=args.num_workers,
        start_idx=args.start_idx,
        num_samples=args.num_samples,
    )
    inverse_model = build_inverse_model(
        pat_size=args.pat_size,
        input_size=args.input_size,
        checkpoint_path=args.stage2_checkpoint,
        device=device,
    )
    forward_model = None
    if args.recon_checkpoint is not None:
        forward_model = build_forward_model(
            input_size=args.input_size,
            pat_size=args.pat_size,
            input_channels=input_channels,
            checkpoint_path=args.recon_checkpoint,
            device=device,
        )

    saver = AsyncResultSaver(
        save_input_dir=save_input_dir,
        save_pat_dir=save_pat_dir if args.save_pat else None,
        save_pred_dir=save_pred_dir if args.save_pred else None,
        max_workers=args.save_workers,
        max_pending=args.max_pending_saves,
    )

    progress = tqdm(loader, total=len(loader), ncols=100, desc="Two-stage inference")
    try:
        with torch.inference_mode():
            for img_batch, sample_indices in progress:
                img_batch = img_batch.to(device, non_blocking=True)
                _, input_tensor = predict_input_batch(
                    model=inverse_model,
                    img_batch=img_batch,
                    input_mode=args.input_mode,
                    input_scale=args.input_scale,
                    use_amp=use_amp,
                )

                pred_batch = None
                if forward_model is not None:
                    pred_batch = reconstruct_pattern_batch(
                        model=forward_model,
                        input_tensor=input_tensor,
                        input_mode=args.input_mode,
                        input_size=args.input_size,
                        pat_size=args.pat_size,
                        use_amp=use_amp,
                    )

                input_cpu = input_tensor.cpu()
                img_cpu = img_batch.cpu() if args.save_pat else None
                pred_cpu = pred_batch.cpu() if pred_batch is not None else None

                for batch_offset, sample_idx in enumerate(sample_indices.tolist()):
                    saver.submit(
                        input_tensor=input_cpu[batch_offset],
                        sample_idx=sample_idx,
                        img=None if img_cpu is None else img_cpu[batch_offset],
                        pred=None if pred_cpu is None else pred_cpu[batch_offset],
                    )
    finally:
        saver.close()


if __name__ == "__main__":
    main()
