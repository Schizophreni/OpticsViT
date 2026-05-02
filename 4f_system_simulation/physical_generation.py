import os
import math
import numpy as np
import torch
import torch.fft as fft
import cv2
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Optional, Tuple
import scipy.ndimage as ndimage
import argparse


def to_complex(x):
    return x.to(torch.complex64) if not torch.is_complex(x) else x


def H(dim, lmb, d, pixel_size, device):
    """
    生成自由空间传播的传递函数（Transfer Function）。
    使用角谱法（ASM），并包含带限（Band-limiting）以防止混叠。
    """
    df = 1.0 / (pixel_size * dim)

    fx = torch.linspace(-dim / 2, dim / 2 - 1, dim, device=device) * df
    fy = torch.linspace(-dim / 2, dim / 2 - 1, dim, device=device) * df
    fX, fY = torch.meshgrid(fx, fy, indexing="xy")
    fR2 = fX**2 + fY**2

    f_limit = 1.0 / (math.sqrt((2 * lmb * d * df) ** 2 + 1.0) * lmb)

    condition = (1.0 / lmb) ** 2 - fR2
    mask = (fR2 < (1.0 / lmb) ** 2).to(torch.complex64)

    kz = 2 * math.pi * torch.sqrt(torch.abs(condition))
    h = torch.exp(1j * kz * d) * mask

    limit_mask = (torch.sqrt(fR2) < f_limit).to(torch.complex64)
    return h * limit_mask


def lens_generator(dpi, pxl, lmb, f):
    """
    生成薄透镜的相位调制函数：exp(-i * k / 2f * (x^2 + y^2))
    """
    inner_radius = np.sqrt(f * 2 * lmb) / pxl
    H_dim, W = dpi

    x = np.linspace(1, W, W, dtype=np.float32) - W / 2
    y = np.linspace(1, H_dim, H_dim, dtype=np.float32) - H_dim / 2

    x2 = (x * x)[None, :]
    y2 = (y * y)[:, None]

    phase = 2 * np.pi * (x2 + y2) / (inner_radius ** 2)
    return phase % (2 * np.pi)


def a0_to_uz(a0, Hf):
    """
    频域相乘，计算传播后的场。
    Supports both (H, W) and (B, H, W) inputs.
    """
    az = a0 * Hf
    return fft.ifft2(az)


def diff(input_, Hf, dim_out=None):
    """
    执行一次衍射传播（单张）：FFT -> Multiply H -> IFFT
    Supports cropping output to dim_out.
    """
    dim_in = input_.shape[-1]
    dim_all = Hf.shape[-1]
    border_in = (dim_all - dim_in) // 2
    dim_out = dim_out or dim_in
    border_out = (dim_all - dim_out) // 2

    u0 = torch.zeros((dim_all, dim_all), dtype=torch.complex64, device=input_.device)
    u0[border_in:border_in + dim_in, border_in:border_in + dim_in] = input_

    a0 = fft.fft2(u0)
    uz = a0_to_uz(a0, Hf)

    if border_out > 0:
        return uz[border_out:-border_out, border_out:-border_out]
    return uz


def diff_batch(input_batch, Hf, dim_out=None):
    """
    Batched diffraction propagation: FFT -> Multiply H -> IFFT.
    input_batch: (B, H, W) complex tensor
    Hf:          (dim_all, dim_all) transfer function (shared across batch)
    Returns:     (B, dim_out, dim_out) complex tensor
    """
    B, dim_in, _ = input_batch.shape
    dim_all = Hf.shape[-1]
    border_in = (dim_all - dim_in) // 2
    dim_out = dim_out or dim_in
    border_out = (dim_all - dim_out) // 2

    # Fast path: when the input already matches the simulation grid, avoid an
    # extra zero-padded buffer allocation and directly propagate in place.
    if border_in == 0 and border_out == 0:
        a0 = fft.fft2(input_batch)
        az = a0 * Hf.unsqueeze(0)
        return fft.ifft2(az)

    # Zero-pad each sample in the batch
    u0 = torch.zeros((B, dim_all, dim_all), dtype=torch.complex64, device=input_batch.device)
    u0[:, border_in:border_in + dim_in, border_in:border_in + dim_in] = input_batch

    a0 = fft.fft2(u0)                         # (B, dim_all, dim_all)
    # Hf is already fft-shift corrected and broadcast over batch dim
    az = a0 * Hf.unsqueeze(0)  # (B, dim_all, dim_all)
    uz = fft.ifft2(az)                         # (B, dim_all, dim_all)

    if border_out > 0:
        return uz[:, border_out:-border_out, border_out:-border_out]
    return uz


def debug_propagation_4f(
    input_field, phase1, phase2,
    f1, f2, f3, d1, d2, d3, d4, wavelength, pixel_size
) -> Dict[str, torch.Tensor]:
    """
    核心物理模拟函数（单张）：带有两个相位掩膜的 4f 系统。
    """
    device = input_field.device
    N = input_field.shape[-1]

    lens1_phi = lens_generator((N, N), pixel_size, wavelength, f1)
    lens2_phi = lens_generator((N, N), pixel_size, wavelength, f2)
    lens3_phi = lens_generator((N, N), pixel_size, wavelength, f3)

    lens1 = torch.exp(-1j * torch.tensor(lens1_phi, device=device))
    lens2 = torch.exp(-1j * torch.tensor(lens2_phi, device=device))
    lens3 = torch.exp(-1j * torch.tensor(lens3_phi, device=device))

    H_f1 = H(N, wavelength, f1, pixel_size, device)
    u1 = diff(input_field, H_f1) * lens1

    H_d1 = H(N, wavelength, d1, pixel_size, device)
    u2 = diff(u1, H_d1)

    ph1 = torch.tensor(phase1, device=device)
    ph2 = torch.tensor(phase2, device=device)

    ph1_H = ph1.shape[1]
    if ph1_H != N:
        start = (ph1_H - N) // 2
        ph1 = ph1[start:start + N, start:start + N]

    ph2_H = ph2.shape[1]
    if ph2_H != N:
        start = (ph2_H - N) // 2
        ph2 = ph2[start:start + N, start:start + N]

    ph1 = torch.exp(-1j * ph1)
    ph2 = torch.exp(-1j * ph2)

    u2 *= ph1

    H_m1_m2 = H(N, wavelength, d2 - d1, pixel_size, device)
    u3 = diff(u2, H_m1_m2)
    u3 *= ph2

    H_m2_l2 = H(N, wavelength, (f1 + f2) - d2, pixel_size, device)
    u4 = diff(u3, H_m2_l2) * lens2

    H_l2_l3 = H(N, wavelength, d3, pixel_size, device)
    u5 = diff(u4, H_l2_l3) * lens3

    H_f2 = H(N, wavelength, d4, pixel_size, device)
    u_sensor = diff(u5, H_f2, dim_out=N)

    return {"sensor": u_sensor}


def debug_propagation_4f_batch(
    input_batch, phase1, phase2,
    f1, f2, f3, d1, d2, d3, d4, wavelength, pixel_size
) -> Dict[str, torch.Tensor]:
    """
    Batched 4f system simulation.
    input_batch: (B, N, N) complex tensor
    phase1/phase2: numpy arrays, pre-loaded and padded (dim_all, dim_all)
    Returns dict with 'sensor' key -> (B, N, N) intensity-ready complex tensor.
    """
    device = input_batch.device
    B, N, _ = input_batch.shape

    # --- Build shared optical elements (computed once per batch) ---
    lens1_phi = lens_generator((N, N), pixel_size, wavelength, f1)
    lens2_phi = lens_generator((N, N), pixel_size, wavelength, f2)
    lens3_phi = lens_generator((N, N), pixel_size, wavelength, f3)

    lens1 = torch.exp(-1j * torch.tensor(lens1_phi, device=device))  # (N, N)
    lens2 = torch.exp(-1j * torch.tensor(lens2_phi, device=device))
    lens3 = torch.exp(-1j * torch.tensor(lens3_phi, device=device))

    H_f1    = H(N, wavelength, f1,             pixel_size, device)
    H_d1    = H(N, wavelength, d1,             pixel_size, device)
    H_m1_m2 = H(N, wavelength, d2 - d1,       pixel_size, device)
    H_m2_l2 = H(N, wavelength, (f1 + f2) - d2, pixel_size, device)
    H_l2_l3 = H(N, wavelength, d3,             pixel_size, device)
    H_f2    = H(N, wavelength, d4,             pixel_size, device)

    # --- Phase masks ---
    ph1 = torch.tensor(phase1, device=device)
    ph2 = torch.tensor(phase2, device=device)

    # Crop to N if masks are larger
    if ph1.shape[0] != N:
        s = (ph1.shape[0] - N) // 2
        ph1 = ph1[s:s + N, s:s + N]
    if ph2.shape[0] != N:
        s = (ph2.shape[0] - N) // 2
        ph2 = ph2[s:s + N, s:s + N]

    ph1 = torch.exp(-1j * ph1)  # (N, N)
    ph2 = torch.exp(-1j * ph2)  # (N, N)

    # --- Propagation (all ops broadcast over batch dim B) ---
    # lens and mask tensors are (N, N); unsqueeze to (1, N, N) for broadcasting
    u1 = diff_batch(input_batch, H_f1) * lens1.unsqueeze(0)         # (B, N, N)
    u2 = diff_batch(u1, H_d1) * ph1.unsqueeze(0)                    # (B, N, N)
    u3 = diff_batch(u2, H_m1_m2) * ph2.unsqueeze(0)                 # (B, N, N)
    u4 = diff_batch(u3, H_m2_l2) * lens2.unsqueeze(0)               # (B, N, N)
    u5 = diff_batch(u4, H_l2_l3) * lens3.unsqueeze(0)               # (B, N, N)
    u_sensor = diff_batch(u5, H_f2, dim_out=N)                       # (B, N, N)

    return {"sensor": u_sensor}


def center_pad_to(arr, target):
    out = np.zeros((target, target), dtype=arr.dtype)
    h, w = arr.shape
    c = target // 2
    out[c - h // 2:c + h // 2, c - w // 2:c + w // 2] = arr
    return out


def norm(x):
    if x.max() == x.min():
        return np.zeros_like(x)
    return np.clip(x / 60, 0, 1)


def load_resized_grayscale(
    path: str,
    inner_size: int,
    interpolation: int = cv2.INTER_NEAREST,
) -> Tuple[np.ndarray, str]:
    """
    Load and resize a single grayscale image to the target inner size.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    img = cv2.resize(img, (inner_size, inner_size), interpolation=interpolation)
    base = os.path.splitext(os.path.basename(path))[0]
    return img, base


def build_input_batch(
    batch_paths: List[str],
    inner_size: int,
    sim_size: int,
    insert_offset: int,
    aperture_mask: np.ndarray,
    is_phase_input: bool,
    phase_lut: Optional[np.ndarray],
    load_pool: ThreadPoolExecutor,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a full complex64 input batch on CPU, using threaded image loading and
    a phase lookup table to avoid per-pixel exp() calls.
    """
    loader = partial(load_resized_grayscale, inner_size=inner_size, interpolation=cv2.INTER_NEAREST)
    loaded = list(load_pool.map(loader, batch_paths))

    batch_size = len(loaded)
    img_batch = np.zeros((batch_size, sim_size, sim_size), dtype=np.uint8)
    bases: List[str] = []

    y0 = insert_offset
    y1 = insert_offset + inner_size
    for idx, (img, base) in enumerate(loaded):
        img_batch[idx, y0:y1, y0:y1] = img
        bases.append(base)

    if is_phase_input:
        if phase_lut is None:
            raise ValueError("phase_lut must be provided when is_phase_input=True")
        field_batch = phase_lut[img_batch]
        field_batch *= aperture_mask[None, ...]
    else:
        amp_batch = img_batch.astype(np.float32) / 255.0
        field_batch = amp_batch.astype(np.complex64)
        field_batch *= aperture_mask[None, ...]

    return field_batch, bases


def save_sensor_outputs(
    base: str,
    sensor: np.ndarray,
    out_dir: str,
    crop_slice: slice,
) -> Tuple[str, str]:
    sensor_crop = sensor[crop_slice, crop_slice]
    out_npy = os.path.join(out_dir, f"{base}_sensor.npy")
    out_png = os.path.join(out_dir, f"{base}_sensor.png")
    np.save(out_npy, sensor_crop)
    cv2.imwrite(out_png, (norm(sensor_crop) * 255).astype(np.uint8))
    return out_png, out_npy


def process_image(
    path, phase1_path, phase2_path,
    tile_repeat, pad_scale,
    pixel_size, wavelength,
    f1, f2, f3, d1, d2, d3, d4,
    times, is_phase_input, phase_max,
    out_dir, device="cuda"
):
    """Original single-image processing (unchanged)."""
    inner_size = 50 * tile_repeat
    H1 = int(inner_size * pad_scale)
    insert_offset = (H1 - inner_size) // 2
    center = H1 // 2
    radius = int(400 * 210 / (127.7 * 2))
    y_g, x_g = np.ogrid[:H1, :H1]
    aperture_mask = (
        np.sqrt((x_g - center) ** 2 + (y_g - center) ** 2) <= radius
    ).astype(np.complex64)

    img, base = load_resized_grayscale(path, inner_size=inner_size, interpolation=cv2.INTER_NEAREST)
    img_pad = np.zeros((H1, H1), dtype=np.uint8)
    img_pad[insert_offset:insert_offset + inner_size, insert_offset:insert_offset + inner_size] = img
    if is_phase_input:
        phase_values = (np.arange(256, dtype=np.float32) / 255.0) * phase_max
        phase_lut = np.exp(1j * phase_values).astype(np.complex64)
        field_np = phase_lut[img_pad] * aperture_mask
    else:
        field_np = (img_pad.astype(np.float32) / 255.0).astype(np.complex64) * aperture_mask

    input_field = torch.tensor(field_np, device=device, dtype=torch.complex64)

    p1_smooth = ndimage.gaussian_filter(np.load(phase1_path), sigma=0.5)
    p2_smooth = ndimage.gaussian_filter(np.load(phase2_path), sigma=0.5)
    p1 = center_pad_to(p1_smooth * 2 * np.pi, H1 * times)
    p2 = center_pad_to(p2_smooth * 2 * np.pi, H1 * times)

    result = debug_propagation_4f(
        input_field, p1, p2,
        f1, f2, f3, d1, d2, d3, d4,
        wavelength, pixel_size
    )

    sensor = torch.abs(result["sensor"]) ** 2
    sensor = sensor.cpu().numpy()

    c = sensor.shape[0] // 2
    vs = 400
    sensor = sensor[c - vs // 2:c + vs // 2, c - vs // 2:c + vs // 2]

    out_npy = os.path.join(out_dir, f"{base}_sensor.npy")
    out_png = os.path.join(out_dir, f"{base}_sensor.png")
    np.save(out_npy, sensor)
    cv2.imwrite(out_png, (norm(sensor) * 255).astype(np.uint8))
    return out_png, out_npy


def batch_generation(
    file_list: List[str],
    phase1_path: str,
    phase2_path: str,
    out_dir: str,
    tile_repeat: int = 8,
    pad_scale: float = 8.0,
    pixel_size: float = 8e-6,
    wavelength: float = 660e-9,
    f1: float = 200e-3,
    f2: float = 100e-3,
    f3: float = 100e-3,
    d1: float = 50e-3,
    d2: float = 100e-3,
    d3: float = 425e-3,
    d4: float = 140e-3,
    times: int = 8,
    is_phase_input: bool = True,
    phase_max: float = np.pi * 2,
    batch_size: int = 8,
    device: str = "cuda",
    crop_size: int = 400,
) -> List[Tuple[str, str]]:
    """
    Process a list of image files through the 4f optical system in batches.

    Compared to calling process_image() in a loop, this function:
      - Stacks multiple input fields into a (B, N, N) tensor
      - Runs a single batched forward pass (diff_batch) per propagation step
      - Amortises FFT and phase-mask overhead across the batch
      - Keeps transfer functions and phase masks on GPU across batches

    Parameters
    ----------
    file_list   : List of image file paths to process.
    phase1_path : Path to the .npy file for phase mask 1.
    phase2_path : Path to the .npy file for phase mask 2.
    out_dir     : Directory to write output .png and .npy files.
    batch_size  : Number of images processed per GPU forward pass.
                  Reduce if you run out of VRAM.
    device      : 'cuda' (recommended) or 'cpu'.
    crop_size   : Half-width of the sensor crop window (pixels on each side
                  of centre); default 400 → 400×400 output.
    (all other parameters match process_image / main() defaults)

    Returns
    -------
    List of (png_path, npy_path) tuples in the same order as file_list.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-load and pre-process phase masks (done once for all batches)
    # ------------------------------------------------------------------
    p1_raw = ndimage.gaussian_filter(np.load(phase1_path), sigma=0.5)
    p2_raw = ndimage.gaussian_filter(np.load(phase2_path), sigma=0.5)

    # We need the spatial size of one padded input to know the final mask size.
    # Derive it from the first file (all images are resized to 50×50 then tiled).
    _sample_img = cv2.imread(file_list[0], cv2.IMREAD_GRAYSCALE)
    _sample_img = cv2.resize(_sample_img, (50, 50))
    inner_size = _sample_img.shape[0] * tile_repeat  # after tiling
    H1 = int(inner_size * pad_scale)                 # after padding
    N = H1                                           # simulation grid size
    insert_offset = (H1 - inner_size) // 2

    center = H1 // 2
    radius = int(400 * 210 / (127.7 * 2))
    y_g, x_g = np.ogrid[:H1, :H1]
    aperture_mask = (
        np.sqrt((x_g - center) ** 2 + (y_g - center) ** 2) <= radius
    ).astype(np.complex64)
    crop_slice = slice(center - crop_size // 2, center + crop_size // 2)

    p1_np = center_pad_to(p1_raw * 2 * np.pi, H1 * times) 
    p2_np = center_pad_to(p2_raw * 2 * np.pi, H1 * times)

    # Move masks to GPU as tensors; they are reused across every batch
    ph1_t = torch.tensor(p1_np, device=device, dtype=torch.float32)
    ph2_t = torch.tensor(p2_np, device=device, dtype=torch.float32)

    # Crop phase masks to N if they are larger (mirrors original logic)
    if ph1_t.shape[0] != N:
        s = (ph1_t.shape[0] - N) // 2
        ph1_t = ph1_t[s:s + N, s:s + N]
    if ph2_t.shape[0] != N:
        s = (ph2_t.shape[0] - N) // 2
        ph2_t = ph2_t[s:s + N, s:s + N]

    ph1_exp = torch.exp(-1j * ph1_t)   # (N, N) complex — kept on GPU
    ph2_exp = torch.exp(-1j * ph2_t)

    # ------------------------------------------------------------------
    # Pre-compute all transfer functions (done once, shared across batches)
    # ------------------------------------------------------------------
    H_f1    = fft.ifftshift(H(N, wavelength, f1,              pixel_size, device))
    H_d1    = fft.ifftshift(H(N, wavelength, d1,              pixel_size, device))
    H_m1_m2 = fft.ifftshift(H(N, wavelength, d2 - d1,        pixel_size, device))
    H_m2_l2 = fft.ifftshift(H(N, wavelength, (f1 + f2) - d2, pixel_size, device))
    H_l2_l3 = fft.ifftshift(H(N, wavelength, d3,             pixel_size, device))
    H_f2    = fft.ifftshift(H(N, wavelength, d4,              pixel_size, device))

    # Lenses (also shared)
    def _lens(f):
        phi = lens_generator((N, N), pixel_size, wavelength, f)
        return torch.exp(-1j * torch.tensor(phi, device=device))

    lens1 = _lens(f1)
    lens2 = _lens(f2)
    lens3 = _lens(f3)

    # ------------------------------------------------------------------
    # Process files in chunks of batch_size
    # ------------------------------------------------------------------
    results: List[Tuple[str, str]] = []
    n_total = len(file_list)
    save_pool = ThreadPoolExecutor(max_workers=min(4, max(1, batch_size)))
    load_pool = ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 1)))
    pending_saves = []
    phase_lut = None
    if is_phase_input:
        phase_values = (np.arange(256, dtype=np.float32) / 255.0) * phase_max
        phase_lut = np.exp(1j * phase_values).astype(np.complex64)

    try:
        for batch_start in range(0, n_total, batch_size):
            batch_paths = file_list[batch_start: batch_start + batch_size]
            B = len(batch_paths)

            # --- Prepare input fields for this batch ---
            field_batch_np, bases = build_input_batch(
                batch_paths=batch_paths,
                inner_size=inner_size,
                sim_size=N,
                insert_offset=insert_offset,
                aperture_mask=aperture_mask,
                is_phase_input=is_phase_input,
                phase_lut=phase_lut,
                load_pool=load_pool,
            )

            # Move stacked (B, N, N) batch to device
            input_batch = torch.as_tensor(
                field_batch_np, device=device, dtype=torch.complex64
            )   # (B, N, N)

            # --- Batched forward pass ---
            with torch.inference_mode():
                u1 = diff_batch(input_batch, H_f1) * lens1.unsqueeze(0)
                u2 = diff_batch(u1, H_d1)          * ph1_exp.unsqueeze(0)
                u3 = diff_batch(u2, H_m1_m2)       * ph2_exp.unsqueeze(0)
                u4 = diff_batch(u3, H_m2_l2)       * lens2.unsqueeze(0)
                u5 = diff_batch(u4, H_l2_l3)       * lens3.unsqueeze(0)
                u_sensor = diff_batch(u5, H_f2, dim_out=N)   # (B, N, N)

            # Compute intensity on GPU, then bring to CPU
            intensity = torch.abs(u_sensor) ** 2   # (B, N, N)
            intensity_np = intensity.cpu().numpy()  # avoid repeated .cpu() calls

            # --- Save each sample's output ---
            for i, base in enumerate(bases):
                pending_saves.append(
                    save_pool.submit(
                        save_sensor_outputs,
                        base=base,
                        sensor=intensity_np[i],
                        out_dir=out_dir,
                        crop_slice=crop_slice,
                    )
                )

            while len(pending_saves) > 2 * batch_size:
                results.append(pending_saves.pop(0).result())

            print(f"[batch_generation] {min(batch_start + batch_size, n_total)}/{n_total} done")
    finally:
        for future in pending_saves:
            results.append(future.result())
        save_pool.shutdown(wait=True)
        load_pool.shutdown(wait=True)

    return results

def get_args():
    parser = argparse.ArgumentParser(description="4f System Simulation with Batched Processing")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--phase1_path", type=str, required=True, help="Path to phase mask 1 (.npy)")
    parser.add_argument("--phase2_path", type=str, required=True, help="Path to phase mask 2 (.npy)")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output .png and .npy files")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process in each batch")
    parser.add_argument("--phase_max", type=str, default="2_pi")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = get_args()
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir
    # input_dir = "input_perlin_20k"

    # Recommended pair from slm_phase_generation_freq.py:
    # layer1 mainly scatters, layer2 partially compensates and refocuses to centre.
    phase1 = args.phase1_path
    phase2 = args.phase2_path

    out_dir = os.path.join(args.out_dir)

    tile_repeat    = 8
    pad_scale      = 8
    pixel_size     = 8e-6
    wavelength     = 660e-9
    times          = 8
    is_phase_input = True
    if "pi" in args.phase_max:
        scale = float(args.phase_max.split("_")[0])
        phase_max = scale * np.pi
    else:
        phase_max = float(args.phase_max)

    f1, f2, f3 = 200e-3, 100e-3, 100e-3
    d1, d2, d3, d4 = 50e-3, 100e-3, 425e-3, 144.44e-3 # 140e-3

    os.makedirs(out_dir, exist_ok=True)
    valid_ext = {'.png', '.jpg', '.jpeg'}
    files = [
        os.path.join(root, f)
        for root, _, fnames in os.walk(input_dir)
        for f in fnames
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    if not files:
        print('No images found in', input_dir)
        return

    print(f"Found {len(files)} images.")

    # -----------------------------------------------------------------------
    # Use batch_generation for fast GPU-parallel processing.
    # Tune batch_size to fit your GPU VRAM (start with 8, increase if possible).
    # -----------------------------------------------------------------------
    results = batch_generation(
        file_list=sorted(files),
        phase1_path=phase1,
        phase2_path=phase2,
        out_dir=out_dir,
        tile_repeat=tile_repeat,
        pad_scale=pad_scale,
        pixel_size=pixel_size,
        wavelength=wavelength,
        f1=f1, f2=f2, f3=f3,
        d1=d1, d2=d2, d3=d3, d4=d4,
        times=times,
        is_phase_input=is_phase_input,
        phase_max=phase_max,
        batch_size=32,        # ← adjust to your GPU
        device="cuda:0",
        crop_size=400,
    )

    for png, npy in results:
        print(f"Saved: {os.path.basename(png)}")


if __name__ == '__main__':
    main()
