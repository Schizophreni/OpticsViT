import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


@torch.no_grad()
def get_correlation_length_torch(acf, nbins=200):
    """
    acf: (B, H, W), normalized s.t. acf(center)=1
    return: mean correlation length (pixels)
    """
    acf = acf.squeeze(1)
    B, H, W = acf.shape
    cy, cx = H // 2, W // 2

    y, x = torch.meshgrid(
        torch.arange(H, device=acf.device),
        torch.arange(W, device=acf.device),
        indexing='ij'
    )
    r = torch.sqrt((x - cx)**2 + (y - cy)**2)

    r_max = min(cx, cy)
    r_bins = torch.linspace(0, r_max, nbins + 1, device=acf.device)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    corr_lens = []

    for b in range(B):
        radial = torch.zeros(nbins, device=acf.device)

        for i in range(nbins):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if mask.any():
                radial[i] = acf[b][mask].mean()
            else:
                radial[i] = torch.nan

        # 去掉 nan
        valid = ~torch.isnan(radial)
        radial = radial[valid]
        r_centers_valid = r_centers[valid]

        # 找 0.5 crossing（从中心向外）
        below = radial <= 0.5
        if below.any():
            i = torch.where(below)[0][0]
            if i == 0:
                corr_lens.append(r_centers_valid[0])
            else:
                # 线性插值
                x0, x1 = r_centers_valid[i-1], r_centers_valid[i]
                y0, y1 = radial[i-1], radial[i]
                r_half = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
                corr_lens.append(r_half)
        else:
            corr_lens.append(r_centers_valid[-1])

    return torch.stack(corr_lens).mean().item()


# Example Usage:
# Assuming 'my_loader' provides (input_phase, target_speckle)
# for data_pi, target_pi in my_loader:
#    fwhm, _ = get_correlation_length_torch(target_pi)
#    print(f"Average Speckle Grain Size: {fwhm:.2f} pixels")

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def contractive_loss(model, x):
    x = x.clone().detach().requires_grad_(True)
    y = model(x, use_mask=False)
    grads = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return grads.pow(2).mean()


def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

def pad_image(image, height, width):
    h, w = image.shape[2], image.shape[3]
    pad_h, pad_w = (height - h) // 2, (width - w) // 2
    image = F.pad(image, pad=(pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    return image

def unpad_image(image, height, width):
    h, w = image.shape[2], image.shape[3]
    pad_h, pad_w = (h - height) // 2, (w - width) // 2
    assert pad_h >= 0 and pad_w >= 0, "unpad cannot run!"
    return image[:, :, pad_h:pad_h+height, pad_w:pad_w+width]
 
def compute_edge_mask(img):
    # Assuming grayscale [B, 1, H, W]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3).to(img.device)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    edge = torch.sqrt(gx**2 + gy**2 + 1e-6)
    return edge

class MaskedEdgeLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # Compute absolute difference
        edge_mask = compute_edge_mask(target)
        diff = torch.square(pred - target)

        # Apply edge mask
        edge_diff = diff * edge_mask

        if self.reduction == 'mean':
            return edge_diff.sum() / (edge_mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return edge_diff.sum()
        else:
            return edge_diff

def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2, dim=[1,2,3])
    score = - 10 * torch.log10(mse)
    score = score.mean()
    return score

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


# def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
#     x, y = x / data_range, y / data_range
#     # average pool image if the size is large enough
#     f = max(1, round(min(x.size()[-2:]) / 256))
#     if f > 1:
#         x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

#     # gaussian filter
#     coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
#     coords -= (kernel_size - 1) / 2.0
#     g = coords ** 2
#     g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
#     g /= g.sum()
#     kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

#     # compute
#     c1, c2 = k1 ** 2, k2 ** 2
#     n_channels = x.size(1)
#     mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
#     mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

#     mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
#     sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
#     sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
#     sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

#     # contrast sensitivity (CS) with alpha = beta = gamma = 1.
#     cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
#     # structural similarity (SSIM)
#     ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
#     return ss.mean()

def mask_loss(pred, target, eps=1e-5):
    mask = (target > eps).float().detach()
    loss = F.binary_cross_entropy_with_logits(pred, mask)
    return loss

def margin_loss(pred, margin=1.0):
    loss = torch.abs(pred) - margin
    loss = loss.clamp(min=0.0)
    mask = (loss > 0.0).float().detach()
    return loss.sum() / (mask.sum() + 1e-5)

def get_gaussian_kernel(kernel_size=5, sigma=1.0, device='cpu'):
    """Create a 2D Gaussian kernel"""
    # Create 1D Gaussian
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    ax = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = ax / ax.sum()

    # Outer product to get 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def apply_gaussian_blur(img: torch.Tensor, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to a batched tensor image.
    Args:
        img: Tensor of shape (B, C, H, W)
        kernel_size: int, size of the Gaussian kernel
        sigma: float, standard deviation of Gaussian
    Returns:
        Blurred image of shape (B, C, H, W)
    """
    B, C, H, W = img.shape
    device = img.device

    # Get Gaussian kernel and reshape to (1,1,k,k)
    kernel = get_gaussian_kernel(kernel_size, sigma, device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    # Expand kernel to apply to all channels
    kernel = kernel.repeat(C, 1, 1, 1)

    # Apply convolution (groups=C for depthwise conv)
    img_blur = F.conv2d(img, kernel, padding=kernel_size//2, groups=C)
    return img_blur

def weight_mask(img, ratio=0.9, max_importance=10.0, min_importance=1.0):
    """
    Batch version: computes importance weights per image in a batch.
    
    Args:
        img (Tensor): [B, 1, H, W] tensor.
        ratio (float): Desired importance mass ratio per image.
        max_importance (float): Importance value for selected region.
        min_importance (float): Importance value for remaining region.

    Returns:
        Tensor: Importance weight map of shape [B, 1, H, W].
    """
    B, C, H, W = img.shape
    flat = img.view(B, -1)  # [B, H*W]
    total = flat.sum(dim=1, keepdim=True)  # [B, 1]

    # Sort pixels in descending order
    sorted_vals, indices = torch.sort(flat, dim=1, descending=True)  # [B, H*W]
    cum_vals = torch.cumsum(sorted_vals, dim=1)  # [B, H*W]

    # Find threshold indices per batch element
    target = ratio * total  # [B, 1]
    k = torch.sum(cum_vals < target, dim=1) + 1  # [B]

    # Build the importance weight map
    weight_map = torch.full_like(flat, min_importance)  # [B, H*W]

    for i in range(B):
        weight_map[i, indices[i, :k[i]]] = max_importance

    # Reshape back to original shape
    return weight_map.view(B, C, H, W).to(img)

def criterionGAN(prediction, target=True):
    if target:
        loss = torch.square(prediction - 1.0).mean()
    else:
        loss = torch.square(prediction).mean()
    return loss

class PearsonLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PearsonLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        # Flatten if needed
        x = x.view(x.size(0), -1)  # [B, N]
        y = y.view(y.size(0), -1)  # [B, N]

        # Subtract mean
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean

        # Compute Pearson correlation
        numerator = (x_centered * y_centered).mean(dim=1)
        denominator = torch.sqrt((x_centered**2).mean(dim=1) * (y_centered**2).mean(dim=1)) + self.eps

        correlation = (numerator / denominator)  # shape: [B]
        # loss = -torch.log((1 + correlation)*0.5)        # scalar loss
        loss = 1 - correlation

        return loss.mean()

    
def corr(x, y):
    m1, m2 = x.mean(), y.mean()
    v12 = ((x - m1)*(y-m2)).mean()
    v1 = torch.square(x-m1).mean()
    v2 = torch.square(y-m2).mean()
    cov = v12 / (torch.sqrt(v1*v2) + 1e-5)
    return -torch.log((1+cov)*0.5)
    

def BetaSampling(alpha: float, batch_size: int, device=None, dtype=torch.float32):
    """
    Draw 'batch_size' samples λ ~ Beta(α, α) and return a tensor
    shaped for mix‑up broadcasting:  (B, 1, 1, 1).

    Parameters
    ----------
    alpha       β shape parameter (α > 0); α = 0.2 … 1.0 is common for mix‑up
    batch_size  number of samples to draw (usually your mini‑batch size)
    device      torch.device or None → infer from default tensor type
    dtype       floating dtype (default: torch.float32)

    Returns
    -------
    lam         tensor of shape (batch_size, 1, 1, 1)
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
    # reshape for broadcasting over [B, C, H, W]
    return lam.view(batch_size, 1, 1, 1)


def norm_call(x):
    """
    norm of a complex number of two channels
    """
    norm = torch.sqrt(torch.square(x).sum(dim=1, keepdim=True))
    return norm


def CosLoss(x1, x2, norm=False):
    if norm:
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
    sim = (x1 * x2).sum(dim=1).mean()
    return 1 - sim