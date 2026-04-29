import torch
from torch import nn
from einops import rearrange
from timm.layers import trunc_normal_
import numpy as np
import torch.nn.functional as F
from timm.models.layers import drop_path
from siren import SIREN

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# define basic attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ConvFeedForward(nn.Module):
    """
    Feed-Forward MLP with depthwise convolution for local context.
    Replaces standard FFN in ViT to improve fine-detail reconstruction.
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
            groups=hidden_dim  # depthwise conv
        )
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, H=None, W=None):
        """
        x: [B, N, C] tokens (N = H×W)
        H, W: spatial dimensions of token grid
        """
        B, N, C = x.shape
        h = self.fc1(x)
        h = self.act(h)

        if H is None or W is None:
            # try to infer square layout
            HW = int(N ** 0.5)
            H = W = HW

        # reshape to [B, hidden_dim, H, W]
        h = h.transpose(1, 2).view(B, -1, H, W)
        h = self.dwconv(h)
        # back to [B, N, hidden_dim]
        h = h.flatten(2).transpose(1, 2)

        h = self.fc2(h)
        h = self.drop(h)
        return h


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., drop_path=0.0) -> None:
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        assert dim == heads * dim_head, "dim must be equal to heads * dim_head"
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ffn = ConvFeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drop_path_rates=(0.0)):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, drop_path=drop_path_rates[i]))

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class FourierFeatures(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, coords):
        # coords: (B, N, 2)
        pts = coords.unsqueeze(-1) * self.freq_bands   # (B, N, 2, F)
        sin = torch.sin(torch.pi * pts)
        cos = torch.cos(torch.pi * pts)
        return torch.cat([sin, cos], dim=-1).flatten(-2)  # (B,N,2F*2)


class SpeckleINR(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_freqs=6):
        super().__init__()
        self.posenc = FourierFeatures(num_freqs=num_freqs)
        self.norm = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            # nn.LayerNorm(feat_dim + num_freqs*4),
            nn.Linear(feat_dim + num_freqs*4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
            # nn.Sigmoid()
        )
        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(feat_dim + num_freqs*4),
        #     SIREN([256, 256, 256, 256], feat_dim + num_freqs*4, 2, w0=1.0, w0_initial=30, initializer='siren')
        #     # nn.Sigmoid()
        # )

    def forward(self, feat, scale=4):
        B, C, H, W = feat.shape
        out_h = int(scale*H)
        out_w = int(scale*W)
        # 1) Build coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1,1,out_h,device=feat.device),
            torch.linspace(-1,1,out_w,device=feat.device),
            indexing='ij'
        )
        coords = torch.stack([x,y],dim=-1).reshape(1,-1,2).repeat(B,1,1)
        # 2) Fourier encode coordinates
        enc = self.posenc(coords)    # (B, 40k, freq_dim)
        # 3) Interpolate low-res ViT features to HR
        # feat_hr = F.interpolate(feat, size=(out_h,out_w), mode="bicubic", align_corners=True)
        feat_hr = feat.permute(0,2,3,1).reshape(B,-1,C)
        # 4) Concatenate (Fourier coords + features)
        x_in = torch.cat([self.norm(feat_hr), enc], dim=-1)
        # 5) Predict
        out = self.mlp(x_in)    # (B, 40k, 1)
        # print(out.shape)
        out = out.reshape(B, out_h, out_w, 2).permute(0, 3, 1, 2)
        # print(out.shape)
        # out = torch.square(out).sum(dim=1, keepdims=True)
        # print(out.shape)
        return out

# forward transformers: [real, imag] --> [pat]
# inverse transformers: [pat] -> [real, imag]
class OpticsViTINR(nn.Module):
    def __init__(self, image_size=50, patch_size=5, enc_depth=6, dec_depth=6, dim=512, heads=8, dim_head=64,
                 mlp_dim=2048, in_channels=2, out_channels=1, act=nn.Sigmoid, out_dim=512, use_learnable_pos=False, num_reg=0, drop_path_rate=0.0, input_pad=0, pat_size=150):
        super().__init__()
        """
        complex input formulated as [real, imag], return a real image
        """
        image_size = image_size + input_pad
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_pad = input_pad
        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_reg = num_reg
        self.use_learnable_pos = use_learnable_pos
        assert pat_size % image_size == 0, 'pat_size should be divided by image_size'
        self.scale = pat_size // image_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches
        
        # drop_path_rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, enc_depth+dec_depth)] 

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=use_learnable_pos)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, out_dim), requires_grad=use_learnable_pos)
        if self.num_reg > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, num_reg, dim), requires_grad=True)

        self.encoder = nn.Sequential(
            Transformer(dim, depth=enc_depth, heads=heads, mlp_dim=mlp_dim, dropout=0.0, dim_head=dim_head, drop_path_rates=tuple(dpr[:enc_depth])),
            nn.LayerNorm(dim)
        )
        if dim == out_dim:
            self.tran = nn.Identity()
        else:
            self.tran = nn.Linear(dim, out_dim)
        self.decoder = nn.Sequential(
            Transformer(out_dim, depth=dec_depth, heads=heads, dim_head=out_dim//heads, mlp_dim=int(mlp_dim*out_dim/dim), drop_path_rates=tuple(dpr[enc_depth:])),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 24*self.patch_size**2*self.scale**2),
            # act()
        )

        self.out_embedding = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

        self.inr = SpeckleINR(feat_dim=96, hidden=256, num_freqs=24)

        self.cnt_params()
        self.init_weights()
    
    def init_weights(self):
        if not self.use_learnable_pos:
            print("=== use fixed pos")
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        else:
            print("=== use learnable pos")
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.decoder_pos_embed, std=0.02)
        # initialize register token
        if self.num_reg > 0:
            nn.init.normal_(self.reg_token, std=1e-6)

        # w = self.patch_embed.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def unpatchify(self, x):
        """
        x: [b, c, h, w] --> [b, l, d]
        """
        x = rearrange(x, 'b (h w) (p q c) -> b c (h p) (w q)', h=self.image_size//self.patch_size, p=self.patch_size, q=self.patch_size)
        x = rearrange(x, 'b (s r c) h w -> b c (s h) (r w)', s=self.scale, r=self.scale)
        return x
    
    def forward_features(self, x, scale=1):
        b = x.shape[0]
        if self.in_channels == 2:
            x = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
        # x = self.pre_conv(x)
        x = F.pad(x, (0, self.input_pad, 0, self.input_pad), mode='constant', value=0)
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c') # [b, l, c]
        x = x + self.pos_embed
        if self.num_reg > 0:
            x = torch.cat([self.reg_token.repeat(b, 1, 1), x], dim=1)
        x = self.encoder(x)
        x = self.tran(x)
        if self.num_reg > 0:
            reg_token, patch_token = x[:, :self.num_reg, :], x[:, self.num_reg:, :]
            patch_token = patch_token + self.decoder_pos_embed
            x = torch.cat([reg_token, patch_token], dim=1)
        else:
            x = x + self.decoder_pos_embed
        out = self.decoder(x)
        out = out[:, self.num_reg:, :]
        out = self.unpatchify(out)[:, :, :self.image_size*self.scale-self.input_pad, :self.image_size*self.scale-self.input_pad]
        out_0 = out
        # do inr
        feat_embed = self.out_embedding(out)
        out_inr = self.inr.forward(feat=feat_embed, scale=1)
        return out, out_inr
    
    def forward(self, x, scale=4):
        out = self.forward_features(x, scale=scale)
        return out
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
        print("Freeze all parameters ... ")
    
    def cnt_params(self):
        num_param = sum([param.numel() for param in self.parameters()])
        num_param = num_param / (10**6)
        print("=== Number of params: {:.3f}M".format(num_param))


if __name__ == "__main__":
    x = torch.rand(1, 2, 50, 50)
    model = OpticsViTINR(50, 5, 6, 6, 512, 8, 64, 2048, 2, 1, nn.Sigmoid)
    model.eval()
    out_0, out_inr = model(x, scale=2.7)
    print(out_0.shape)
    print(out_inr.shape)
    print(out_0)
    print(out_inr)
