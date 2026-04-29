import torch
from torch import nn
from einops import rearrange
from timm.layers import trunc_normal_
import numpy as np
import torch.nn.functional as F


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
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.) -> None:
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        assert dim == heads * dim_head, "dim must be equal to heads * dim_head"
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ffn = ConvFeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim))

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x

# forward transformers: [real, imag] --> [pat]
# inverse transformers: [pat] -> [real, imag]
class OpticsViT(nn.Module):
    def __init__(self, input_size=200, input_patch_size=20, enc_depth=6, output_size=50, output_patch_size=5, dec_depth=6, 
                 dim=512, heads=8, dim_head=64, mlp_dim=2048, in_channels=1, out_channels=2, act=nn.Tanh, 
                 out_norm=True):
        super().__init__()
        """
        complex input formulated as [real, imag], return a real image
        """
        self.in_patch_size, self.out_patch_size = input_patch_size, output_patch_size
        self.in_channels, self.out_channels = in_channels, out_channels
        self.in_size, self.out_size = input_size, output_size
        self.out_channels = out_channels
        self.out_norm = out_norm

        assert input_size % input_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert output_size % output_patch_size == 0, 'Image dimension must be divisible by mask size'
        # assert (input_size // input_patch_size) == (output_size // output_patch_size), 'input and output tokens should be equal'
        num_patches = (input_size // input_patch_size) ** 2
        self.num_patches = num_patches

        self.out_num_patches = (self.out_size//self.out_patch_size)**2

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=(self.in_patch_size, self.in_patch_size), stride=(self.in_patch_size, self.in_patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.out_num_patches, dim), requires_grad=False)

        self.encoder = nn.Sequential(
            Transformer(dim, depth=enc_depth, heads=heads, mlp_dim=mlp_dim, dropout=0.0, dim_head=dim_head),
            nn.LayerNorm(dim)
        )
        self.decoder = nn.Sequential(
            Transformer(dim, depth=dec_depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, out_channels*self.out_patch_size * self.out_patch_size),
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*8, kernel_size=(2+output_patch_size), stride=1, padding=(2+output_patch_size)//2, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(out_channels*8, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            act()
        )

        self.cnt_params()
        self.init_weights()
    
    def init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.out_num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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
        x = rearrange(x, 'b (c h w) (p q) -> b c (h p) (w q)', h=self.out_size//self.out_patch_size, p=self.out_patch_size, c=self.out_channels)
        return x
    
    def match_enc_dec(self, x):
        # match the size of enc and dec
        # enc: [b, l, c] --> dec: [b, l', c]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.in_size // self.in_patch_size)
        h_out = self.out_size // self.out_patch_size
        x = F.interpolate(x, (h_out, h_out), mode='bilinear')
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
    def forward_features(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c') # [b, l, c]
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.match_enc_dec(x) # align spatial resolution
        x = x + self.decoder_pos_embed
        out = self.decoder(x)
        out = self.unpatchify(out)
        out = self.conv_out(out)
        if self.out_norm:
            out = F.normalize(out, dim=1)
        return out
    
    def forward(self, x):
        out = self.forward_features(x)
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
    x = torch.rand(1, 1, 200, 200)
    model = OpticsViT(200, 10, 6, 50, 5, 6, act=nn.Tanh, out_norm=True)
    model.eval()
    out = model(x)
    print(out.shape)
    print(out)
