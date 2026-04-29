import torch
import torch.nn as nn
import pdb


class MLP(nn.Module):
    def __init__(self, input_size=50, output_size=200, in_channels=2, out_channels=1, mode='amp'):
        super().__init__()
        
        self.img_size = output_size
        self.out_channels = out_channels
        self.mode = mode
        act = nn.Sigmoid if mode == 'amp' else nn.Tanh
        
        self.layer = nn.Sequential(
            nn.Linear(input_size**2*in_channels, output_size**2*out_channels),
            act()
        )
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
        
    def forward(self, x, mode='amp'):
        if mode == 'comp':
            cos_sig = torch.cos(x*torch.pi*2)
            sin_sig = torch.sin(x*torch.pi*2)
            x = torch.cat([cos_sig, sin_sig], dim=1)
        b = x.shape[0]
        out = self.layer(x.view(b, -1))
        out = out.reshape(b, self.out_channels, self.img_size, self.img_size)
        if self.mode != "amp":
            out = nn.functional.normalize(out, dim=1)
        # pdb.set_trace()
        return out