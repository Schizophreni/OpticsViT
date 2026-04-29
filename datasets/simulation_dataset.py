# import torch
import torch.utils.data as Data
import os
import numpy as np
import glob
import torchvision
import torch.nn.functional as F
import torch


class OpticsDataset(Data.Dataset):
    """
    Paired dataset for correspond input / output image pairs
    """
    def __init__(self, train:bool=True, root_dir="data") -> None:
        super().__init__()
        """
        data_dir: root folder of noisy images
        img_size: cropped img size
        """
        if train:
            phase_dir = os.path.join(root_dir, "train", "phase")
            self.pat_dir = os.path.join(root_dir, "train", "pat")
        else:
            phase_dir = os.path.join(root_dir, "eval", "phase")
            self.pat_dir = os.path.join(root_dir, "eval", "pat")
        # initialize
        phase_files = glob.glob(os.path.join(phase_dir, "*.npy"))
        phase_files = [item for item in phase_files if "psf" not in item and "checkpoint" not in item]
        self.inputs = phase_files
        print("=== Dataset size ({}): {}".format("train" if train else "eval", len(self.inputs)))
        self.EMNIST = torchvision.datasets.EMNIST(root='data', download=True, split='letters')
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((100, 100)),
            torchvision.transforms.ToTensor()
        ])
        self.crop = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((100, 100))
            # torchvision.transforms.Resize((50, 50))
        ])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        phase_file = self.inputs[idx] # "matrix_0000.npy" format
        # # phase_name = phase_file.split('/')[-1].split(".")[0].split("_")[-1]
        # pat_file = os.path.join(self.pat_dir, f"pat_{int(phase_name)}.npy")
        phase_name = phase_file.split('/')[-1].replace("phase_", "")
        pat_file = os.path.join(self.pat_dir, f"pat_{phase_name}")
        # read file
        phase, pat = np.load(phase_file), np.load(pat_file)
        phase, pat = torch.from_numpy(phase), torch.from_numpy(pat)
        # pat = pat.float() / (28.0)
        pat = (pat / 120).clamp(0, 1.0)
        # pat = torch.log(1 + pat) / np.log(1+40)
        # pat = (pat - 0.05).clamp(0, 1.0)
        # pat = pat.clamp(0, 0.7) * 1/0.7
        # pat = torch.sqrt(pat)
        # pat = self.crop(pat.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0)
        # pat = pat[0, :, 120:220, 160:260]
        pat = self.crop(pat).squeeze(0)
        
        # phase_shift = (phase.unsqueeze(0) + np.random.rand()) * 2 * torch.pi
        
        phase = phase.unsqueeze(0) # * 2 * torch.pi
        # mask = (phase > torch.pi).float()
        # phase = torch.cat([torch.cos(phase), torch.sin(phase), mask], dim=0)
        # phase = phase.unsqueeze(0)
        
        # phase_shift = torch.cat([torch.cos(phase_shift), torch.sin(phase_shift)], dim=0)
        
        emnist = self.EMNIST[idx % len(self.EMNIST)][0]
        emnist = self.transform(emnist)
        # print(phase.shape, pat.shape)
        
        # print(phase.shape, pat.shape)
        return phase, pat, pat_file.split("/")[-1].split(".")[0], emnist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = OpticsDataset(root_dir="../data/phase_data", train=True)
    print(len(dataset))
    inp, out, _  = dataset[0]
    print(inp.shape, out.shape, inp.min(), inp.max(), out.min(), out.max())