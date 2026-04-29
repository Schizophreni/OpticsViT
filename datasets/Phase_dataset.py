import torch
import torch.utils.data as Data
import os
import numpy as np
import glob
import torchvision
import torch.nn.functional as F
import random


class OpticsDataset(Data.Dataset):
    """
    Paired dataset for correspond input / output image pairs
    """
    def __init__(self, train:bool=True, root_dir="data", input_size=50) -> None:
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
        # pat_files = self.get_stage_imgs(train)
        self.inputs = phase_files # pat_files
        print("=== Dataset size ({}): {}".format("train" if train else "eval", len(self.inputs)))
        self.EMNIST = torchvision.datasets.EMNIST(root='data', download=True, split='letters')
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor()
        ])
        
        self.crop = torchvision.transforms.CenterCrop((input_size, input_size))
    
    def __len__(self):
        return len(self.inputs)
    
    def get_stage_imgs(self, train=True):
        if train:
            pat_dir = "data/phase_data_simulation_stage1/train/pat"
        else:
            pat_dir = "data/phase_data_simulation_stage1/eval/pat"
        pat_files = glob.glob(os.path.join(pat_dir, "*.npy"))
        return pat_files
    
    def __getitem__(self, idx):
        phase_file = self.inputs[idx] # "matrix_0000.npy" format
        phase_name = phase_file.split("_")[-1].split(".")[0]
        phase_name = int(phase_name)
        pat_file = os.path.join(self.pat_dir, f"pat_{phase_name}.npy")
        # pat_file = self.inputs[idx]
        # pat_name = pat_file.split("_")[-1].split(".")[0]
        # phase_file = f"newton_results/phase_150_phase_data_simulation/emnist_{int(pat_name)}.npy"
        # read file
        phase, pat = np.load(phase_file), np.load(pat_file)
        phase, pat = torch.from_numpy(phase), torch.from_numpy(pat)
        pat = pat.float()
        pat = self.crop(pat).squeeze(0)
        # print(phase.min(), phase.max(), pat.min(), pat.max())
        # phase = phase + random.random()
        phase = phase.float() * 2 * torch.pi
        pat = (pat/60).clamp(0.0, 1.0)
        # phase = phase * (2*torch.pi)
        # print(phase.min(), phase.max())
        phase.unsqueeze_(0)
        phase = torch.cat([torch.cos(phase), torch.sin(phase)], dim=0)
        
        emnist = self.EMNIST[idx % len(self.EMNIST)][0]
        emnist = self.transform(emnist)
        return phase, pat, pat_file.split("/")[-1].split(".")[0], emnist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = OpticsDataset(root_dir="../data/phase_data", train=True)
    print(len(dataset))
    inp, out, _, _  = dataset[0]
    print(inp.shape, out.shape, inp.min(), inp.max(), out.min(), out.max())