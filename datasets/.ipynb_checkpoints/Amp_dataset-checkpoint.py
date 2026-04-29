import torch
import torch.utils.data as Data
import os
import numpy as np
import glob
import torchvision
import torch.nn.functional as F


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
            amp_dir = os.path.join(root_dir, "train", "amp")
            self.pat_dir = os.path.join(root_dir, "train", "pat")
        else:
            amp_dir = os.path.join(root_dir, "eval", "amp")
            self.pat_dir = os.path.join(root_dir, "eval", "pat")
        # initialize
        amp_files = glob.glob(os.path.join(amp_dir, "*.npy"))
        # pat_files = self.get_stage_imgs(train)
        self.inputs = amp_files
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
            pat_dir = "data/data_6000us_iter1/train/pat"
        else:
            pat_dir = "data/data_6000us_iter1/eval/pat"
        pat_files = glob.glob(os.path.join(pat_dir, "*.npy"))
        return pat_files
    
    def __getitem__(self, idx):
        amp_file = self.inputs[idx] # "matrix_0000.npy" format
        amp_name = amp_file.split("_")[-1].split(".")[0]
        pat_file = os.path.join(self.pat_dir, f"pat_{amp_name}.npy")
        # pat_file = self.inputs[idx]
        # pat_name = pat_file.split("_")[-1].split(".")[0]
        # phase_file = f"newton_results/phase_150_experiment_6000us_emnist_all/emnist_{int(pat_name)}.npy"
        # read file
        amp, pat = np.load(amp_file), np.load(pat_file)
        amp, pat = torch.from_numpy(amp), torch.from_numpy(pat)
        pat = pat.float()
        pat = self.crop(pat).squeeze(0)
        pat = pat.clamp(0.0, 1.0)
        amp = amp.unsqueeze(0)
        # pat = pat.unsqueeze(0)
        # phase = phase.float() / 1023.0 * torch.pi * 2
        # pat = F.interpolate(pat.unsqueeze(0), (50,50)).clamp(0, 1.0).squeeze(0)
        # phase = phase / (2*torch.pi)
        # phase = torch.cat([torch.cos(phase), torch.sin(phase)], dim=0)
        
        emnist = self.EMNIST[idx % len(self.EMNIST)][0]
        emnist = self.transform(emnist)
        # print(amp.shape, pat.shape)
        # print(pat.shape, amp.shape)
        return amp, pat, pat_file.split("/")[-1].split(".")[0], emnist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = OpticsDataset(root_dir="../data/phase_data", train=True)
    print(len(dataset))
    inp, out, _, _  = dataset[0]
    print(inp.shape, out.shape, inp.min(), inp.max(), out.min(), out.max())