import os
import numpy as np
import glob
import torchvision
import torch
import torch.utils.data as Data


class OpticsDataset(Data.Dataset):
    """
    Paired dataset for correspond input / output image pairs
    """
    def __init__(self, train:bool=True, args=None) -> None:
        super().__init__()
        """
        data_dir: root folder of noisy images
        img_size: cropped img size
        """
        assert args is not None, "Arguments cannot be None"
        root_dir = args.data_dir
        self.clip_speckle = args.clip_speckle # clip speckle amplitude
        self.input_scale = args.input_scale # scale input value (e.g., phase 2pi)
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
        ])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        phase_file = self.inputs[idx]
        phase_name = phase_file.split('/')[-1].replace("phase_", "")
        pat_file = os.path.join(self.pat_dir, f"pat_{phase_name}")
        # read file
        phase, pat = np.load(phase_file), np.load(pat_file)
        phase, pat = torch.from_numpy(phase), torch.from_numpy(pat)
        pat = (pat / self.clip_speckle).clamp(0, 1.0)
        pat = self.crop(pat).squeeze(0)
                
        phase = phase.unsqueeze(0) * self.input_scale
        emnist = self.EMNIST[idx % len(self.EMNIST)][0]
        emnist = self.transform(emnist)
        return phase, pat, emnist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = OpticsDataset(root_dir="../data/phase_data", train=True)
    print(len(dataset))
    inp, out, _  = dataset[0]
    print(inp.shape, out.shape, inp.min(), inp.max(), out.min(), out.max())