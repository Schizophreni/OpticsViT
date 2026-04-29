import torch
import torch.utils.data as Data
import os
import numpy as np
import glob


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
            amp_dir = os.path.join(root_dir, "train", "amp")
            self.pat_dir = os.path.join(root_dir, "train", "pat")
        else:
            amp_dir = os.path.join(root_dir, "eval", "amp")
            self.pat_dir = os.path.join(root_dir, "eval", "pat")
        train_phase_dir = os.path.join(root_dir, "train", "phase")
        val_phase_dir = os.path.join(root_dir, "eval", "phase")
        # initialize
        amp_files = glob.glob(os.path.join(amp_dir, "*.npy"))
        train_phase_files = glob.glob(os.path.join(os.path.join(train_phase_dir, "*.npy")))
        val_phase_files = glob.glob(os.path.join(val_phase_dir, "*.npy"))
        phase_files = train_phase_files + val_phase_files
        self.inputs = []
        for amp_f in amp_files:
            for phase_f in phase_files:
                self.inputs.append((amp_f, phase_f))
        print("=== Dataset size ({}): {}".format("train" if train else "eval", len(self.inputs)))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        amp_file, phase_file = self.inputs[idx] # "matrix_0000.npy" format
        amp_name, phase_name = amp_file.split("_")[-1].split(".")[0], phase_file.split("_")[-1].split(".")[0]
        pat_file = os.path.join(self.pat_dir, f"amp_{amp_name}_phase_{phase_name}.npy")
        # read file
        amp, phase, pat = np.load(amp_file), np.load(phase_file), np.load(pat_file)
        amp, phase, pat = torch.from_numpy(amp), torch.from_numpy(phase), torch.from_numpy(pat)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        signal = torch.cat([real, imag], dim=0).float()
        pat = pat.float()
        return signal, pat, pat_file.split("/")[-1].split(".")[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = OpticsDataset(root_dir="../data", train=True)
    print(len(dataset))
    inp, out = dataset[0]
    print(inp.shape, out.shape)