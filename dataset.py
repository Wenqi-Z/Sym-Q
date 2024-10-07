import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from utils import seq_to_tree, get_seq2action


def get_h5_files_in_folder(folder_path):
    h5_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    return sorted(h5_files)


class HDF5Dataset(Dataset):
    def __init__(self, folder_path, cfg):
        self.cfg = cfg
        self.seq2action = get_seq2action(cfg)

        if "train" in folder_path:
            self.hdf5_files = get_h5_files_in_folder(folder_path)[:cfg.training.dataset_num]
        elif "val" in folder_path:
            self.hdf5_files = get_h5_files_in_folder(folder_path)

        self.datasets = []
        self.indices = []
        self.cumulative_sizes = []

        for hdf5_file in tqdm(self.hdf5_files):
            with h5py.File(hdf5_file, "r") as hf:
                length = len(hf["action"])
                self.indices.append((hdf5_file, length))
                self.cumulative_sizes.append(
                    self.cumulative_sizes[-1] + length
                    if self.cumulative_sizes
                    else length
                )

    def __len__(self):
        """Return the total number of prefix samples"""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        file_idx = next(i for i, v in enumerate(self.cumulative_sizes) if v > idx)
        if file_idx > 0:
            idx -= self.cumulative_sizes[file_idx - 1]

        hdf5_file = self.hdf5_files[file_idx]

        # Open the HDF5 file lazily (each time a sample is requested)
        with h5py.File(hdf5_file, "r") as hf:
            points_idx = hf["map"][idx]
            points = torch.tensor(hf["points"][points_idx], dtype=torch.float32)

            prefix = hf["prefix"][idx]
            prefix = [int(p) for p in prefix if not np.isnan(p)]
            prefix = seq_to_tree(prefix, self.cfg)

            eq_id = torch.tensor(hf["eq_id"][idx], dtype=torch.long)
            action = torch.tensor(hf["action"][idx], dtype=torch.long)
            q_values = torch.zeros(len(self.seq2action), dtype=torch.float32)
            q_values[action] = 1

        return points, prefix, eq_id, action, q_values


if __name__ == "__main__":
    from utils import load_cfg
    cfg = load_cfg("cfg.yaml")

    folder_path = f"{cfg.Dataset.dataset_folder}/{cfg.num_vars}_var/train"
    dataset = HDF5Dataset(folder_path, cfg)

    # for i in range(1000):
    #     sample = dataset[i]
    #     points = sample[0]
    #     prefix = sample[1]
    #     eq_id = sample[2]
    #     action = sample[3]
    #     q_values = sample[4]
    #     print(f"Points: {points[:, :10]}")
    #     print(f"Prefix: {prefix}")
    #     print(f"Eq ID: {eq_id}")
    #     print(f"Next Action: {action}")
    #     print(f"Q Values: {q_values}")
    #     input("Press Enter to continue...")


    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    batch = next(iter(dataloader))
    print(f"Points: {batch[0].shape}")
    print(f"Prefix: {batch[1].shape}")
    print(f"Eq ID: {batch[2].shape}")
    print(f"Next Action: {batch[3].shape}")
    print(f"Q Values: {batch[4].shape}")

