import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(
    str(Path(__file__).parent.parent) + "/Joint_Supervised_Learning_for_SR/"
)

import glob
import json
import torch
from omegaconf import omegaconf

from Joint_Supervised_Learning_for_SR.src.data import JCLDataset, tokenize
from Joint_Supervised_Learning_for_SR.src.utils import processDataFiles

from utils import seq_to_tree


def load_dataset(mode, cfg, debug=False):
    """
    Load the dataset based on the given mode.

    Args:
    - mode (str): The mode to load data for. Either "train", "SSDNC" or "eval".
    - cfg: Configuration for data loading.

    Returns:
    - SymDataset: A dataset instance with the loaded data.
    """
    # Metadata path
    metadata_path = Path("../Joint_Supervised_Learning_for_SR/data/raw_datasets/100000")

    # Load configuration
    JCL_cfg = omegaconf.OmegaConf.load(
        "../Joint_Supervised_Learning_for_SR/config.yaml"
    )

    # Determine data path based on mode
    if mode == "train" or mode == "full":
        data_path = (
            "../Joint_Supervised_Learning_for_SR/Dataset/2_var/5000000/Train/*.json"
        )
    elif mode == "SSDNC" or mode == "eval":
        data_path = (
            "../Joint_Supervised_Learning_for_SR/Dataset/2_var/5000000/SSDNC/*.json"
        )
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Supported modes are ['train', 'SSDNC']."
        )

    # Process files
    files = glob.glob(data_path)
    if not debug:
        text = processDataFiles(files)
    else:
        text = processDataFiles(files[:1])

    # Convert the raw text to a set of examples
    examples = text.split("\n")
    examples = examples[:-1] if len(examples[-1]) == 0 else examples
    examples = examples

    if mode == "eval" or mode == "full":
        return SymEvalDataset(examples, metadata_path, JCL_cfg, cfg)

    # Return dataset
    return SymDataset(examples, metadata_path, JCL_cfg, cfg)

class JCL_Dataset(JCLDataset):
    def __getitem__(self, index):

        chunk = self.data[index]
        try:
            chunk = json.loads(chunk)
        except:
            # try the previous example
            index = index - 1
            index = index if index >= 0 else 0
            chunk = self.data[index]
            chunk = json.loads(chunk)

        # if self.mode == "train" or self.mode == "val":
        traversal = chunk['traversal']
        eq_id = torch.tensor(self.eq2id[str(traversal)], dtype=torch.long)
        # print(traversal)
        # print(chunk['eq'])
        # print(chunk['skeleton'])
        tokenized_expr = tokenize(traversal, self.word2id)
        Padding_size = max(self.block_size - len(tokenized_expr), 0)
        trg = tokenized_expr + [self.cfg.architecture.trg_pad_idx] * Padding_size
        points = torch.zeros(self.num_Vars + self.num_Ys, self.number_of_points)
        for idx, xy in enumerate(zip(chunk['X'], chunk['y'])):
            x = xy[0]  # list x
            # x = [(e-minX[eID])/(maxX[eID]-minX[eID]+eps) for eID, e in enumerate(x)] # normalize x
            x = x + [0] * (max(self.num_Vars - len(x), 0))  # padding

            y = [xy[1]] if type(xy[1]) == float or type(xy[1]) == np.float64 else xy[1]  # list y

            # y = [(e-minY)/(maxY-minY+eps) for e in y]
            y = y + [0] * (max(self.num_Ys - len(y), 0))  # padding
            p = x + y
            p = torch.tensor(p)

            # p = torch.nan_to_num(p, nan=self.threshold[1],
            #                      posinf=self.threshold[1],
            #                      neginf=self.threshold[0])
            points[:, idx] = p

        # points = torch.nan_to_num(points, nan=self.threshold[1],
        #                           posinf=self.threshold[1],
        #                           neginf=self.threshold[0])
        trg = torch.tensor(trg, dtype=torch.long)
        return points, traversal, eq_id, chunk['eq'], chunk['skeleton']


class SymDataset(torch.utils.data.Dataset):
    def __init__(self, text, metadata_path, cfg_data, cfg):
        self._dataset = JCL_Dataset(text, metadata_path, cfg_data, mode="train")
        self._cfg = cfg

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        point_set, seq, eq_id, eq_expr, skeleton = self._dataset[idx]
        eq_tree = seq_to_tree(seq, self._cfg["max_step"])

        # Select random steps
        max_step = (eq_tree.sum(dim=1) == 0).nonzero(as_tuple=False)[0].item()
        step = torch.randint(0, max_step, (1,))
        action = eq_tree[step].squeeze()
        tree = torch.zeros(eq_tree.shape)
        if step != 0:
            tree[:step] = eq_tree[:step]

        return point_set, tree, action, point_set, eq_id, skeleton, str(seq), eq_expr


class SymEvalDataset(torch.utils.data.Dataset):
    def __init__(self, trainText, metadata_path, cfg_data, cfg):
        self._dataset = JCL_Dataset(trainText, metadata_path, cfg_data, mode="train")
        self._cfg = cfg

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        point_set, seq, eq_id, eq_expr, skeleton = self._dataset[idx]
        point_set = point_set.unsqueeze(0)
        eq_tree = seq_to_tree(seq, self._cfg["max_step"])

        # Return observations at all steps
        max_step = (eq_tree.sum(dim=1) == 0).nonzero(as_tuple=False)[0].item()
        data = []
        for step in range(max_step):
            action = eq_tree[step].squeeze()
            tree = torch.zeros(eq_tree.shape)
            if step != 0:
                tree[:step] = eq_tree[:step]
            data.append(
                [
                    point_set,
                    tree.unsqueeze(0),
                    action,
                    point_set,
                    eq_id,
                    skeleton,
                    seq,
                    eq_expr,
                ]
            )
        return data
