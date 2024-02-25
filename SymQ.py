import torch
import torch.nn as nn

from encoder import SetEncoder, TreeEncoder


class SymQ(nn.Module):
    """
    Attributes:
    - set_encoder (SetEncoder): Encoder for input point sets.
    - tree_encoder (TreeStructureEncoder): Encoder for tree-structured data.
    - device (torch.device): Device for storing tensors.
    """

    def __init__(self, cfg, device):
        super(SymQ, self).__init__()
        self.device = device
        self.set_encoder = SetEncoder(cfg["SetEncoder"]).to(device)
        self.tree_encoder = TreeEncoder(cfg["TreeEncoder"]).to(device)
        self.cfg = cfg["SymQ"]
        self.dropout = nn.Dropout(0)

        if self.cfg["batch_norm"]:
            self.batch_norm1 = nn.BatchNorm1d(self.cfg["dim_hidden"])
            self.batch_norm2 = nn.BatchNorm1d(self.cfg["dim_hidden"])

        if self.cfg["embedding_fusion"] == "concat":
            dim_fusion = (
                cfg["SetEncoder"]["dim_output"] + cfg["TreeEncoder"]["dim_output"]
            )
        elif self.cfg["embedding_fusion"] == "add":
            assert cfg["SetEncoder"]["dim_output"] == cfg["TreeEncoder"]["dim_output"]
            dim_fusion = cfg["SetEncoder"]["dim_output"]
        else:
            raise KeyError(
                f"Unknown embedding fusion method: {self.cfg['embedding_fusion']}"
            )
        self.downsample = nn.Linear(
            int(512 * cfg["SetEncoder"]["num_features"]),
            cfg["SetEncoder"]["dim_output"],
        )
        self.BN_downsample = nn.BatchNorm1d(cfg["SetEncoder"]["dim_output"])
        self.act_downsample = nn.GELU()
        self.linear1 = nn.Linear(dim_fusion, self.cfg["dim_hidden"])
        self.relu1 = nn.GELU()

        self.linear2 = nn.Linear(self.cfg["dim_hidden"], self.cfg["dim_hidden"])
        self.relu2 = nn.GELU()

        self.linear3 = nn.Linear(self.cfg["dim_hidden"], self.cfg["dim_hidden"])
        self.relu3 = nn.GELU()

        self.linear4 = nn.Linear(self.cfg["dim_hidden"], self.cfg["num_actions"])

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        Uses Xavier Uniform initialization for linear layers and sets bias to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _encode_set(self, x):
        x = x.permute(0, 2, 1)
        x = self.set_encoder(x)

        return x.view(x.size(0), -1)

    def _encode_tree(self, x):
        return self.tree_encoder(x)

    def forward(self, point_set, tree, support):
        # point_set: [batch_size, 2, num_point]
        # operator: [batch_size, seq_len, num_action]

        set_embedding = self._encode_set(point_set)
        set_embedding = self.act_downsample(
            self.BN_downsample(self.downsample(set_embedding))
        )
        forward_info = set_embedding.clone()
        tree_embedding = self._encode_tree(tree)

        if self.cfg["embedding_fusion"] == "concat":
            embeddings = torch.cat((self.dropout(forward_info), tree_embedding), 1)
        elif self.cfg["embedding_fusion"] == "add":
            embeddings = self.dropout(forward_info) + tree_embedding

        x = self.linear1(embeddings)
        if self.cfg["batch_norm"]:
            x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        if self.cfg["batch_norm"]:
            x = self.batch_norm2(x)
        fusion = x
        x = self.relu2(x)
        x = self.relu3(self.linear3(x))
        if self.cfg["set_skip_connection"]:
            assert x.shape == forward_info.shape
            x = x + forward_info
        x = self.linear4(x)
        q_values = x

        return q_values, set_embedding, set_embedding, fusion

    def act(self, point_set, tree):
        with torch.no_grad():
            q_values, _, _, _ = self.forward(point_set, tree, point_set)
        return q_values.argmax(dim=-1), q_values


if __name__ == "__main__":
    pass
