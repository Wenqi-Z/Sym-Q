import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
import math

from attention_block import MAB, ISAB, PMA


class SetEncoder(pl.LightningModule):
    """
    Set Encoder Module.

    A neural network model designed for encoding sets into a fixed-sized representation.
    This model provides the flexibility to choose different data representations,
    normalization strategies, and neural architectures.

    Attributes:
        linear (bool): Flag to use a linear transformation.
        bit16 (bool): Flag to represent input data in 16-bit format.
        norm (bool): Flag for normalization.
        activation (str): Type of activation function after the linear transformation.
        input_normalization (bool): Flag to normalize the input data.
        linearl (nn.Linear): Linear transformation layer.
        selfatt (nn.ModuleList): List of ISAB (Induced Set Attention Blocks) layers.
        selfatt1 (ISAB): Initial ISAB layer.
        outatt (PMA): Point-wise Multihead Attention layer.
        _device (torch.device): The device (CPU or GPU) where the module is deployed.
    """

    def __init__(self, cfg):
        """
        Initialize the SetEncoder.

        Args:
            cfg (dict): Configuration dictionary containing model hyperparameters.
        """
        super(SetEncoder, self).__init__()
        self.linear = cfg["linear"]
        self.bit16 = cfg["bit16"]
        self.norm = cfg["norm"]
        assert (
            cfg["linear"] != cfg["bit16"]
        ), "one and only one between linear and bit16 must be true at the same time"
        if cfg["norm"]:
            self.register_buffer("mean", torch.tensor(cfg["mean"]))
            self.register_buffer("std", torch.tensor(cfg["std"]))

        self.activation = cfg["activation"]
        self.input_normalization = cfg["input_normalization"]
        if cfg["linear"]:
            self.linearl = nn.Linear(cfg["dim_input"], 16 * cfg["dim_input"])
        self.selfatt = nn.ModuleList()
        # dim_input = 16*dim_input
        self.selfatt1 = ISAB(
            16 * cfg["dim_input"],
            cfg["dim_hidden"],
            cfg["num_heads"],
            cfg["num_inds"],
            ln=cfg["ln"],
        )
        for i in range(cfg["n_l_enc"]):
            self.selfatt.append(
                ISAB(
                    cfg["dim_hidden"],
                    cfg["dim_hidden"],
                    cfg["num_heads"],
                    cfg["num_inds"],
                    ln=cfg["ln"],
                )
            )
        self.outatt = PMA(
            cfg["dim_hidden"], cfg["num_heads"], cfg["num_features"], ln=cfg["ln"]
        )

    def float2bit(
        self, f, num_e_bits=5, num_m_bits=10, bias=127.0, dtype=torch.float32
    ):
        """
        Convert floating-point numbers to a bit representation.

        Args:
            f (torch.Tensor): Input tensor of floating-point numbers.
            num_e_bits (int, optional): Number of bits for the exponent. Defaults to 5.
            num_m_bits (int, optional): Number of bits for the mantissa. Defaults to 10.
            bias (float, optional): Bias for the conversion. Defaults to 127.0.
            dtype (torch.dtype, optional): Data type for the output tensor. Defaults to torch.float32.

        Returns:
            torch.Tensor: Tensor with the bit representation of the input floating-point numbers.
        """
        ## SIGN BIT
        s = (
            torch.sign(f + 0.001) * -1 + 1
        ) * 0.5  # Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        ## EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        ## MANTISSA
        f2 = f1 / 2 ** (e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:, :, :, :num_m_bits]  # [:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        """
        Convert remainder of floating-point number to bit representation.

        Args:
            remainder (torch.Tensor): Input tensor with remainders of floating-point numbers.
            num_bits (int, optional): Number of bits for the conversion. Defaults to 127.

        Returns:
            torch.Tensor: Tensor with the bit representation of the input remainders.
        """
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device=remainder.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2**exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        """
        Convert integer values to a bit representation.

        Args:
            integer (torch.Tensor): Input tensor of integer values.
            num_bits (int, optional): Number of bits for the conversion. Defaults to 8.

        Returns:
            torch.Tensor: Tensor with the bit representation of the input integer values.
        """
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=integer.device).type(
            dtype
        )
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2**exponent_bits
        return (out - (out % 1)) % 2

    def forward(self, x):
        if self.bit16:
            x = self.float2bit(x)
            x = x.view(x.shape[0], x.shape[1], -1)
            if self.norm:
                x = (x - 0.5) * 2
        if self.input_normalization:
            means = x[:, :, -1].mean(axis=1).reshape(-1, 1)
            std = x[:, :, -1].std(axis=1).reshape(-1, 1)
            std[std == 0] = 1
            x[:, :, -1] = (x[:, :, -1] - means) / std

        if self.linear:
            if self.activation == "relu":
                x = torch.relu(self.linearl(x))
            elif self.activation == "sine":
                x = torch.sin(self.linearl(x))
            else:
                x = self.linearl(x)
        x = self.selfatt1(x)
        for layer in self.selfatt:
            x = layer(x)
        x = self.outatt(x)
        return x


class TreeEncoder(nn.Module):
    """
    Tree Encoder.

    Implements an encoder that takes tree-structured data and encodes it into a fixed-sized vector using
    Transformer architecture. The assumption is that the tree data has been flattened into a sequential
    representation that can be input to this model.

    Attributes:
        input_dim (int): The dimension of the input data.
        embed_dim (int): The desired embedding dimension after the initial linear layer.
        max_length (int): The maximum length of the sequential representation of the tree structure.
        embedding_layer (nn.Linear): Linear layer to transform the input data to the desired embedding dimension.
        transformer_encoder (nn.TransformerEncoder): The main encoder based on Transformer architecture.
        output_layer (nn.Linear): Linear layer to map from the embed_dim to the desired output_dim.
    """

    def __init__(self, cfg):
        """
        Initialize the TreeStructureEncoder.

        Args:
            input_dim (int): The dimension of the input data.
            max_length (int): The maximum length of the sequential representation of the tree structure.
            embed_dim (int): Desired embedding dimension.
            num_heads (int): Number of attention heads in the TransformerEncoder.
            num_encoder_layers (int): Number of layers in the TransformerEncoder.
            output_dim (int): Desired output dimension after the final linear layer.
        """
        super(TreeEncoder, self).__init__()

        self.max_length = cfg["dim_input"][0]
        self.embed_dim = cfg["dim_hidden"]
        self.embedding_layer = nn.Linear(cfg["dim_input"][1], cfg["dim_hidden"])
        self.pos_embedding = nn.Embedding(
            num_embeddings=cfg["dim_input"][1], embedding_dim=cfg["dim_hidden"]
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg["dim_hidden"], nhead=cfg["num_heads"]
            ),
            num_layers=cfg["num_layers"],
        )
        # self.transformer_encoder=nn.Sequential(*[Block(cfg) for _ in range(cfg["num_layers"])])

        self.output_layer = nn.Linear(cfg["dim_hidden"], cfg["dim_output"])

    def create_positional_encodings(self, max_length, embed_dim):
        """
        Create positional encodings.

        Args:
            max_length (int): Maximum length of the sequential representation.
            embed_dim (int): Embedding dimension.

        Returns:
            torch.Tensor: Tensor containing positional encodings of shape (max_length, embed_dim).
        """
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Forward pass for TreeStructureEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_length, input_dim).

        Returns:
            torch.Tensor: Encoded output tensor after processing through the TransformerEncoder and
                          a final linear layer.
        """
        embeddings = self.embedding_layer(x)

        # Uncomment below if you want to use positional encodings
        positional_encodings = self.create_positional_encodings(
            self.max_length, self.embed_dim
        )
        positional_encodings = positional_encodings.repeat(x.shape[0], 1, 1).to(
            x.device
        )
        embeddings += positional_encodings

        embeddings = embeddings.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(embeddings)

        # Average over all sequence positions to get a fixed-sized representation
        sequence_representation = transformer_output.mean(dim=0)

        output = self.output_layer(sequence_representation)

        return output
