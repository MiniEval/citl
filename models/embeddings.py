import math

from torch import nn
import torch


# Concatenated Temporal Embedding
class TemporalEmbedding(nn.Module):
    def __init__(self, dimension, length, concat=False, concat_length=None):
        super(TemporalEmbedding, self).__init__()

        self.concat = concat
        self.concat_length = concat_length
        if self.concat_length is None:
            self.concat_length = dimension

        # Compute the positional encodings once in log space.
        pe = torch.zeros(length, dimension, dtype=torch.float32)
        pe.require_grad = False

        position = torch.arange(0, length).float().unsqueeze(1)

        div_term = (torch.arange(0, dimension, 2).float() * -(math.log(10000.0) / dimension)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, indices=None):
        if indices is None:
            pe = self.pe[:, :x.size(1)].expand(x.shape)
        else:
            pe = torch.reshape(self.pe[0][indices], x.shape)

        if self.concat:
            x = torch.cat((x, pe[:, :, :self.concat_length]), dim=-1)
        else:
            x += pe

        return x


class PoseEmbedding(nn.Module):
    def __init__(self, embed_size, pose_data_size, max_length=72, concat_length=16):
        super(PoseEmbedding, self).__init__()

        self.embed_in = nn.Linear(pose_data_size, embed_size, bias=False)
        self.te = TemporalEmbedding(embed_size, max_length, concat=True, concat_length=concat_length)

        self.embed_out = nn.Linear(embed_size + self.te.concat_length, embed_size)

    def forward(self, x, indices=None):
        x = self.embed_in(x)
        x = self.te(x, indices)
        x = self.embed_out(x)
        return x
