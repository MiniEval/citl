from torch import nn
import torch
from models.embeddings import PoseEmbedding, TemporalEmbedding
from models.rmsnorm import RMSNorm


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_size=None, dropout=0.1):
        super(FeedForward, self).__init__()
        if ff_size is None:
            ff_size = 4 * embed_size
        self.w1 = nn.Linear(embed_size, ff_size)
        self.activation = nn.GELU()
        self.w2 = nn.Linear(ff_size, embed_size)
        self.norm = RMSNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ff = x
        ff = self.w1(ff)
        ff = self.activation(ff)
        ff = self.w2(ff)
        ff = self.dropout(ff) + x
        ff = self.norm(ff)
        return ff


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        self.mhsa = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        self.norm = RMSNorm(embed_size)
        # self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None, mask=None):
        # q_norm = self.norm(q.transpose(-2, -1)).transpose(-2, -1)
        if k is None or v is None:
            attn, _ = self.mhsa(q, q, q, attn_mask=mask)
        else:
            attn, _ = self.mhsa(q, k, v, attn_mask=mask)
        attn = self.dropout(attn) + q
        attn = self.norm(attn)
        return attn


class KeyframeEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads=8, dropout=0.1):
        super(KeyframeEncoderBlock, self).__init__()

        self.self_attn = MultiheadAttention(embed_size, heads, dropout=dropout)
        self.ff = FeedForward(embed_size, dropout=dropout)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.ff(x)

        return x


class KeyframeEncoder(nn.Module):
    def __init__(self, embed_size, pose_data_size, max_length=72, heads=8, layers=6, dropout=0.1):
        super(KeyframeEncoder, self).__init__()

        self.embed = PoseEmbedding(embed_size, pose_data_size, max_length, concat_length=16)
        self.blocks = nn.ModuleList([KeyframeEncoderBlock(embed_size, heads, dropout=dropout) for _ in range(layers)])

    def forward(self, x, indices):
        # [batches, frames, pose data]
        x = self.embed(x, indices)

        for block in self.blocks:
            x = block(x)

        return x


class IntermediateEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads=8, dropout=0.1):
        super(IntermediateEncoderBlock, self).__init__()

        self.attn = MultiheadAttention(embed_size, heads, dropout=dropout)
        self.ff = FeedForward(embed_size, dropout=dropout)

    def forward(self, x, key_enc):
        x = self.attn(x, key_enc, key_enc)
        x = self.ff(x)

        return x


class IntermediateEncoder(nn.Module):
    def __init__(self, embed_size, n_te=16, max_length=72, heads=8, layers=6, dropout=0.1):
        super(IntermediateEncoder, self).__init__()

        self.n_te = n_te

        self.te = TemporalEmbedding(self.n_te, max_length)
        self.te_embed = nn.Linear(16, embed_size)

        self.blocks = nn.ModuleList([IntermediateEncoderBlock(embed_size, heads, dropout=dropout) for _ in range(layers)])

    def forward(self, indices, key_enc):
        # indices: [batches, intermediate frames], CPU
        # key_enc: [batches, key frames, encodings]
        zeros = torch.zeros((*indices.shape, self.n_te), device=key_enc.device)
        x = self.te(zeros)
        x = self.te_embed(x)

        for block in self.blocks:
            x = block(x, key_enc)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads=8, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.self_attn = MultiheadAttention(embed_size, heads, dropout=dropout)
        self.ff = FeedForward(embed_size, dropout=dropout)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.ff(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embed_size, output_size, heads=8, layers=6, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size

        self.key_ff = FeedForward(embed_size, dropout=dropout)

        self.conv_in = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1, padding_mode="replicate")
        self.blocks = nn.ModuleList([DecoderBlock(embed_size, heads, dropout=dropout) for _ in range(layers)])
        self.conv_out = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1, padding_mode="replicate")

        self.out = nn.Linear(embed_size, output_size)

    def forward(self, interm_enc, interm_frames, key_enc, keyframes):
        # interm_enc: [batches, intermediate frames, encodings]
        # key_enc: [batches, key frames, encodings]
        # keyframes: [batches, key frames], CPU

        key_idx = torch.repeat_interleave(keyframes.unsqueeze(-1), key_enc.shape[-1], dim=-1)
        interm_idx = torch.repeat_interleave(interm_frames.unsqueeze(-1), interm_enc.shape[-1], dim=-1)

        x = torch.cat([key_enc, interm_enc], dim=1)
        idx = torch.cat([key_idx, interm_idx], dim=1)
        x = torch.gather(x, 1, idx)

        # Reformulation FFN
        x = self.key_ff(x)

        # Stage III
        x = self.conv_in(x.transpose(-2, -1)).transpose(-2, -1)
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x.transpose(-2, -1)).transpose(-2, -1)

        x = self.out(x)

        return x
