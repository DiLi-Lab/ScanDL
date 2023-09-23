"""
Various utilities for neural networks.
"""

import math

import torch
import torch as th
import torch.nn as nn

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


def concatenate_sn_sp(sn_repr_emb, sp_repr_emb, sn_repr_len):
    # Create an empty tensor to store the concatenated embeddings
    concat_emb = torch.empty_like(sn_repr_emb)

    # Iterate over the batch size
    for i in range(sn_repr_emb.size(0)):
        # Get the true sequence length for the sn embedding
        seq_len = sn_repr_len[i]
        # get the true sequence length for the sp embedding
        sp_idx = sp_repr_emb.size(1) - seq_len
        # Cut off the extra dimensions in sn_repr_emb and sp_repr_emb
        sn_emb = sn_repr_emb[i, :seq_len]
        sp_emb = sp_repr_emb[i, :sp_idx]

        # Concatenate the embeddings along the sequence length dimension
        concat_emb[i] = torch.cat([sn_emb, sp_emb], dim=0)

    return concat_emb



def split_into_sn_and_sp(model_output, sn_repr_len):
    # create an empty tensor to store the split sn_repr
    model_output_sn = torch.empty_like(model_output)
    model_output_sp = torch.empty_like(model_output)

    microbatch_size, seq_len, emb_dim = model_output.shape

    model_output_sn_mask = torch.empty((microbatch_size, seq_len))
    model_output_sp_mask = torch.empty((microbatch_size, seq_len))

    # iterate over the batc size
    for i in range(microbatch_size):
        # the start index of the sp output is the length of the sn
        orig_sn_len = sn_repr_len[i]

        # the SP
        # split the sp_repr of the current instance off from the model output
        sp_repr_out = model_output[i, orig_sn_len:]
        # TODO change this to more sensible padding
        # pad the sp representation model output until it is again args.seq_len long
        sp_padding = sp_repr_out[-1].repeat(orig_sn_len, 1)
        # concatenate the embeddings along the
        model_output_sp[i] = torch.cat([sp_repr_out, sp_padding], dim=0)

        # the SN
        # split the sn_repr of the current instance off from the model output
        sn_repr_out = model_output[i, :orig_sn_len]
        sn_padding = sn_repr_out[-1].repeat(seq_len-orig_sn_len, 1)
        model_output_sn[i] = torch.cat([sn_repr_out, sn_padding], dim=0)

        # the masks: they only mask the additional padding added now, not the original padding added after the sp
        sp_mask = torch.ones(seq_len).to(model_output.device)
        sp_mask[-orig_sn_len:] = 0
        model_output_sp_mask[i] = sp_mask
        sn_mask = torch.ones(seq_len).to(model_output.device)
        sn_mask[orig_sn_len:] = 0
        model_output_sn_mask[i] = sn_mask

    model_output_sn_mask = model_output_sn_mask.to(model_output.device)
    model_output_sp_mask = model_output_sp_mask.to(model_output.device)

    return model_output_sn, model_output_sp, model_output_sn_mask, model_output_sp_mask

