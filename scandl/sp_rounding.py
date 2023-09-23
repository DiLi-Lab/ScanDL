import numpy as np
import torch


def get_knn(model_emb, text_emb, dist='cos'):
    if dist == 'cos':
        adjacency = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
    elif dist == 'l2':
        adjacency = model_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
            model_emb.size(0), -1, -1)
        adjacency = -torch.norm(adjacency, dim=-1)
    topk_out = torch.topk(adjacency, k=6, dim=0)
    return topk_out.values, topk_out.indices


def get_efficient_knn(sn_sp_repr_embedding_weight, text_emb):
    """
    :param sn_sp_repr_embedding_weight:
    :param text_emb:
    """
    emb_norm = (sn_sp_repr_embedding_weight**2).sum(-1).view(-1, 1)
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
    dist = emb_norm + arr_norm.cpu().transpose(0, 1) - 2.0 * torch.mm(sn_sp_repr_embedding_weight, text_emb_t.cpu())  # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices


def denoised_fn_round(args, sn_sp_repr_embedding, text_emb, t):
    """
    :param sn_sp_repr_embedding: the weights/parameter of the embedding layer that embeds the concatenated word IDs
    :param text_emb: the model output at denoising step t; the transformer received the noise as input; this is the pred.
        shape [batch size, args.seq_len, hidden_dim=768]
    :param t: the current time step, shape [batch size] (same t for each instance in the batch)
    """
    sn_sp_repr_embedding_weight = sn_sp_repr_embedding.weight
    old_shape = text_emb.shape
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    text_emb.to(sn_sp_repr_embedding_weight.device)

    val, indices = get_efficient_knn(sn_sp_repr_embedding_weight=sn_sp_repr_embedding_weight, text_emb=text_emb)
    rounded_tokens = indices[0]
    new_embeds = sn_sp_repr_embedding(rounded_tokens).view(old_shape).to(old_device)
    return new_embeds




