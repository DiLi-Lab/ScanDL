from typing import Tuple, List, Union, Dict, Optional
import numpy as np
import pandas as pd
import torch
import pickle
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast

### not in use!!
def attention_visualization_complete(
        attention_scores: Tuple[torch.Tensor],
        sn_sp_repr: torch.tensor,
        words_for_mapping: List[str],
        sn_repr_len: List[int],
        sn_input_ids: torch.tensor,
        tokenizer: BertTokenizerFast,
        path_to_dir: str = './',
        aggregate: bool = True,
        inference: bool = True,
        ticks: str = 'words',  # 'words', 'word_ids'
):
    """

    :param attention_scores: tuple of len(no transformer layers);
        tensor of shape [batch size, attention heads, seq_len, seq_len]
    :param sn_sp_repr: the combined representation of sentence and scanpath, in word IDs. shape [batch size, seq_len].
        at training time this is sn-sp, at inference time the sp part is noised
    :param words_for_mapping: the original sentences, list of len batch size
    :param sn_repr_len: length of each sentence (in subwords) in sn_sp_repr, shape [batch size]
    :param aggregate: aggregating over the attention heads or plotting each one individually
    :param inference: plot attention scores during inference. if false, during testing
    :param ticks: whether to use words/subwords as ticks or word IDs.
    """
    path_to_dir = os.path.join(path_to_dir, 'heatmaps')
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    # iterating through layers; only first and last
    for layer_idx, layer in [(0, 'first'), (len(attention_scores)-1, 'last')]:
        attention_scores_layer = attention_scores[layer_idx]

        # iterating through instances in batch
        for i in range(sn_sp_repr.shape[0]):

            subwords_pad = tokenizer.convert_ids_to_tokens(sn_input_ids[i])
            subwords = [w for w in subwords_pad if w != '[PAD]']
            sn_word_ids = [word_id.item() for word_id in sn_sp_repr[i][:sn_repr_len[i]]]

            if inference:
                words = subwords
                word_ids = sn_word_ids

            else:
                split_words = words_for_mapping[i].split()
                cur_word_ids_sp = sn_sp_repr[i][sn_repr_len[i]:]
                cur_words_sp = [split_words[word_id] for word_id in cur_word_ids_sp]
                # remove padding
                words_sp = [w for w in cur_words_sp if w != '[PAD]']
                word_ids_sp = [word_id.item() for word_id in cur_word_ids_sp if word_id.item() != 511]

                words = subwords + words_sp
                word_ids = sn_word_ids + word_ids_sp

            if ticks == 'words':
                ticks_axes = words
            else:
                ticks_axes = word_ids
            n_ticks = len(ticks_axes)

            # take the mean attention scores over all heads
            if aggregate:

                filename = f'att-heatmap_aggregated_instance-{i}_layer-{layer_idx + 1}.png'
                save_path = os.path.join(path_to_dir, filename)

                aggregated_scores = torch.mean(attention_scores_layer[i], dim=0)
                fig, ax = plt.subplots(figsize=(n_ticks / 2, n_ticks / 2))
                cax = ax.imshow(aggregated_scores[:len(ticks_axes), :len(ticks_axes)].T.cpu().detach().numpy(),
                                cmap='gist_ncar', vmin=0., vmax=1.)
                cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
                cbar.ax.set_yticklabels(['0', '0.5', '1'])
                cbar.ax.get_yaxis().labelpad = 15
                cbar.ax.set_ylabel('Attention scores', rotation=270)
                plt.yticks([j for j in range(len(ticks_axes))], ticks_axes)
                plt.xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                ax_t = ax.secondary_xaxis('top')
                ax_t.set_xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                ax_t.set_xticks([j for j in range(len(ticks_axes))])
                ax_t.set_xticklabels(ticks_axes, rotation=90)
                plt.ylabel('sentence-scanpath-sequence')
                plt.xlabel('sentence-scanpath-sequence')
                plt.title(f'Aggregated self-attention scores layer {layer_idx + 1}')
                plt.savefig(save_path)
                plt.clf()
                plt.close()

            # plot the heatmap for each attention head individually
            else:
                for head_idx in range(attention_scores_layer.shape[1]):
                    filename = f'att-heatmap_not-aggregated_instance-{i}_layer-{layer_idx + 1}_head-{head_idx + 1}.png'
                    save_path = os.path.join(path_to_dir, filename)

                    attention_scores_head = attention_scores_layer[i][head_idx]
                    fig, ax = plt.subplots(figsize=(n_ticks / 2, n_ticks / 2))
                    cax = ax.imshow(attention_scores_head[:len(ticks_axes), :len(ticks_axes)].T.cpu().detach().numpy(),
                                    cmap='gist_ncar', vmin=0., vmax=1.)
                    cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
                    cbar.ax.set_yticklabels(['0', '0.5', '1'])
                    cbar.ax.get_yaxis().labelpad = 15
                    cbar.ax.set_ylabel('Attention scores', rotation=270)
                    plt.yticks([j for j in range(len(ticks_axes))], ticks_axes)
                    plt.xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                    ax_t = ax.secondary_xaxis('top')
                    ax_t.set_xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                    ax_t.set_xticks([j for j in range(len(ticks_axes))])
                    ax_t.set_xticklabels(ticks_axes, rotation=90)
                    plt.ylabel('sentence-scanpath-sequence')
                    plt.xlabel('sentence-scanpath-sequence')
                    plt.title(f'Self-attention scores layer {layer_idx + 1}, head {head_idx + 1}')
                    plt.savefig(save_path)
                    plt.clf()
                    plt.close()


def attention_visualization(
        attention_scores: Tuple[torch.Tensor],
        subwords_list: List[List[str]],
        batch_idx: int,
        rank: int,
        denoising_step: int,
        path_to_dir: str = './',
        aggregate: bool = True,
        atten_vis_sp: bool = False,
        pred_sp_words: Optional[List[List[str]]] = None,
):
    """

    :param attention_scores: tuple of len(no transformer layers);
        tensor of shape [batch size, attention heads, seq_len, seq_len]
    :param sn_sp_repr: the combined representation of sentence and scanpath, in word IDs. shape [batch size, seq_len].
        at training time this is sn-sp, at inference time the sp part is noised
    :param words_for_mapping: the original sentences, list of len batch size
    :param sn_repr_len: length of each sentence (in subwords) in sn_sp_repr, shape [batch size]
    :param aggregate: aggregating over the attention heads or plotting each one individually
    :param inference: plot attention scores during inference. if false, during testing
    :param ticks: whether to use words/subwords as ticks or word IDs.
    :param atten_vis_sp: if given, the attention heatmap is done for the concatenation of both sentence and scanpath,
    and only for the last denoising step t=0.
    :param pred_sp_words:
    """

    if aggregate:
        if atten_vis_sp:
            path_to_dir = os.path.join(path_to_dir, 'heatmaps/sn_sp')
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)
        else:
            path_to_dir = os.path.join(path_to_dir, 'heatmaps')
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)
    else:
        path_to_dir = os.path.join(path_to_dir, 'heatmaps/attention_heads')
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

    # iterating through layers; only first and last
    for layer_idx, layer in [(0, 'first'), (len(attention_scores)-1, 'last')]:
        attention_scores_layer = attention_scores[layer_idx]

        # iterating through instances in batch
        for i in range(len(subwords_list)):

            subwords = [w for w in subwords_list[i] if w != '[PAD]']
            ticks_axes = subwords
            n_ticks = len(subwords)

            # take the mean attention scores over all heads
            if aggregate:

                if not atten_vis_sp:

                    filename = f'att-heatmap_aggregated_rank{rank}_batch{batch_idx+1}_instance{i+1}_layer{layer_idx+1}_t{denoising_step}.png'
                    save_path = os.path.join(path_to_dir, filename)
                    x_label, y_label = 'Sentence', 'Sentence'

                else:

                    words_scanpath = pred_sp_words[i]
                    ticks_axes = subwords + words_scanpath
                    n_ticks = len(ticks_axes)

                    filename = f'att-heatmap_aggregated_rank{rank}_batch{batch_idx + 1}_instance{i + 1}_layer{layer_idx + 1}_t0_sn-sp-concat.png'
                    save_path = os.path.join(path_to_dir, filename)

                    x_label, y_label = 'Sentence-Scanpath', 'Sentence-Scanpath'

                aggregated_scores = torch.mean(attention_scores_layer[i], dim=0)
                fig, ax = plt.subplots(figsize=(n_ticks / 2, n_ticks / 2))
                cax = ax.imshow(aggregated_scores[:len(ticks_axes), :len(ticks_axes)].T.cpu().detach().numpy(),
                                cmap='gist_ncar', vmin=0., vmax=1.)
                cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
                cbar.ax.set_yticklabels(['0', '0.5', '1'])
                cbar.ax.get_yaxis().labelpad = 15
                cbar.ax.set_ylabel('Attention scores', rotation=270)
                plt.yticks([j for j in range(len(ticks_axes))], ticks_axes)
                plt.xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                ax_t = ax.secondary_xaxis('top')
                ax_t.set_xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                ax_t.set_xticks([j for j in range(len(ticks_axes))])
                ax_t.set_xticklabels(ticks_axes, rotation=90)
                plt.ylabel(y_label)
                plt.xlabel(x_label)
                plt.title(f'Aggregated self-attention scores layer {layer_idx + 1}')
                plt.savefig(save_path)
                plt.clf()
                plt.close()



            # plot the heatmap for each attention head individually
            else:

                if atten_vis_sp:
                    raise NotImplementedError('plotting individual attention heads for the concatenation of sentence'
                                              ' and scanpath is not implemented.')

                for head_idx in range(attention_scores_layer.shape[1]):
                    filename = f'att-heatmap_not-aggregated_rank{rank}_batch{batch_idx+1}_instance{i+1}_layer{layer_idx+1}_head{head_idx+1}_t{denoising_step}.png'
                    save_path = os.path.join(path_to_dir, filename)

                    attention_scores_head = attention_scores_layer[i][head_idx]
                    fig, ax = plt.subplots(figsize=(n_ticks / 2, n_ticks / 2))
                    cax = ax.imshow(attention_scores_head[:len(ticks_axes), :len(ticks_axes)].T.cpu().detach().numpy(),
                                    cmap='gist_ncar', vmin=0., vmax=1.)
                    cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
                    cbar.ax.set_yticklabels(['0', '0.5', '1'])
                    cbar.ax.get_yaxis().labelpad = 15
                    cbar.ax.set_ylabel('Attention scores', rotation=270)
                    plt.yticks([j for j in range(len(ticks_axes))], ticks_axes)
                    plt.xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                    ax_t = ax.secondary_xaxis('top')
                    ax_t.set_xticks([j for j in range(len(ticks_axes))], [j for j in range(len(ticks_axes))])
                    ax_t.set_xticks([j for j in range(len(ticks_axes))])
                    ax_t.set_xticklabels(ticks_axes, rotation=90)
                    plt.ylabel('Sentence')
                    plt.xlabel('Sentence')
                    plt.title(f'Self-attention scores layer {layer_idx + 1}, head {head_idx + 1}')
                    plt.savefig(save_path)
                    plt.clf()
                    plt.close()




