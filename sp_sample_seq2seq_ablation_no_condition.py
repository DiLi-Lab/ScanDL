"""
Run inference on ScanDL.
Ablation: without condition (sentence): unconditional scanpath generation (New Reader/New Sentence)
"""

import argparse
import os, json
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import set_seed
from datasets import load_from_disk
from functools import partial
from transformers import BertTokenizerFast, AutoConfig

from scandl.sp_rounding import denoised_fn_round
from scandl.utils import dist_util, logger
from scandl.utils.nn import *
from sp_load_celer_zuco_ablation_no_condition import celer_zuco_dataset_and_loader
from visualizations.attention_visualization import attention_visualization
from sp_basic_utils_ablation_no_condition import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)


def get_parser() -> argparse.ArgumentParser:
    defaults = dict(
        model_path='',
        step=0,
        out_dir='',
        top_p=0,
        clamp_first='yes',
        test_set_sns='mixed',
        atten_vis=False,
        notes='-',
        tsne_vis=False,
        sp_vis=False,
        no_inst=0,
        atten_vis_sp=False,
    )
    decode_defaults = dict(
        split='valid',
        clamp_step=0,
        seed2=105,
        clip_denoised=False,
    )
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@torch.no_grad()
def main():

    args = get_parser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1  # the number of processes in the current process group
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    if args.clamp_first == 'yes':
        clamp_first_bool = True
    else:
        clamp_first_bool = False

    # set mask_padding to False (otherwise would be leak)
    args.mask_padding = False

    tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

    args.vocab_size = tokenizer.vocab_size

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location='cpu')
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    model_config = AutoConfig.from_pretrained(args.config_name)

    sn_sp_repr_embedding = nn.Embedding(
        num_embeddings=args.hidden_t_dim,
        embedding_dim=args.hidden_dim,
        _weight=model.sn_sp_repr_embedding.weight.clone().cpu(),
    )

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    if args.inference != 'cv':   # cross-dataset setting: train on celer, test on zuco
        raise SystemExit('these ablation runs should be done on CV on celer only.')

    else:  # inference on the cross-validation settings

        # load the test data that was split off from all data during training
        test_data = load_from_disk(f'{args.checkpoint_path}/test_data')

        test_loader = celer_zuco_dataset_and_loader(
            data=test_data,
            data_args=args,
            split='test',
            loop=False,
        )

        model_base_name = args.model_path.split('/')[3]
        if args.notes == '-':
            out_dir = os.path.join('generation_outputs', args.model_path.split('/')[1], args.model_path.split('/')[2])
        else:
            out_dir = os.path.join('generation_outputs', args.model_path.split('/')[1] + f'_{args.notes}', args.model_path.split('/')[2])

        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        out_path = os.path.join(out_dir, f'{model_base_name}.samples')
        if rank == 0:
            if not os.path.exists(out_path):
                os.makedirs(out_path)

        out_path_incrementally_pad = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}_clamp-first-{args.clamp_first}_running_remove-PAD_rank{rank}.json")
        out_path_all_pad = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}_clamp-first-{args.clamp_first}_all_remove-PAD_rank{rank}.json")
        out_path_incrementally_sep = os.path.join(out_path,  f"seed{args.seed2}_step{args.clamp_step}_clamp-first-{args.clamp_first}_running_cut-off-SEP_rank{rank}.json")
        out_path_all_sep = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}_clamp-first-{args.clamp_first}_all_cut-off-SEP_rank{rank}.json")
        if os.path.exists(out_path_all_pad):
            print(' --- inference has already been done on this output.')
            sys.exit()

    start_t = time.time()

    all_test_data = []
    idx = 0

    try:
        while True:
            batch = next(test_loader)
            if idx % world_size == rank:  # split data per nodes
                all_test_data.append(batch)
            idx += 1
    except StopIteration:
        print('### End of reading iteration ...')

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # dummy data for Remainder: for dist.barrier()   ??

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    predicted_sp_words_pad = []
    predicted_sp_ids_pad = []
    original_sp_words_pad = []
    original_sp_ids_pad = []
    original_sn_pad = []
    sn_ids_pad = []
    reader_ids_pad = []

    predicted_sp_words_sep = []
    predicted_sp_ids_sep = []
    original_sp_words_sep = []
    original_sp_ids_sep = []
    original_sn_sep = []
    sn_ids_sep = []
    reader_ids_sep = []

    data_dict_initial_pad = {
        'predicted_sp_words': list(),
        'original_sp_words': list(),
        'predicted_sp_ids': list(),
        'original_sp_ids': list(),
        'original_sn': list(),
        'sn_ids': list(),
        'reader_ids': list(),
    }
    data_dict_initial_sep = {
        'predicted_sp_words': list(),
        'original_sp_words': list(),
        'predicted_sp_ids': list(),
        'original_sp_ids': list(),
        'original_sn': list(),
        'sn_ids': list(),
        'reader_ids': list(),
    }

    for batch_idx, batch in enumerate(iterator):

        if not batch:
            for i in range(world_size):
                dist.barrier()
            continue

        sp_word_ids = batch['sp_word_ids'].to(dist_util.dev())
        sn_input_ids = batch['sn_input_ids'].to(dist_util.dev())
        indices_pos_enc = batch['indices_pos_enc'].to(dist_util.dev())
        words_for_mapping = batch['words_for_mapping']
        sn_ids = batch['sn_ids']
        reader_ids = batch['reader_ids']

        # needed for attention visualisation
        subwords = [tokenizer.convert_ids_to_tokens(i) for i in sn_input_ids]

        sn_sp_emb, pos_enc = model.get_embeds(
            sn_sp_repr=sp_word_ids,
            indices_pos_enc=indices_pos_enc,
        )

        x_start = sn_sp_emb
        noise = torch.randn_like(x_start)

        # replace the scan path with Gaussian noise
        x_noised = noise

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model=model,
            shape=sample_shape,
            noise=x_noised,
            pos_enc=pos_enc,
            mask_sn_padding=None,
            mask_transformer_att=None,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, sn_sp_repr_embedding),
            model_kwargs=None,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=clamp_first_bool,
            mask=None,
            x_start=None,
            subwords_list=subwords,
            atten_vis=args.atten_vis,
            atten_vis_fn=attention_visualization,
            atten_vis_path=out_path,
            batch_idx=batch_idx,
            gap=step_gap,
        )

        # len(samples) is the number of diffusion steps
        print(samples[0].shape)  # samples for each diffusion step; [batch_size, args.seq_len, hidden=768]

        # take the output of the very last denoising step
        sample = samples[-1]

        print('decoding for seq2seq')
        print(sample.shape)

        logits = model.get_logits(sample)
        cands = torch.topk(logits, k=1, dim=-1)

        for instance_idx, (pred_seq, orig_words, orig_ids, sn_id, reader_id) in enumerate(zip(
                cands.indices, words_for_mapping, sp_word_ids, sn_ids, reader_ids
        )):

            pred_seq_sp = pred_seq
            orig_ids_sp = orig_ids

            words_split = orig_words.split()
            predicted_sp = [words_split[i] for i in pred_seq_sp]
            pred_sp_ids = [e.item() for e in pred_seq_sp]
            original_sp = [words_split[i] for i in orig_ids_sp]
            orig_sp_ids = [e.item() for e in orig_ids_sp]

            ### cut off all trailing PAD tokens ###

            while len(predicted_sp) > 1 and predicted_sp[-1] == '[PAD]':
                predicted_sp.pop()
            while len(pred_sp_ids) > 1 and pred_sp_ids[-1] == args.seq_len-1:
                pred_sp_ids.pop()
            while len(original_sp) > 1 and original_sp[-1] == '[PAD]':
                original_sp.pop()
            while len(orig_sp_ids) > 1 and orig_sp_ids[-1] == args.seq_len-1:
                orig_sp_ids.pop()
            while len(words_split) > 1 and words_split[-1] == '[PAD]':
                words_split.pop()

            # predicted scan path in words
            predicted_sp_words_pad.append(' '.join(predicted_sp))
            # original scan path in words
            original_sp_words_pad.append(' '.join(original_sp))
            # predicted scan path IDs
            predicted_sp_ids_pad.append(pred_sp_ids)
            # true/gold label scan path IDs
            original_sp_ids_pad.append(orig_sp_ids)

            # add the original sn
            original_sn_pad.append(' '.join(words_split))

            # add the IDs to the list
            sn_ids_pad.append(sn_id)
            reader_ids_pad.append(reader_id.item())

            if batch_idx == 0 and instance_idx == 0:

                data_dict_initial_pad['predicted_sp_words'].append(' '.join(predicted_sp))
                data_dict_initial_pad['original_sp_words'].append(' '.join(original_sp))
                data_dict_initial_pad['predicted_sp_ids'].append(pred_sp_ids)
                data_dict_initial_pad['original_sp_ids'].append(orig_sp_ids)
                data_dict_initial_pad['original_sn'].append(' '.join(words_split))
                data_dict_initial_pad['sn_ids'].append(sn_id)
                data_dict_initial_pad['reader_ids'].append(reader_id.item())
                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_pad, 'w') as f:
                            json.dump(data_dict_initial_pad, f)
                    dist.barrier()

            else:

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_pad, 'rb') as f:
                            loaded_data_dict = json.load(f)
                    dist.barrier()

                loaded_data_dict['predicted_sp_words'].append(' '.join(predicted_sp))
                loaded_data_dict['original_sp_words'].append(' '.join(original_sp))
                loaded_data_dict['predicted_sp_ids'].append(pred_sp_ids)
                loaded_data_dict['original_sp_ids'].append(orig_sp_ids)
                loaded_data_dict['original_sn'].append(' '.join(words_split))
                loaded_data_dict['sn_ids'].append(sn_id)
                loaded_data_dict['reader_ids'].append(reader_id.item())

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_pad, 'w') as f:
                            json.dump(loaded_data_dict, f)
                    dist.barrier()


            ### cut off at first SEP token ###

            words_split = orig_words.split()
            # map the predicted indices to the words
            predicted_sp = [words_split[i] for i in pred_seq_sp]
            # the predicted sp IDs
            pred_sp_ids = [e.item() for e in pred_seq_sp]
            # original scan path in words
            original_sp = [words_split[i] for i in orig_ids_sp]
            # the original sp IDs
            orig_sp_ids = [e.item() for e in orig_ids_sp]

            while len(original_sp) > 1 and original_sp[-1] == '[PAD]':
                original_sp.pop()
            while len(orig_sp_ids) > 1 and orig_sp_ids[-1] == args.seq_len-1:
                orig_sp_ids.pop()
            while len(words_split) > 1 and words_split[-1] == '[PAD]':
                words_split.pop()
            sep_token_id = orig_sp_ids[-1]

            for idx, predicted_sp_id in enumerate(pred_sp_ids):
                if predicted_sp_id == sep_token_id:
                    pred_sp_ids = pred_sp_ids[:idx+1]
                    break

            for idx, predicted_sp_word in enumerate(predicted_sp):
                if predicted_sp_word == '[SEP]':
                    predicted_sp = predicted_sp[:idx+1]
                    break

            # predicted scan path in words
            predicted_sp_words_sep.append(' '.join(predicted_sp))
            # original scan path in words
            original_sp_words_sep.append(' '.join(original_sp))
            # predicted scan path IDs
            predicted_sp_ids_sep.append(pred_sp_ids)
            # true/gold label scan path IDs
            original_sp_ids_sep.append(orig_sp_ids)

            # add the original sn
            original_sn_sep.append(' '.join(words_split))

            # add the IDs to the list
            sn_ids_sep.append(sn_id)
            reader_ids_sep.append(reader_ids)

            if batch_idx == 0 and instance_idx == 0:

                data_dict_initial_sep['predicted_sp_words'].append(' '.join(predicted_sp))
                data_dict_initial_sep['original_sp_words'].append(' '.join(original_sp))
                data_dict_initial_sep['predicted_sp_ids'].append(pred_sp_ids)
                data_dict_initial_sep['original_sp_ids'].append(orig_sp_ids)
                data_dict_initial_sep['original_sn'].append(' '.join(words_split))
                data_dict_initial_sep['sn_ids'].append(sn_id)
                data_dict_initial_sep['reader_ids'].append(reader_id.item())

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_sep, 'w') as f:
                            json.dump(data_dict_initial_sep, f)
                    dist.barrier()

            else:

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_sep, 'rb') as f:
                            loaded_data_dict = json.load(f)
                    dist.barrier()

                loaded_data_dict['predicted_sp_words'].append(' '.join(predicted_sp))
                loaded_data_dict['original_sp_words'].append(' '.join(original_sp))
                loaded_data_dict['predicted_sp_ids'].append(pred_sp_ids)
                loaded_data_dict['original_sp_ids'].append(orig_sp_ids)
                loaded_data_dict['original_sn'].append(' '.join(words_split))
                loaded_data_dict['sn_ids'].append(sn_id)
                loaded_data_dict['reader_ids'].append(reader_id.item())

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_sep, 'w') as f:
                            json.dump(loaded_data_dict, f)
                    dist.barrier()

        del batch
        del sp_word_ids
        del sn_input_ids
        del indices_pos_enc
        del words_for_mapping
        del x_start
        del noise
        del x_noised
        del samples
        del sample

    data_dict_complete_pad = {
        'predicted_sp_words': predicted_sp_words_pad,
        'original_sp_words': original_sp_words_pad,
        'predicted_sp_ids': predicted_sp_ids_pad,
        'original_sp_ids': original_sp_ids_pad,
        'original_sn': original_sn_pad,
        'sn_ids': sn_ids_pad,
        'reader_ids': reader_ids_pad,
    }
    data_dict_complete_sep = {
        'predicted_sp_words': predicted_sp_words_sep,
        'original_sp_words': original_sp_words_sep,
        'predicted_sp_ids': predicted_sp_ids_sep,
        'original_sp_ids': original_sp_ids_sep,
        'original_sn': original_sn_sep,
        'sn_ids': sn_ids_sep,
        'reader_ids': reader_ids_sep,
    }

    for i in range(world_size):
        if i == rank:  # write files sequentially
            with open(out_path_all_pad, 'w') as f:
                json.dump(data_dict_complete_pad, f)
        dist.barrier()
    for i in range(world_size):
        if i == rank:  # write files sequentially
            with open(out_path_all_sep, 'w') as f:
                json.dump(data_dict_complete_sep, f)
        dist.barrier()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == '__main__':
    main()
