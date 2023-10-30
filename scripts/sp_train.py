"""
Train ScanDL for scanpath generation conditioned on text input in the following settings:
New Reader
New Sentence
New Reader / New Sentence
Cross-Dataset
"""

import argparse
import json
import os
import numpy as np
import wandb

import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from transformers import set_seed, BertTokenizerFast
from datasets import load_from_disk

from scandl.utils import dist_util, logger
from scandl.step_sample import create_named_schedule_sampler
from scripts.sp_basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from scripts.sp_train_util import TrainLoop
from scripts.sp_load_celer_zuco import load_celer, load_celer_speakers, process_celer, celer_zuco_dataset_and_loader
from scripts.sp_load_celer_zuco import get_kfold, get_kfold_indices_combined
from scripts.sp_load_celer_zuco import flatten_data, unflatten_data


os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    """ Loads the config from the file scandl/config.json and adds all keys and values in the config dict
    to the argument parser where config values are the argparse arguments' default values. """
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()

    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse
    return parser


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)

    assert args.seq_len == args.hidden_t_dim

    # set up distributed processing group
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    rank = dist.get_rank() or 0

    tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

    args.vocab_size = tokenizer.vocab_size

    if rank == 0:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

    if args.inference != 'cv':  # Train in Cross-Dataset setting

        if args.load_train_data == '-':
            if args.inference == 'zuco':

                if args.corpus == 'zuco':
                    raise RuntimeError('Training and inference cannot be done on the same corpus if not CV.')

                else:
                    split_sizes = {'val_size': 0.1}

                    if args.corpus == 'celer':
                        word_info_df, eyemovement_df = load_celer()
                        reader_list = load_celer_speakers(only_native_speakers=args.celer_only_L1)
                        # list with unique sentences / sentence IDs
                        sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

                        train_data, val_data = process_celer(
                            sn_list=sn_list,
                            reader_list=reader_list,
                            word_info_df=word_info_df,
                            eyemovement_df=eyemovement_df,
                            tokenizer=tokenizer,
                            args=args,
                            split='train-val',
                            split_sizes=split_sizes,
                            splitting_criterion=args.data_split_criterion,
                        )
                        train_data.save_to_disk(os.path.join(args.checkpoint_path, 'train_data'))
                        val_data.save_to_disk(os.path.join(args.checkpoint_path, 'val_data'))

        else:  # load preprocessed and saved data

            path_to_data_dir = os.path.join(args.load_train_data, 'cross_dataset')
            train_data = load_from_disk(os.path.join(path_to_data_dir, 'train_data'))
            val_data = load_from_disk(os.path.join(path_to_data_dir, 'val_data'))
            print('\t\t--- load data from disk!')

        train_loader = celer_zuco_dataset_and_loader(
            data=train_data,
            data_args=args,
            split='train',
        )
        val_loader = celer_zuco_dataset_and_loader(
            data=val_data,
            data_args=args,
            split='val',
        )

        logger.log("### Creating model and diffusion...")
        if torch.cuda.is_available():
            print('#' * 30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, load_defaults_config().keys())
        )

        if args.load_from_checkpoint:
            print('\t--- load model from pretrained checkpoint:', end=' ')
            path_to_pretrained = args.load_trained_model
            model.load_state_dict(
                dist_util.load_state_dict(path_to_pretrained)
            )
            print(path_to_pretrained)

        model.to(dist_util.dev())

        pytorch_total_params = sum(p.numel() for p in model.parameters())

        logger.log(f'### The parameter count is {pytorch_total_params}')
        # args.schedule_sampler = lossaware
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
        with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "ScanDL"),
                name=args.checkpoint_path,
            )
            wandb.config.update(args.__dict__, allow_val_change=True)

        logger.log('### Training...')

        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=train_loader,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            learning_steps=args.learning_steps,
            checkpoint_path=args.checkpoint_path,
            gradient_clipping=args.gradient_clipping,
            eval_data=val_loader,
            eval_interval=args.eval_interval,
        ).run_loop()

    else:  # performing k-fold cross-validation (New Reader, New Sentence, New Reader/New Sentence)

        if args.corpus == 'zuco':
            raise RuntimeError('K-fold CV is only supposed to be for Celer.')

        else:

            word_info_df, eyemovement_df = load_celer()

            reader_list = load_celer_speakers(only_native_speakers=args.celer_only_L1)
            sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

            if args.load_train_data == '-':
                data, splitting_IDs_dict = process_celer(
                    sn_list=sn_list,
                    reader_list=reader_list,
                    word_info_df=word_info_df,
                    eyemovement_df=eyemovement_df,
                    tokenizer=tokenizer,
                    args=args,
                    inference='cv',
                )

                # flatten the data for subsequent splitting
                flattened_data = flatten_data(data)

                original_checkpoint_path = args.checkpoint_path

                if args.data_split_criterion != 'combined':

                    for fold_idx, (train_idx, test_idx) in enumerate(
                        get_kfold(
                            data=flattened_data,
                            splitting_IDs_dict=splitting_IDs_dict,
                            splitting_criterion=args.data_split_criterion,
                            n_splits=args.n_folds,
                        )
                    ):

                        fold_path = os.path.join(f'{original_checkpoint_path}', f'fold-{fold_idx}')

                        # change checkpoint path to CV fold path within checkpoint, so that models are saved to correct fold
                        args.checkpoint_path = fold_path

                        if rank == 0:  # avoid file exists errors when parallelising
                            if not os.path.exists(args.checkpoint_path):
                                os.makedirs(args.checkpoint_path)

                        train_data = np.array(flattened_data)[train_idx].tolist()
                        test_data = np.array(flattened_data)[test_idx].tolist()

                        train_data_save = unflatten_data(flattened_data=train_data, split='train')
                        train_data_save.save_to_disk(os.path.join(args.checkpoint_path, 'train_data'))

                        # split train data into train and val data
                        train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=77)

                        # unflatten the data
                        train_data = unflatten_data(flattened_data=train_data, split='train')
                        val_data = unflatten_data(flattened_data=val_data, split='val')
                        test_data = unflatten_data(flattened_data=test_data, split='test')

                        # save test data to disk for later inference
                        test_data.save_to_disk(os.path.join(args.checkpoint_path, 'test_data'))

                        train_loader = celer_zuco_dataset_and_loader(
                            data=train_data,
                            data_args=args,
                            split='train',
                        )
                        val_loader = celer_zuco_dataset_and_loader(
                            data=val_data,
                            data_args=args,
                            split='val',
                        )

                        logger.log("### Creating model and diffusion...")
                        if torch.cuda.is_available():
                            print('#' * 30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])

                        model, diffusion = create_model_and_diffusion(
                            **args_to_dict(args, load_defaults_config().keys())
                        )
                        model.to(dist_util.dev())

                        pytorch_total_params = sum(p.numel() for p in model.parameters())

                        logger.log(f'### The parameter count is {pytorch_total_params}')
                        # args.schedule_sampler = lossaware
                        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

                        logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
                        with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
                            json.dump(args.__dict__, f, indent=2)

                        if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
                            wandb.init(
                                project=os.getenv("WANDB_PROJECT", "ScanDL"),
                                name=args.checkpoint_path,
                            )
                            wandb.config.update(args.__dict__, allow_val_change=True)

                        logger.log('### Training...')

                        TrainLoop(
                            model=model,
                            diffusion=diffusion,
                            data=train_loader,
                            batch_size=args.batch_size,
                            microbatch=args.microbatch,
                            lr=args.lr,
                            ema_rate=args.ema_rate,
                            log_interval=args.log_interval,
                            save_interval=args.save_interval,
                            resume_checkpoint=args.resume_checkpoint,
                            use_fp16=args.use_fp16,
                            fp16_scale_growth=args.fp16_scale_growth,
                            schedule_sampler=schedule_sampler,
                            weight_decay=args.weight_decay,
                            learning_steps=args.learning_steps,
                            checkpoint_path=args.checkpoint_path,
                            gradient_clipping=args.gradient_clipping,
                            eval_data=val_loader,
                            eval_interval=args.eval_interval,
                        ).run_loop()

                else:  # New Reader/New Sentence setting

                    reader_indices, sentence_indices = get_kfold_indices_combined(
                        data=flattened_data,
                        splitting_IDs_dict=splitting_IDs_dict,
                    )

                    reader_IDs = splitting_IDs_dict['reader']
                    sn_IDs = splitting_IDs_dict['sentence']

                    for fold_idx, ((reader_train_idx, reader_test_idx), (sn_train_idx, sn_test_idx)) in enumerate(zip(reader_indices, sentence_indices)):

                        # create dir for each training fold
                        fold_path = os.path.join(f'{original_checkpoint_path}', f'fold-{fold_idx}')

                        # change checkpoint path to CV fold path within checkpoint, so that models are saved to correct fold
                        args.checkpoint_path = fold_path
                        if rank == 0:
                            if not os.path.exists(args.checkpoint_path):
                                os.makedirs(args.checkpoint_path)

                        # create data sets with only unique readers and sentences in test set
                        unique_reader_test_IDs = set(np.array(reader_IDs)[reader_test_idx].tolist())
                        unique_sn_test_IDs = set(np.array(sn_IDs)[sn_test_idx].tolist())
                        # subset the data: if an ID is both in the IDs for sentence and reader sampled for the test set,
                        # add the data point to the test data; if it is in neither of them, add to train data. if
                        # in one of them, unfortunately discard
                        train_data, test_data = list(), list()

                        for i in range(len(flattened_data)):
                            if reader_IDs[i] in unique_reader_test_IDs and sn_IDs[i] in unique_sn_test_IDs:
                                test_data.append(flattened_data[i])
                            elif reader_IDs[i] not in unique_reader_test_IDs and sn_IDs[i] not in unique_sn_test_IDs:
                                train_data.append(flattened_data[i])
                            else:
                                continue
                        train_data_save = unflatten_data(flattened_data=train_data, split='train')
                        train_data_save.save_to_disk(os.path.join(args.checkpoint_path, 'train_data'))

                        # split the train data into train and val
                        train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=77)

                        # unflatten the data
                        train_data = unflatten_data(flattened_data=train_data, split='train')
                        val_data = unflatten_data(flattened_data=val_data, split='val')
                        test_data = unflatten_data(flattened_data=test_data, split='test')

                        # save data to disk for later inference
                        test_data.save_to_disk(os.path.join(args.checkpoint_path, 'test_data'))

                        train_loader = celer_zuco_dataset_and_loader(
                            data=train_data,
                            data_args=args,
                            split='train',
                        )
                        val_loader = celer_zuco_dataset_and_loader(
                            data=val_data,
                            data_args=args,
                            split='val',
                        )

                        logger.log("### Creating model and diffusion...")
                        if torch.cuda.is_available():
                            print('#' * 30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])

                        model, diffusion = create_model_and_diffusion(
                            **args_to_dict(args, load_defaults_config().keys())
                        )
                        model.to(dist_util.dev())

                        pytorch_total_params = sum(p.numel() for p in model.parameters())

                        logger.log(f'### The parameter count is {pytorch_total_params}')
                        # args.schedule_sampler = lossaware
                        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

                        logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
                        with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
                            json.dump(args.__dict__, f, indent=2)

                        if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
                            wandb.init(
                                project=os.getenv("WANDB_PROJECT", "ScanDL"),
                                name=args.checkpoint_path,
                            )
                            wandb.config.update(args.__dict__, allow_val_change=True)

                        logger.log('### Training...')

                        TrainLoop(
                            model=model,
                            diffusion=diffusion,
                            data=train_loader,
                            batch_size=args.batch_size,
                            microbatch=args.microbatch,
                            lr=args.lr,
                            ema_rate=args.ema_rate,
                            log_interval=args.log_interval,
                            save_interval=args.save_interval,
                            resume_checkpoint=args.resume_checkpoint,
                            use_fp16=args.use_fp16,
                            fp16_scale_growth=args.fp16_scale_growth,
                            schedule_sampler=schedule_sampler,
                            weight_decay=args.weight_decay,
                            learning_steps=args.learning_steps,
                            checkpoint_path=args.checkpoint_path,
                            gradient_clipping=args.gradient_clipping,
                            eval_data=val_loader,
                            eval_interval=args.eval_interval,
                        ).run_loop()

            else:  # load preprocessed and saved data from disk

                original_checkpoint_path = args.checkpoint_path
                path_to_data_dir = os.path.join(args.load_train_data, args.data_split_criterion)
                folds = [dir for dir in os.listdir(path_to_data_dir) if dir.startswith('fold')]
                for fold_idx, fold in enumerate(folds):
                    path_to_data_fold = os.path.join(path_to_data_dir, fold)

                    fold_path = os.path.join(f'{original_checkpoint_path}', f'fold-{fold_idx}')
                    args.checkpoint_path = fold_path

                    if rank == 0:  # avoid file exists errors when parallelising
                        if not os.path.exists(args.checkpoint_path):
                            os.makedirs(args.checkpoint_path)

                    print('\t\t--- load train data from disk!')
                    train_data = load_from_disk(os.path.join(path_to_data_fold, 'train_data'))
                    print('\t\t--- loaded train data from disk!')
                    flattened_data = flatten_data(train_data['train'].to_dict())
                    train_data, val_data = train_test_split(flattened_data, test_size=0.1, shuffle=True, random_state=77)

                    # unflatten the data
                    train_data = unflatten_data(flattened_data=train_data, split='train')
                    val_data = unflatten_data(flattened_data=val_data, split='val')

                    train_loader = celer_zuco_dataset_and_loader(
                        data=train_data,
                        data_args=args,
                        split='train',
                    )
                    val_loader = celer_zuco_dataset_and_loader(
                        data=val_data,
                        data_args=args,
                        split='val',
                    )

                    logger.log("### Creating model and diffusion...")
                    if torch.cuda.is_available():
                        print('#' * 30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])

                    model, diffusion = create_model_and_diffusion(
                        **args_to_dict(args, load_defaults_config().keys())
                    )
                    model.to(dist_util.dev())

                    pytorch_total_params = sum(p.numel() for p in model.parameters())

                    logger.log(f'### The parameter count is {pytorch_total_params}')
                    # args.schedule_sampler = lossaware
                    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

                    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
                    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
                        json.dump(args.__dict__, f, indent=2)

                    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
                        wandb.init(
                            project=os.getenv("WANDB_PROJECT", "ScanDL"),
                            name=args.checkpoint_path,
                        )
                        wandb.config.update(args.__dict__, allow_val_change=True)

                    logger.log('### Training...')

                    TrainLoop(
                        model=model,
                        diffusion=diffusion,
                        data=train_loader,
                        batch_size=args.batch_size,
                        microbatch=args.microbatch,
                        lr=args.lr,
                        ema_rate=args.ema_rate,
                        log_interval=args.log_interval,
                        save_interval=args.save_interval,
                        resume_checkpoint=args.resume_checkpoint,
                        use_fp16=args.use_fp16,
                        fp16_scale_growth=args.fp16_scale_growth,
                        schedule_sampler=schedule_sampler,
                        weight_decay=args.weight_decay,
                        learning_steps=args.learning_steps,
                        checkpoint_path=args.checkpoint_path,
                        gradient_clipping=args.gradient_clipping,
                        eval_data=val_loader,
                        eval_interval=args.eval_interval,
                    ).run_loop()


if __name__ == '__main__':
    raise SystemExit(main())
