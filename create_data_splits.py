"""
Create the data splits for all settings (new reader, new sentence, combined (= new reader/new sentence), and cross dataset
(i.e., train on celer test on zuco) to keep train and test data consistent across all baselines and for hyper-parameter tuning.
Save all data sets as well as only the reader and sn ids (for the baselines).
"""

import argparse
import random
import json
import os
import numpy as np

from scandl.utils import dist_util, logger
from scandl.step_sample import create_named_schedule_sampler
from sp_basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from sp_train_util import TrainLoop
from sp_load_celer_zuco import load_celer, load_celer_speakers, process_celer, celer_zuco_dataset_and_loader
from sp_load_celer_zuco import load_zuco, process_zuco, get_kfold, get_kfold_indices_combined
from sp_load_celer_zuco import flatten_data, unflatten_data
from sp_load_celer_zuco import get_kfold_indices_scanpath
from transformers import set_seed, BertTokenizerFast, BertConfig
import wandb
from datasets import load_from_disk

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

    print('loading args')
    args = create_argparser().parse_args()

    set_seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

    cv_settings = [('reader', 'cv'), ('sentence', 'cv'), ('scanpath', 'cv'), ('combined', 'cv')]
    cross_dataset_settings = [('scanpath', 'zuco')]

    # load celer for all settings except cross-dataset

    print('loading word info df')
    word_info_df, eyemovement_df = load_celer()
    reader_list = load_celer_speakers(only_native_speakers=args.celer_only_L1)
    sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

    print('loading data')
    data, splitting_IDs_dict = process_celer(
        sn_list=sn_list,
        reader_list=reader_list,
        word_info_df=word_info_df,
        eyemovement_df=eyemovement_df,
        tokenizer=tokenizer,
        args=args,
        inference='cv',
 #       subset_size=200,
    )

    # flatten the data for subsequent splitting
    flattened_data = flatten_data(data)

    for data_split_criterion, inference in cv_settings:

        data_path = os.path.join('processed_data', data_split_criterion)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        args.data_split_criterion = data_split_criterion
        args.inference = inference

        if args.data_split_criterion != 'combined':
            for fold_idx, (train_idx, test_idx) in enumerate(
                get_kfold(
                    data=flattened_data,
                    splitting_IDs_dict=splitting_IDs_dict,
                    splitting_criterion=args.data_split_criterion,
                    n_splits=args.n_folds,
                )
            ):
                fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)

                train_data = np.array(flattened_data)[train_idx].tolist()
                test_data = np.array(flattened_data)[test_idx].tolist()

                # save the train and test IDs separately (though they are also contained within train_data/test_data)
                train_ids_reader = np.array(splitting_IDs_dict['reader'])[train_idx].tolist()
                train_ids_sn = np.array(splitting_IDs_dict['sentence'])[train_idx].tolist()
                test_ids_reader = np.array(splitting_IDs_dict['reader'])[test_idx].tolist()
                test_ids_sn = np.array(splitting_IDs_dict['sentence'])[test_idx].tolist()
                train_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(train_ids_sn, train_ids_reader)]
                test_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(test_ids_sn, test_ids_reader)]
                with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                    np.save(f, test_ids, allow_pickle=True)
                with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                    np.save(f, train_ids, allow_pickle=True)

                # save the train data
                train_data_save = unflatten_data(flattened_data=train_data, split='train')
                train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                # save the test data
                test_data_save = unflatten_data(flattened_data=test_data, split='test')
                test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))

        else:  # new reader/new sentence setting
            reader_indices, sentence_indices = get_kfold_indices_combined(
                data=flattened_data,
                splitting_IDs_dict=splitting_IDs_dict,
            )

            reader_IDs = splitting_IDs_dict['reader']
            sn_IDs = splitting_IDs_dict['sentence']

            for fold_idx, ((reader_train_idx, reader_test_idx), (sn_train_idx, sn_test_idx)) in enumerate(
                zip(reader_indices, sentence_indices)
            ):
                fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)

                # create data sets with only unique readers and sentences in test set
                unique_reader_test_IDs = set(np.array(reader_IDs)[reader_test_idx].tolist())
                unique_sn_test_IDs = set(np.array(sn_IDs)[sn_test_idx].tolist())

                # subset the data: if an ID is both in the IDs for sentence and reader sampled for the test set,
                # add the data point to the test data; if it is in neither of them, add to train data. if
                # in one of them, unfortunately discard
                train_data, test_data = list(), list()
                train_ids, test_ids = list(), list()
                for i in range(len(flattened_data)):
                    if reader_IDs[i] in unique_reader_test_IDs and sn_IDs[i] in unique_sn_test_IDs:
                        test_data.append(flattened_data[i])
                        test_ids.append([sn_IDs[i], reader_IDs[i]])
                    elif reader_IDs[i] not in unique_reader_test_IDs and sn_IDs[i] not in unique_sn_test_IDs:
                        train_data.append(flattened_data[i])
                        train_ids.append([sn_IDs[i], reader_IDs[i]])
                    else:
                        continue

                # save the train data
                train_data_save = unflatten_data(flattened_data=train_data, split='train')
                train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                # save the test data
                test_data_save = unflatten_data(flattened_data=test_data, split='test')
                test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))

                # save the train and test ids
                with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                    np.save(f, test_ids, allow_pickle=True)
                with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                    np.save(f, train_ids, allow_pickle=True)

    del data
    del splitting_IDs_dict
    del word_info_df
    del eyemovement_df
    del reader_list
    del sn_list

    # load and save the data for the cross-dataset evaluation (i.e., train celer, test zuco)
    for data_split_criterion, inference in cross_dataset_settings:
        args.data_split_criterion = data_split_criterion
        args.inference = inference

        data_path = os.path.join('processed_data', 'cross_dataset')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        split_sizes = {'val_size': 0.1}

        word_info_df, eyemovement_df = load_celer()
        reader_list = load_celer_speakers(only_native_speakers=args.celer_only_L1)
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

        train_data.save_to_disk(os.path.join(data_path, 'train_data'))
        val_data.save_to_disk(os.path.join(data_path, 'val_data'))

        # save the train IDs (i.e., including the validation IDs --> used for the baselines)
        train_sn_ids = train_data['train']['sn_ids'] + val_data['val']['sn_ids']
        train_reader_ids = train_data['train']['reader_ids'] + val_data['val']['reader_ids']
        train_ids = [[sn_id, reader_id] for sn_id, reader_id in zip(train_sn_ids, train_reader_ids)]
        with open(os.path.join(data_path, 'train_ids.npy'), 'wb') as f:
            np.save(f, train_ids, allow_pickle=True)

        # loading ZuCo: onla ZuCo (1), not ZuCo2.0
        # only tasks 1 (Sentiment) and task 2 (Wikipedia), which are normal reading
        word_info_df, eyemovement_df = load_zuco(task='zuco11')  # task: 'zuco11', 'zuco12'
        word_info_df2, eyemovement_df2 = load_zuco(task='zuco12')
        # combine the two corpora
        word_info_df2.SN = word_info_df2.SN.values + word_info_df.SN.values.max()
        eyemovement_df2.sn = eyemovement_df2.sn.values + eyemovement_df.sn.values.max()
        word_info_df = word_info_df.append(word_info_df2)
        eyemovement_df = eyemovement_df.append(eyemovement_df2)

        # lists with unique sentence and reader IDs
        sn_list = np.unique(eyemovement_df.sn.values).tolist()
        reader_list = np.unique(eyemovement_df.id.values).tolist()

        # call the split 'train' so that the data is not split at all; use all zuco data for inference
        test_data = process_zuco(
            sn_list=sn_list,
            reader_list=reader_list,
            word_info_df=word_info_df,
            eyemovement_df=eyemovement_df,
            tokenizer=tokenizer,
            args=args,
            split='train',
            splitting_criterion=args.data_split_criterion,
        )
        test_data.save_to_disk(os.path.join(data_path, 'test_data'))

        # save test IDs
        test_ids = [[sn_id, reader_id] for sn_id, reader_id in zip(
            test_data['train']['sn_ids'], test_data['train']['reader_ids']
        )]
        with open(os.path.join(data_path, 'test_ids.npy'), 'wb') as f:
            np.save(f, test_ids, allow_pickle=True)









if __name__ == '__main__':
    main()
