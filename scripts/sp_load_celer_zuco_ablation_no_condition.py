import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as Dataset2
import datasets
import CONSTANTS as C
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold, StratifiedKFold
from typing import Optional, List, Tuple, Union, Any, Dict
from itertools import product


def load_celer():
    path_to_fix = C.PATH_TO_FIX
    path_to_ia = C.PATH_TO_IA
    eyemovement_df = pd.read_csv(path_to_fix, delimiter='\t')
    eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace('\t(.*)',
                                                                                                               '',
                                                                                                               regex=True)
    word_info_df = pd.read_csv(path_to_ia, delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
    return word_info_df, eyemovement_df


def load_celer_speakers(only_native_speakers: bool = True):
    sub_metadata_path = C.SUB_METADATA_PATH
    sub_info = pd.read_csv(sub_metadata_path, delimiter='\t')
    if only_native_speakers:
        readers_list = sub_info[sub_info.L1 == 'English'].List.values
    else:
        readers_list = sub_info.List.values
    return readers_list.tolist()


def compute_word_length(arr):
    # length of a punctuation is 0, plus an epsilon to avoid division output inf
    arr = arr.astype('float64')
    arr[arr == 0] = 1 / (0 + 0.5)
    arr[arr != 0] = 1 / (arr[arr != 0])
    return arr


def compute_word_frequency(arr):
    arr[arr == np.inf] = np.nan
    arr[arr != np.inf] = np.log10(arr[arr != np.inf])
    return arr


def process_celer_orig(
        sn_list,
        reader_list,
        word_info_df,
        eyemovement_df,
        tokenizer,
        args,
        src_and_trg: bool = True,
        split: str = 'train',
        subset_size: Optional[int] = None,
):
    """
    SN embedding   <CLS>, bla, bla, <SEP>
    SP_token       <CLS>, bla, bla, <SEP>
    SP_ordinal_pos 0, bla, bla, max_sp_len
    SP_fix_dur     0, bla, bla, 0

    :param sn_list: contains the unique text IDs for the current dataset (either the train, val or test IDs for train,
    val, test dataset)
    :param reader_list: contains the subject IDs for the current dataset (either the train, val or test IDs for train,
     val, test dataset)); either only native speakers or all speakers
    :param word_info_df: contains the celer interest area report in the form of a data frame
    :param eyemovement_df: contains the celer fixations report in the form of a data frame
    :param tokenizer: the BERT Tokenizer
    :param cf: the config dictionary
    """
    SP_ordinal_pos = []
    SP_landing_pos = []
    SP_fix_dur = []

    data = {
        'input_ids_sn': list(),
        'input_ids_sp': list(),
        'input_ids': list(),
        'input_mask': list(),
    }

    for sn_id_idx, sn_id in tqdm(enumerate(sn_list)):  # for text/sentence ID

        if subset_size is not None:
            if sn_id_idx == subset_size:
                break

        # subset the fixations report DF to a DF containing only the current sentence/text ID (each sentence appears multiple times)
        sn_df = eyemovement_df[eyemovement_df.sentenceid == sn_id]
        # notice: Each sentence is recorded multiple times in file |word_info_df|.
        # subset the interest area report DF to a DF containing only the current sentence/text ID
        sn = word_info_df[word_info_df.sentenceid == sn_id]
        # sn is a dataframe containing only one sentence (the sentence with the current sentence ID)
        sn = sn[
            sn['list'] == sn.list.values.tolist()[0]]  # list = experimental list number (unique to each participant).
        sn_str = sn.sentence.iloc[-1]  # the whole sentence as string
        if sn_id == '1987/w7_019/w7_019.295-3' or sn_id == '1987/w7_036/w7_036.147-43' or sn_id == '1987/w7_091/w7_091.360-6':
            # extra inverted commas at the end of the sentence
            sn_str = sn_str[:-3] + sn_str[-1:]
        if sn_id == '1987/w7_085/w7_085.200-18':
            sn_str = sn_str[:43] + sn_str[44:]
        # sn_len = len(tokenizer.tokenize(sn_str))
        sn_len = len(sn_str.split())

        sn_split = sn_str.split()

        # tokenization and padding
        tokenizer.padding_side = 'right'
        # # TODO maybe change this --> don't add CLS and SEP and have add_special_tokens=True in .encode_plus
        sn_str = '[CLS]' + ' ' + sn_str + ' ' + '[SEP]'
        # # pre-tokenized input
        # tokens = tokenizer.encode_plus(
        #     sn_str.split(),
        #     add_special_tokens=False,  # bc we manually added CLS and SEP to the string
        #     truncation=False,
        #     max_length=args.seq_len,
        #     padding='max_length',
        #     return_attention_mask=True,
        #     is_split_into_words=True,
        # )
        # encoded_sn = tokens['input_ids']
        # mask_sn = tokens['attention_mask']
        # # use offset mapping to determine if two tokens are in the same word.
        # # index start from 0, CLS -> 0 and SEP -> last index
        # word_ids_sn = tokens.word_ids()
        # word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]
        for sub_id_idx, sub_id in enumerate(reader_list):

            if sub_id_idx == 5:
                continue

            sub_df = sn_df[sn_df.list == sub_id]
            # remove fixations on non-words
            sub_df = sub_df.loc[
                sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']  # Label for the interest area to which the currentixation is assigned
            if len(sub_df) == 0:
                # no scanpath data found for the subject
                continue

            # prepare decoder input and output
            sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

            # check if recorded fixation duration are within reasonable limits
            # Less than 15ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
            outlier_indx = np.where(sp_fix_dur < 50)[
                0]  # gives indices of the fixations in the fixations list that were shorter than 50ms

            if outlier_indx.size > 0:
                for out_idx in range(len(outlier_indx)):
                    outlier_i = outlier_indx[out_idx]
                    merge_flag = False

                    # outliers are commonly found in the fixation of the last record and the first record, and are removed directly
                    if outlier_i == len(sp_fix_dur) - 1 or outlier_i == 0:
                        merge_flag = True

                    else:
                        if outlier_i - 1 >= 0 and not merge_flag:
                            # try to merge with the left fixation if they landed both on the same interest area
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        if outlier_i + 1 < len(sp_fix_dur) and not merge_flag:
                            # try to merge with the right fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                    # delete the position (interest area ID), the fixation location and the fixation duration from the respective arrays
                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx - 1

            # sanity check
            # scanpath too long, remove outliers, speed up the inference; more than 50 fixations on sentence
            if len(sp_word_pos) > 50:  # 72/10684
                continue
            # scanpath too short for a normal length sentence
            if len(sp_word_pos) <= 1 and sn_len > 10:
                continue

            sp_ordinal_pos = sp_word_pos.astype(int)  # interest area index, i.e., word IDs in fixation report
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_fix_dur.append(sp_fix_dur)
            # preprocess landing position feature
            # assign missing value to 'nan'
            sp_fix_loc = np.where(sp_fix_loc == '.', np.nan, sp_fix_loc)
            # convert string of number of float type
            sp_fix_loc = [float(i) for i in sp_fix_loc]
            # Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
            if np.nanmax(
                    sp_fix_loc) > 35:  # returns fixation outliers (coordinates very off); np.nanmax returns the max value while igonoring nans
                missing_idx = np.where(np.array(sp_fix_loc) > 5)[0]  # array with indices where fix loc greater than 5
                for miss in missing_idx:
                    if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
                        sp_fix_loc[miss] = np.nan
                    else:
                        print('Landing position calculation error. Unknown cause, needs to be checked')
            SP_landing_pos.append(sp_fix_loc)

            # tokenizing the words in the order they are fixated; tokenizing the scanpath
            # sn_str includes CLS and SEP token; but the sp_ordinal_pos starts from 1, not from 0, that's why
            # iterating over sn_str.split() works. later, we will disregard sn_str again
            sp_split = [sn_str.split()[int(i)] for i in sp_ordinal_pos]  # list of words in the order they are fixated
            #  sp_split = [sn_split[int(i)] for i in sp_ordinal_pos]
            # sp_token_str = '[CLS]' + ' ' + ' '.join(sp_split) + ' ' + '[SEP]'
            # sp_len = len(tokenizer.tokenize(sp_token_str))  # length of scanpath
            # tokenization and padding
            # tokenizer.padding_side = 'right'

            # if during training we have as input both src (the sentence) and trg (the scanpath), just as they
            # do in their ScanDL paper (i.e., we model sn > sp):

            sp_encoded = tokenizer.encode_plus(
                sp_split,
                add_special_tokens=True,
                padding=False,
                return_attention_mask=False,
                is_split_into_words=True,
                truncation=False,
            )
            sp_ids = sp_encoded['input_ids']
            data['input_ids_sp'].append(sp_ids)

            if src_and_trg:

                sn_encoded = tokenizer.encode_plus(
                    sn_split,
                    add_special_tokens=False,  # False because we added CLS and SEP 'manually' to the string above
                    padding=False,
                    return_attention_mask=False,
                    is_split_into_words=True,
                    truncation=False,
                )
                sn_ids = sn_encoded['input_ids']
                data['input_ids_sn'].append(sn_ids)

                end_token = sn_ids[-1]  # the SEP token
                src = sn_ids[:-1]  # excluding separator token
                trg = sp_ids[:-1]

                # TODO seq_len is already at the max (512); maybe this kind of truncation is not the best?
                while len(src) + len(trg) > args.seq_len - 3:
                    if len(src) > len(trg):
                        src.pop()
                    elif len(trg) > len(src):
                        trg.pop()
                    else:
                        src.pop()
                        trg.pop()
                # append SEP token again
                src.append(end_token)
                trg.append(end_token)

                # combine src (sn) and trg (sp) sentences, separated by SEP token
                combined = src + [tokenizer.sep_token_id] + trg
                mask = [0] * (len(src) + 1)

                data['input_ids'].append(combined)
                data['input_mask'].append(mask)

            # else we only have the scanpath as input, i.e., we model sp > sp:
            else:
                raise NotImplementedError

    # pad the data

    if src_and_trg:

        data['input_ids'] = _collate_batch_helper(
            examples=data['input_ids'],
            pad_token_id=tokenizer.pad_token_id,
            max_length=args.seq_len,
        )
        data['input_mask'] = _collate_batch_helper(
            examples=data['input_mask'],
            pad_token_id=1,
            max_length=args.seq_len,
        )

    else:
        raise NotImplementedError('Load celer only implemented for src=sn and trg=sp; sp>sp not yet implemented.')

    # TODO the following section about data splitting is ugly/stupid (also the different dataset dicts); just adopted
    # what they did in the code but we might want to change it to make it more efficient

    if split == 'train':

        dataset = Dataset2.from_dict(data)
        train_dataset = datasets.DatasetDict()
        train_dataset['train'] = dataset
        return train_dataset

    else:

        # flatten the data
        flattened_data = list()
        for i in range(len(data['input_ids'])):
            flattened_data.append(
                (
                    data['input_ids_sn'][i],
                    data['input_ids_sp'][i],
                    data['input_ids'][i],
                    data['input_mask'][i]
                )
            )

        if split == 'train-test':

            train_data, test_data = train_test_split(flattened_data, test_size=0.25, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'input_ids_sn': [instance[0] for instance in train_data],
                    'input_ids_sp': [instance[1] for instance in train_data],
                    'input_ids': [instance[2] for instance in train_data],
                    'input_mask': [instance[3] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'input_ids_sn': [instance[0] for instance in test_data],
                    'input_ids_sp': [instance[1] for instance in test_data],
                    'input_ids': [instance[2] for instance in test_data],
                    'input_mask': [instance[3] for instance in test_data],
                }
            )

            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            return train_data_dict, test_data_dict

        elif split == 'train-val':

            train_data, val_data = train_test_split(flattened_data, test_size=0.1, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'input_ids_sn': [instance[0] for instance in train_data],
                    'input_ids_sp': [instance[1] for instance in train_data],
                    'input_ids': [instance[2] for instance in train_data],
                    'input_mask': [instance[3] for instance in train_data],
                }
            )
            val_dataset = Dataset2.from_dict(
                {
                    'input_ids_sn': [instance[0] for instance in val_data],
                    'input_ids_sp': [instance[1] for instance in val_data],
                    'input_ids': [instance[2] for instance in val_data],
                    'input_mask': [instance[3] for instance in val_data],
                }
            )

            train_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, val_data_dict

        elif split == 'train-val-test':

            train_data, test_data = train_test_split(flattened_data, test_size=0.25, shuffle=True, random_state=77)
            train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'input_ids_sn': [instance[0] for instance in train_data],
                    'input_ids_sp': [instance[1] for instance in train_data],
                    'input_ids': [instance[2] for instance in train_data],
                    'input_mask': [instance[3] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'input_ids_sn': [instance[0] for instance in test_data],
                    'input_ids_sp': [instance[1] for instance in test_data],
                    'input_ids': [instance[2] for instance in test_data],
                    'input_mask': [instance[3] for instance in test_data],
                }
            )
            val_dataset = Dataset2.from_dict(
                {
                    'input_ids_sn': [instance[0] for instance in val_data],
                    'input_ids_sp': [instance[1] for instance in val_data],
                    'input_ids': [instance[2] for instance in val_data],
                    'input_mask': [instance[3] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, test_data_dict, val_data_dict


def _collate_instance_helper(
        instance,
        pad_token_id,
        padding_steps,  # how many steps/dims to pad,
):
    padding_list = [pad_token_id] * padding_steps
    result = instance + padding_list
    return result


def _collate_batch_helper(
        examples,  # List of Lists of input IDs
        pad_token_id,
        max_length,
        return_mask=False,
):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def _dummy_pad_words(
    examples,
    pad_token_id,
    max_length,
):
    padded_examples = list()
    for instance in examples:
        padded_examples.append(instance + (max_length - len(instance)) * [pad_token_id])
    return padded_examples


class CelerDatasetOrig(Dataset):

    def __init__(
            self,
            text_datasets,
            data_args,
            split,
            model_emb=None,  # the loaded embeddings: either pre-trained or nn.Embedding
    ):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets[split])
        self.data_args = data_args
        self.model_emb = model_emb
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            input_ids = self.text_datasets[self.split][idx]['input_ids']
            # map the torch tokenizer input IDs to their embeddings
            hidden_state = self.model_emb(torch.tensor(input_ids))
            # obtain the input vectors, only used when word embedding is fixed (i.e., not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)
            out_kwargs = {
                'input_ids': np.array(self.text_datasets[self.split][idx]['input_ids']),
                'input_mask': np.array(self.text_datasets[self.split][idx]['input_mask']),
            }
            return arr, out_kwargs


def celer_dataset_and_loader_orig(
        data,
        data_args,
        model_emb,
        split: str,
        deterministic=False,
        loop=True,
):
    dataset = CelerDatasetOrig(
        text_datasets=data,
        data_args=data_args,
        split=split,
        model_emb=model_emb,
    )
    # the __getitem__ method returns a tuple: an array with the embedded words, and a dict with the keys
    # 'input_ids' and 'input_mask'
    data_loader = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        shuffle=not deterministic,  # ??
        num_workers=0,
    )
    if loop:
        return infinite_loader(data_loader)
    else:
        return iter(data_loader)


def infinite_loader(data_loader):
    while True:
        yield from data_loader


def process_celer_leak(
        sn_list,
        reader_list,
        word_info_df,
        eyemovement_df,
        tokenizer,
        args,
        split: Optional[str] = 'train',
        subset_size: Optional[int] = None,
        split_sizes: Optional[Dict[str, float]] = None,
        splitting_criterion: Optional[str] = 'scanpath',  # 'reader', 'sentence', 'combined'
        inference: Optional[str] = None,
):
    """
    Process the Celer corpus so that it can be used as input to the Diffusion model, where the original sentence (sn)
    is the condition and the scan path (sp) is the target that will be noised.
    :param sn_list:
    :param reader_list:
    :param word_info_df:
    :param eyemovement_df:
    :param tokenizer:
    :param args:
    :param split: 'train', 'train-test', 'train-test-val', 'train-val'
    :param subset_size:
    :param split_sizes:
    :param splitting_criterion: how the data should be split for testing and validation. if 'scanpath', the split is
    just done at random for novel scanpaths; 'reader' will split according to readers, 'sentence' according to
    sentences, and 'combined' according to the combination of reader and sentence
    """

    SP_ordinal_pos = []
    SP_landing_pos = []
    SP_fix_dur = []

    data = {
        'mask': list(),
        'sn_repr': list(),   # the input IDs
        'sp_repr': list(),   # the fixation position IDs
        'sn_word_ids': list(),  # the word IDs, including CLS (0) and SEP (len(sn)+1)
        'sp_word_ids': list(),  # again the fixation position IDs, including CLS(0) and SEP (len(sn)+1) for a
        # word ID embedding that aligns sn with sp
        'combined_word_ids': list(),
        'sn_repr_len': list(),  # holds the length of the individual sentences, in tokens including CLS and SEP,
        # so they can be cut off and concatenated with sp repr before being fed into the model
        'combined_sn_sp_indices': list(),
        'words_for_mapping': list(),  # will hold the words that the IDs map to at inference time
        # 'reader_ID': list(),  # holds the reader ID for a stratified split
        # 'sn_ID': list(),  # holds the sentence ID for a stratified split
        # 'combined_reader_sn_ID': list(),  # holds the combination of sentence and reader ID for a split of both new
        # readers and new sentences
    }
    reader_IDs, sn_IDs, combined_reader_sn_IDs = list(), list(), list()

    for sn_id_idx, sn_id in tqdm(enumerate(sn_list)):  # for text/sentence ID

        if subset_size is not None:
            if sn_id_idx == subset_size + 1:
                break

        # subset the fixations report DF to a DF containing only the current sentence/text ID (each sentence appears multiple times)
        sn_df = eyemovement_df[eyemovement_df.sentenceid == sn_id]
        # notice: Each sentence is recorded multiple times in file |word_info_df|.
        # subset the interest area report DF to a DF containing only the current sentence/text ID
        sn = word_info_df[word_info_df.sentenceid == sn_id]
        # sn is a dataframe containing only one sentence (the sentence with the current sentence ID)
        sn = sn[
            sn['list'] == sn.list.values.tolist()[0]]  # list = experimental list number (unique to each participant).
        # compute word length and frequency features for each word
        # BLLIP vocabulary size: 229,538 words.
        # so FREQ_BLLIP are already logs of word frequencies
        # why do we log_10 them again in compute_word_frequency()
        sn_str = sn.sentence.iloc[-1]  # the whole sentence as string
        if sn_id == '1987/w7_019/w7_019.295-3' or sn_id == '1987/w7_036/w7_036.147-43' or sn_id == '1987/w7_091/w7_091.360-6':
            # extra inverted commas at the end of the sentence
            sn_str = sn_str[:-3] + sn_str[-1:]
        if sn_id == '1987/w7_085/w7_085.200-18':
            sn_str = sn_str[:43] + sn_str[44:]
        # sn_len = len(tokenizer.tokenize(sn_str))

        # skip nan values bc they are of type float (np.isnan raises an error)
        if isinstance(sn_str, float):
            continue

        sn_len = len(sn_str.split())

        # add CLS and SEP 'manually' to the sentence so that they receive the word IDs 0 and len(sn)+1
        sn_str = '[CLS] ' + sn_str + ' [SEP]'

        tokenizer.padding_side = 'right'

        for sub_id_idx, sub_id in enumerate(reader_list):

            if sub_id_idx == 5:
                continue

            sub_df = sn_df[sn_df.list == sub_id]
            # remove fixations on non-words
            sub_df = sub_df.loc[
                sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']  # Label for the interest area to which the currentixation is assigned
            if len(sub_df) == 0:
                # no scanpath data found for the subject
                continue

            # prepare decoder input and output
            sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

            # check if recorded fixation duration are within reasonable limits
            # Less than 15ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
            outlier_indx = np.where(sp_fix_dur < 50)[
                0]  # gives indices of the fixations in the fixations list that were shorter than 50ms

            if outlier_indx.size > 0:
                for out_idx in range(len(outlier_indx)):
                    outlier_i = outlier_indx[out_idx]
                    merge_flag = False

                    # outliers are commonly found in the fixation of the last record and the first record, and are removed directly
                    if outlier_i == len(sp_fix_dur) - 1 or outlier_i == 0:
                        merge_flag = True

                    else:
                        if outlier_i - 1 >= 0 and not merge_flag:
                            # try to merge with the left fixation if they landed both on the same interest area
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        if outlier_i + 1 < len(sp_fix_dur) and not merge_flag:
                            # try to merge with the right fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                    # delete the position (interest area ID), the fixation location and the fixation duration from the respective arrays
                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx - 1

            # sanity check
            # scanpath too long, remove outliers, speed up the inference; more than 50 fixations on sentence
            if len(sp_word_pos) > 50:  # 72/10684
                continue
            # scanpath too short for a normal length sentence
            if len(sp_word_pos) <= 1 and sn_len > 10:
                continue

            sp_ordinal_pos = sp_word_pos.astype(int)  # interest area index, i.e., word IDs in fixation report
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_fix_dur.append(sp_fix_dur)
            # preprocess landing position feature
            # assign missing value to 'nan'
            sp_fix_loc = np.where(sp_fix_loc == '.', np.nan, sp_fix_loc)
            # convert string of number of float type
            sp_fix_loc = [float(i) for i in sp_fix_loc]
            # Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
            if np.nanmax(
                    sp_fix_loc) > 35:  # returns fixation outliers (coordinates very off); np.nanmax returns the max value while igonoring nans
                missing_idx = np.where(np.array(sp_fix_loc) > 5)[0]  # array with indices where fix loc greater than 5
                for miss in missing_idx:
                    if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
                        sp_fix_loc[miss] = np.nan
                    else:
                        print('Landing position calculation error. Unknown cause, needs to be checked')
            SP_landing_pos.append(sp_fix_loc)

            encoded_sn = tokenizer.encode_plus(
                sn_str.split(),
                add_special_tokens=False,
                padding=False,
                return_attention_mask=False,
                is_split_into_words=True,
                truncation=False,
            )

            sn_repr = encoded_sn['input_ids']
            sn_word_ids = encoded_sn.word_ids()

            # convert the fixation position array to a list and add the word IDs for the CLS (0) and the SEP (len(sn)+1)
            sp_repr = [0] + sp_ordinal_pos.tolist() + [max(encoded_sn.word_ids())]
            # for the sp, the input to the token embedding is the same as the input to the word ID positional embedding
            sp_word_ids = sp_repr.copy()

            # map the word IDs/interest are indices to the words so that at inference we can map the model output
            # to the fixated words
#            id_to_word_mapping = {k: v for k, v in zip(list(range(len(sn_str.split()))), sn_str.split())}

            # truncating
            # TODO seq_len is already at the max (512); maybe this kind of truncation is not the best?
            sep_token_input_id = sn_repr[-1]
            sep_token_word_id = sp_repr[-1]

            sn_repr = sn_repr[:-1]
            sn_word_ids = sn_word_ids[:-1]
            sp_repr = sp_repr[:-1]
            sp_word_ids = sp_word_ids[:-1]

            while len(sn_repr) + len(sp_repr) > args.seq_len - 3:
                if len(sn_repr) > len(sp_repr):
                    sn_repr.pop()
                    sn_word_ids.pop()
                elif len(sp_repr) > len(sn_repr):
                    sp_repr.pop()
                    sp_word_ids.pop()
                else:
                    sn_repr.pop()
                    sn_word_ids.pop()
                    sp_repr.pop()
                    sp_word_ids.pop()

            # add the SEP token word ID and input ID again
            sn_repr.append(sep_token_input_id)
            sn_word_ids.append(sep_token_word_id)
            sp_repr.append(sep_token_word_id)
            sp_word_ids.append(sep_token_word_id)

            # if args.mask_padding:
            #     mask = [0] * len(sn_repr) + [1] * len(sp_repr)
            #     mask_pad_token_id = 0
            # else:
            #     mask = [0] * len(sn_repr)
            #     mask_pad_token_id = 1

            mask = [0] * len(sn_repr)

            combined_word_ids = sn_word_ids + sp_word_ids
            combined_sn_sp_indices = list(range(0, len(sn_repr))) + list(range(0, len(sp_repr)))

            data['sn_repr'].append(sn_repr)
            data['sn_word_ids'].append(sn_word_ids)
            data['sp_repr'].append(sp_repr)
            data['sp_word_ids'].append(sp_word_ids)
            data['mask'].append(mask)
            data['combined_word_ids'].append(combined_word_ids)
            data['sn_repr_len'].append(len(sn_repr))
            data['combined_sn_sp_indices'].append(combined_sn_sp_indices)

            # CLS and SEP are already included in sn_str
            # pad the words until their length is args.seq_len, then join to string because if tokenized,
            # the collate fn in the DataLoader will change the list shapes
            words_for_mapping = sn_str.split() + (args.seq_len - len(sn_str.split())) * ['[PAD]']
            data['words_for_mapping'].append(' '.join(words_for_mapping))

            # add the reader and sn IDs
            #  data['reader_ID'].append(sub_id)
            #  data['sn_ID'].append(sn_id)
            #  data['combined_reader_sn_ID'].append((sub_id, sn_id))
            reader_IDs.append(sub_id)
            sn_IDs.append(sn_id)
            combined_reader_sn_IDs.append((sub_id, sn_id))

    splitting_IDs_dict = {
        'reader': reader_IDs,
        'sentence': sn_IDs,
        'combined': combined_reader_sn_IDs,
    }

    # pad mask and combined ids, and pad the sn repr and sp repr individually because they have to be piped
    # through the embedding layers individually

    # pad the sn_repr, which is the tokenizer input IDs, with the tokenizer pad ID
    data['sn_repr'] = _collate_batch_helper(
        examples=data['sn_repr'],
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.seq_len,
    )
    # pad the sp repr, which is the fixation interest area indices (indices of words in sn) with last possible word idx
    data['sp_repr'] = _collate_batch_helper(
        examples=data['sp_repr'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    # pad the combined input IDs also with the last possible word idx
    data['combined_word_ids'] = _collate_batch_helper(
        examples=data['combined_word_ids'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    # pad the indicces
    data['combined_sn_sp_indices'] = _collate_batch_helper(
        examples=data['combined_sn_sp_indices'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    # pad the mask
    # data['mask'] = _collate_batch_helper(
    #     examples=data['mask'],
    #     pad_token_id=mask_pad_token_id,
    #     max_length=args.seq_len,
    # )
    data['mask'] = _collate_batch_helper(
        examples=data['mask'],
        pad_token_id=1,
        max_length=args.seq_len,
    )

    if inference == 'cv':
        return data, splitting_IDs_dict

    # TODO the following section with the datasets and data splitting is ugly --> maybe change

    if split == 'train':

        dataset = Dataset2.from_dict(data)
        train_dataset = datasets.DatasetDict()
        train_dataset['train'] = dataset
        return train_dataset

    else:

        # flatten the data
        flattened_data = list()
        for i in range(len(data['sn_repr'])):
            flattened_data.append(
                (
                    data['sn_repr'][i],
                    data['sn_word_ids'][i],
                    data['sp_repr'][i],
                    data['sp_word_ids'][i],
                    data['mask'][i],
                    data['combined_word_ids'][i],
                    data['sn_repr_len'][i],
                    data['combined_sn_sp_indices'][i],
                    data['words_for_mapping'][i],
                    # data['reader_ID'][i],
                    # data['sn_ID'][i],
                    # data['combined_reader_sn_ID'][i],
                )
            )

        if split == 'train-test':

            if split_sizes:
                test_size = split_sizes['test_size']
            else:
                test_size = 0.25

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':

                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )

                else:

                    splitting_IDs = splitting_IDs_dict[splitting_criterion]
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
                    for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        test_data = np.array(flattened_data)[test_index].tolist()

            else:
                train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in train_data],
                    'sn_word_ids': [instance[1] for instance in train_data],
                    'sp_repr': [instance[2] for instance in train_data],
                    'sp_word_ids': [instance[3] for instance in train_data],
                    'mask': [instance[4] for instance in train_data],
                    'combined_word_ids': [instance[5] for instance in train_data],
                    'sn_repr_len': [instance[6] for instance in train_data],
                    'combined_sn_sp_indices': [instance[7] for instance in train_data],
                    'words_for_mapping': [instance[8] for instance in train_data],
                    # 'reader_ID': [instance[9] for instance in train_data],
                    # 'sn_ID': [instance[10] for instance in train_data],
                    # 'combined_reader_sn_ID': [instance[11] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in test_data],
                    'sn_word_ids': [instance[1] for instance in test_data],
                    'sp_repr': [instance[2] for instance in test_data],
                    'sp_word_ids': [instance[3] for instance in test_data],
                    'mask': [instance[4] for instance in test_data],
                    'combined_word_ids': [instance[5] for instance in test_data],
                    'sn_repr_len': [instance[6] for instance in test_data],
                    'combined_sn_sp_indices': [instance[7] for instance in test_data],
                    'words_for_mapping': [instance[8] for instance in test_data],
                    # 'reader_ID': [instance[9] for instance in test_data],
                    # 'sn_ID': [instance[10] for instance in test_data],
                    # 'combined_reader_sn_ID': [instance[11] for instance in test_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            return train_data_dict, test_data_dict

        elif split == 'train-val':

            if split_sizes:
                val_size = split_sizes['val_size']
            else:
                val_size = 0.1

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':
                    train_data, val_data, train_reader_IDs, val_reader_IDs, train_sn_IDs, val_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=val_size,
                    )
                else:
                    splitting_IDs = splitting_IDs_dict[splitting_criterion]
                    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
                    for train_index, val_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        val_data = np.array(flattened_data)[val_index].tolist()

            else:
                train_data, val_data = train_test_split(flattened_data, test_size=val_size, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in train_data],
                    'sn_word_ids': [instance[1] for instance in train_data],
                    'sp_repr': [instance[2] for instance in train_data],
                    'sp_word_ids': [instance[3] for instance in train_data],
                    'mask': [instance[4] for instance in train_data],
                    'combined_word_ids': [instance[5] for instance in train_data],
                    'sn_repr_len': [instance[6] for instance in train_data],
                    'combined_sn_sp_indices': [instance[7] for instance in train_data],
                    'words_for_mapping': [instance[8] for instance in train_data],
                    # 'reader_ID': [instance[9] for instance in train_data],
                    # 'sn_ID': [instance[10] for instance in train_data],
                    # 'combined_reader_sn_ID': [instance[11] for instance in train_data],
                }
            )
            val_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in val_data],
                    'sn_word_ids': [instance[1] for instance in val_data],
                    'sp_repr': [instance[2] for instance in val_data],
                    'sp_word_ids': [instance[3] for instance in val_data],
                    'mask': [instance[4] for instance in val_data],
                    'combined_word_ids': [instance[5] for instance in val_data],
                    'sn_repr_len': [instance[6] for instance in val_data],
                    'combined_sn_sp_indices': [instance[7] for instance in val_data],
                    'words_for_mapping': [instance[8] for instance in val_data],
                    # 'reader_ID': [instance[9] for instance in val_data],
                    # 'sn_ID': [instance[10] for instance in val_data],
                    # 'combined_reader_sn_ID': [instance[11] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, val_data_dict

        elif split == 'train-val-test':

            if split_sizes:
                val_size = split_sizes['val_size']
                test_size = split_sizes['test_size']
            else:
                val_size = 0.1
                test_size = 0.25

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':
                    # split train and test data so that unseen readers and sentences are in the test data
                    # train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                    #     data=flattened_data, splitting_IDs_dict=splitting_IDs_dict, test_size=test_size
                    # )
                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )
                    # randomly split train data into train and validation
                    # TODO maybe change val data also into unseen readers and sentences?
                    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=77, shuffle=True)

                else:
                    splitting_IDs = splitting_IDs_dict[splitting_criterion]

                    # split into train and test
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
                    for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        test_data = np.array(flattened_data)[test_index].tolist()
                        train_ids = np.array(splitting_IDs)[train_index].tolist()

                    # split into train and val
                    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
                    for train_index, val_index in gss.split(train_data, groups=train_ids):
                        val_data = np.array(train_data)[val_index].tolist()
                        train_data = np.array(train_data)[train_index].tolist()

            else:
                train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True, random_state=77)
                train_data, val_data = train_test_split(train_data, test_size=val_size, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in train_data],
                    'sn_word_ids': [instance[1] for instance in train_data],
                    'sp_repr': [instance[2] for instance in train_data],
                    'sp_word_ids': [instance[3] for instance in train_data],
                    'mask': [instance[4] for instance in train_data],
                    'combined_word_ids': [instance[5] for instance in train_data],
                    'sn_repr_len': [instance[6] for instance in train_data],
                    'combined_sn_sp_indices': [instance[7] for instance in train_data],
                    'words_for_mapping': [instance[8] for instance in train_data],
                    # 'reader_ID': [instance[9] for instance in train_data],
                    # 'sn_ID': [instance[10] for instance in train_data],
                    # 'combined_reader_sn_ID': [instance[11] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in test_data],
                    'sn_word_ids': [instance[1] for instance in test_data],
                    'sp_repr': [instance[2] for instance in test_data],
                    'sp_word_ids': [instance[3] for instance in test_data],
                    'mask': [instance[4] for instance in test_data],
                    'combined_word_ids': [instance[5] for instance in test_data],
                    'sn_repr_len': [instance[6] for instance in test_data],
                    'combined_sn_sp_indices': [instance[7] for instance in test_data],
                    'words_for_mapping': [instance[8] for instance in test_data],
                    # 'reader_ID': [instance[9] for instance in test_data],
                    # 'sn_ID': [instance[10] for instance in test_data],
                    # 'combined_reader_sn_ID': [instance[11] for instance in test_data],
                }
            )
            val_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in val_data],
                    'sn_word_ids': [instance[1] for instance in val_data],
                    'sp_repr': [instance[2] for instance in val_data],
                    'sp_word_ids': [instance[3] for instance in val_data],
                    'mask': [instance[4] for instance in val_data],
                    'combined_word_ids': [instance[5] for instance in val_data],
                    'sn_repr_len': [instance[6] for instance in val_data],
                    'combined_sn_sp_indices': [instance[7] for instance in val_data],
                    'words_for_mapping': [instance[8] for instance in val_data],
                    # 'reader_ID': [instance[9] for instance in val_data],
                    # 'sn_ID': [instance[10] for instance in val_data],
                    # 'combined_reader_sn_ID': [instance[11] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, test_data_dict, val_data_dict


class CelerZucoDatasetLeak(Dataset):

    def __init__(
            self,
            dataset,
            data_args,
            split,  # 'train', 'test', 'val'
    ):
        super().__init__()
        self.dataset = dataset
        self.length = len(self.dataset[split])
        self.data_args = data_args
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {
            'sn_repr': np.array(self.dataset[self.split][idx]['sn_repr']),
            #    'sn_word_ids': np.array(self.dataset[self.split][idx]['sn_word_ids']),
            'sp_repr': np.array(self.dataset[self.split][idx]['sp_repr']),
            #    'sp_word_ids': np.array(self.dataset[self.split][idx]['sp_word_ids']),
            'mask': np.array(self.dataset[self.split][idx]['mask']),
            'combined_word_ids': np.array(self.dataset[self.split][idx]['combined_word_ids']),
            'sn_repr_len': np.array(self.dataset[self.split][idx]['sn_repr_len']),
            'combined_sn_sp_indices': np.array(self.dataset[self.split][idx]['combined_sn_sp_indices']),
            'words_for_mapping': self.dataset[self.split][idx]['words_for_mapping'],
        }
        return sample


def celer_zuco_dataset_and_loader(
        data,
        data_args,
        split: str,
        deterministic=False,
        loop=True,
):

    # dataset = CelerZucoDatasetLeak(
    #     dataset=data,
    #     data_args=data_args,
    #     split=split,
    # )
    dataset = CelerZucoDataset(
        dataset=data,
        data_args=data_args,
        split=split,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        shuffle=not deterministic,  # ??
        num_workers=0,
    )
    if loop:
        return infinite_loader(data_loader)
    else:
        return iter(data_loader)


# def combined_split(data, splitting_IDs_dict, test_size=0.2):
#     """ Splits the data so that the test data contains both unseen readers and unseen sentences. """
#
#     reader_IDs = splitting_IDs_dict['reader']
#     sn_IDs = splitting_IDs_dict['sentence']
#     unique_readers = set(reader_IDs)
#     unique_sents= set(sn_IDs)
#
#     train_size = 1 - test_size
#
#     # split readers into train and test
#     n_train_readers = int(train_size * len(unique_readers))
#     train_readers = set(random.sample(unique_readers, n_train_readers))
#     test_readers = unique_readers - train_readers
#
#     # get the corresponding sentences for each set of readers
#     train_sents = set([sn_IDs[i] for i in range(len(data)) if reader_IDs[i] in train_readers])
#     test_sents = unique_sents - train_sents
#
#     # split sents into train and test
#     n_train_sents = int(len(train_sents) / (len(train_sents) + len(test_sents)) * len(unique_sents))
#     train_sents = set(random.sample(train_sents, n_train_sents))
#     test_sents = unique_sents - train_sents
#
#     # get the corresponding readers for each set of sentences
#     train_readers = set([reader_IDs[i] for i in range(len(data)) if sn_IDs[i] in train_sents])
#     test_readers = unique_readers - train_readers
#
#     # create the train and test data
#     train_data = [data[i] for i in range(len(data)) if reader_IDs[i] in train_readers and sn_IDs[i] in train_sents]
#     test_data = [data[i] for i in range(len(data)) if reader_IDs[i] in test_readers and sn_IDs[i] in test_sents]
#
#     # also split the IDs
#     train_reader_IDs = [reader_IDs[i] for i in range(len(data)) if reader_IDs[i] in train_readers and sn_IDs[i] in train_sents]
#     test_reader_IDs = [reader_IDs[i] for i in range(len(data)) if reader_IDs[i] in test_readers and sn_IDs[i] in test_sents]
#     train_sn_IDs = [sn_IDs[i] for i in range(len(data)) if reader_IDs[i] in train_readers and sn_IDs[i] in train_sents]
#     test_sn_IDs = [sn_IDs[i] for i in range(len(data)) if reader_IDs[i] in test_readers and sn_IDs[i] in test_sents]
#
#     return train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs


# def combined_split(data, reader_IDs, sn_IDs, test_size):
#     """ Splits the data so that the test data contains both unseen readers and unseen sentences. """
#
#     unique_reader_IDs = set(reader_IDs)
#     unique_sn_IDs = set(sn_IDs)
#
#     # split the reader and sentence IDs into train and test splits
#     reader_IDs_train = set(random.sample(unique_reader_IDs, int((1 - test_size) * len(unique_reader_IDs))))
#     sentence_IDs_train = set(random.sample(unique_sn_IDs, int((1 - test_size) * len(unique_reader_IDs))))
#     breakpoint()
#     train_data, test_data = list(), list()
#     train_reader_IDs, test_reader_IDs = list(), list()
#     train_sn_IDs, test_sn_IDs = list(), list()
#     for i in range(len(data)):
#         if unique_reader_IDs[i] in reader_IDs_train and unique_sn_IDs[i] in sentence_IDs_train:
#             train_data.append(data[i])
#             train_reader_IDs.append(reader_IDs[i])
#             train_sn_IDs.append(sn_IDs[i])
#         else:
#             test_data.append(data[i])
#             test_sn_IDs.append(sn_IDs[i])
#             test_reader_IDs.append(data[i])
#     breakpoint()
#     return train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs


def combined_split(data, reader_IDs, sn_IDs, test_size):
    """ Splits the data so that the test data contains both unseen readers and unseen sentences. """
    random.seed(77)
    unique_reader_IDs = set(reader_IDs)
    unique_sn_IDs = set(sn_IDs)

    # sample the sentence and reader IDs that go into the test set
    unique_reader_IDs_test = random.sample(unique_reader_IDs, int(test_size * len(unique_reader_IDs)))
    unique_sn_IDs_test = random.sample(unique_sn_IDs, int(test_size * len(unique_sn_IDs)))

    train_data, test_data = [], []
    train_reader_IDs, test_reader_IDs = [], []
    train_sn_IDs, test_sn_IDs = [], []

    for i in range(len(data)):
        if reader_IDs[i] in unique_reader_IDs_test and sn_IDs[i] in unique_sn_IDs_test:
            test_data.append(data[i])
            test_reader_IDs.append(reader_IDs[i])
            test_sn_IDs.append(sn_IDs[i])
        elif reader_IDs[i] not in unique_reader_IDs_test and sn_IDs[i] not in unique_sn_IDs_test:
            train_data.append(data[i])
            train_reader_IDs.append(reader_IDs[i])
            train_sn_IDs.append(sn_IDs[i])
        else:
            continue

    return train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs


def load_zuco(task: str = None):  # 'zuco11', 'zuco12'
    dir = C.path_to_zuco
    if task.startswith('zuco1'):
        dir = dir + 'zuco/'
    elif task == 'zuco21':
        dir = dir + 'zuco2/'
    dir = os.path.join(dir, f'task{task[-1]}', 'Matlab_files')
    word_info_path = dir + '/Word_Infor.csv'
    word_info_df = pd.read_csv(word_info_path, sep='\t')
    scanpath_path = dir + '/scanpath.csv'
    eyemovement_df = pd.read_csv(scanpath_path, sep='\t')
    return word_info_df, eyemovement_df


def process_zuco_leak(
        sn_list,
        reader_list,
        word_info_df,
        eyemovement_df,
        tokenizer,
        args,
        split: Optional[str] = 'train',
        subset_size: Optional[int] = None,
        split_sizes: Optional[Dict[str, float]] = None,
        splitting_criterion: Optional[str] = 'scanpath',  # 'reader', 'sentence', 'combined'
):
    """
    Process the ZuCo corpus so that it can be used as input to the Diffusion model, where the original sentence (sn)
    is the condition and the scan path (sp) is the target that will be noised.
    :param sn_list:
    :param reader_list:
    :param word_info_df:
    :param eyemovement_df:
    :param tokenizer:
    :param args:
    :param split: 'train', 'train-test', 'train-test-val', 'train-val'
    :param subset_size:
    :param split_sizes:
    :param splitting_criterion: how the data should be split for testing and validation. if 'scanpath', the split is
    just done at random for novel scanpaths; 'reader' will split according to readers, 'sentence' according to
    sentences, and 'combined' according to the combination of reader and sentence
    """
    SP_ordinal_pos = []
    SP_landing_pos = []
    SP_fix_dur = []

    data = {
        'mask': list(),
        'sn_repr': list(),  # the input IDs
        'sp_repr': list(),  # the fixation position IDs
        'sn_word_ids': list(),  # the word IDs, including CLS (0) and SEP (len(sn)+1)
        'sp_word_ids': list(),  # again the fixation position IDs, including CLS(0) and SEP (len(sn)+1) for a
        # word ID embedding that aligns sn with sp
        'combined_word_ids': list(),
        'sn_repr_len': list(),  # holds the length of the individual sentences, in tokens including CLS and SEP,
        # so they can be cut off and concatenated with sp repr before being fed into the model
        'combined_sn_sp_indices': list(),
        'words_for_mapping': list(),  # will hold the words that the IDs map to at inference time
    }

    reader_IDs, sn_IDs = list(), list()

    for sn_id_idx, sn_id in tqdm(enumerate(sn_list)):

        if subset_size is not None:
            if sn_id_idx == subset_size + 1:
                break

        sn_df = eyemovement_df[eyemovement_df.sn == sn_id]
        sn = word_info_df[word_info_df.SN == sn_id]
        sn_str = ' '.join(sn.WORD.values)
        sn_len = len(sn_str.split())

        tokenizer.padding_side = 'right'
        sn_str = '[CLS] ' + sn_str + ' [SEP]'

        for sub_id_idx, sub_id in enumerate(reader_list):

            sub_df = sn_df[sn_df.id == sub_id]
            # remove fixations on non-words
            sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '']
            if len(sub_df) == 0:
                # no scanpath data found for the subject
                continue

            sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.wn.values, sub_df.fl.values, sub_df.dur.values

            # check if recorded fixation duration are within reasonable limits
            # Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
            outlier_indx = np.where(sp_fix_dur < 50)[0]
            if outlier_indx.size > 0:
                for out_idx in range(len(outlier_indx)):
                    outlier_i = outlier_indx[out_idx]
                    merge_flag = False
                    if outlier_i - 1 >= 0 and not merge_flag:
                        # try to merge with the left fixation
                        if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                            sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                    if outlier_i + 1 < len(sp_fix_dur) and not merge_flag:
                        # try to merge with the right fixation
                        if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                            sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx - 1

            # sanity check
            # scanpath too long, remove outliers, speed up the inference
            # if len(sp_word_pos) > 100:
            # continue
            # scanpath too short for a normal length sentence
            if len(sp_word_pos) <= 1 and sn_len > 10:
                continue

            sp_ordinal_pos = sp_word_pos.astype(int)
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_fix_dur.append(sp_fix_dur)

            # preprocess landing position feature
            # assign missing value to 'nan'
            # sp_fix_loc=np.where(sp_fix_loc=='.', np.nan, sp_fix_loc)
            # convert string of number of float type
            sp_fix_loc = [float(i) if isinstance(i, int) or isinstance(i, float) else np.nan for i in sp_fix_loc if
                          isinstance(i, int) or isinstance(i, float)]
            SP_landing_pos.append(sp_fix_loc)

            # encode the sentence
            encoded_sn = tokenizer.encode_plus(
                sn_str.split(),
                add_special_tokens=False,
                padding=False,
                return_attention_mask=False,
                is_split_into_words=True,
                truncation=False,
            )

            sn_repr = encoded_sn['input_ids']
            sn_word_ids = encoded_sn.word_ids()

            # convert the fixatin position array to a list and add the word IDs for CLS (0) and SEP (len(sn)+1)
            sp_repr = [0] + sp_ordinal_pos.tolist() + [max(encoded_sn.word_ids())]
            # for the sp, the input to the token embedding is the same as the input to the word ID positional embedding
            sp_word_ids = sp_repr.copy()

            # truncation
            sep_token_input_id = sn_repr[-1]
            sep_token_word_id = sp_repr[-1]

            sn_repr = sn_repr[:-1]
            sn_word_ids = sn_word_ids[:-1]
            sp_repr = sp_repr[:-1]
            sp_word_ids = sp_word_ids[:-1]

            while len(sn_repr) + len(sp_repr) > args.seq_len - 3:
                if len(sn_repr) > len(sp_repr):
                    sn_repr.pop()
                    sn_word_ids.pop()
                elif len(sp_repr) > len(sn_repr):
                    sp_repr.pop()
                    sp_word_ids.pop()
                else:
                    sn_repr.pop()
                    sn_word_ids.pop()
                    sp_repr.pop()
                    sp_word_ids.pop()

            # add the SEP token word ID and input ID again
            sn_repr.append(sep_token_input_id)
            sn_word_ids.append(sep_token_word_id)
            sp_repr.append(sep_token_word_id)
            sp_word_ids.append(sep_token_word_id)

            mask = [0] * len(sn_repr)

            combined_word_ids = sn_word_ids + sp_word_ids
            combined_sn_sp_indices = list(range(0, len(sn_repr))) + list(range(0, len(sp_repr)))

            data['sn_repr'].append(sn_repr)
            data['sn_word_ids'].append(sn_word_ids)
            data['sp_repr'].append(sp_repr)
            data['sp_word_ids'].append(sp_word_ids)
            data['mask'].append(mask)
            data['combined_word_ids'].append(combined_word_ids)
            data['sn_repr_len'].append(len(sn_repr))
            data['combined_sn_sp_indices'].append(combined_sn_sp_indices)

            # CLS and SEP are already included in sn_str
            # pad the words until their length is args.seq_len, then join to string because if tokenized,
            # the collate fn in the DataLoader will change the list shapes
            words_for_mapping = sn_str.split() + (args.seq_len - len(sn_str.split())) * ['[PAD]']
            data['words_for_mapping'].append(' '.join(words_for_mapping))

            reader_IDs.append(sub_id)
            sn_IDs.append(sn_id)

    splitting_IDs_dict = {
        'reader': reader_IDs,
        'sentence': sn_IDs,
    }

    # padding

    # pad the sn_repr, which is the tokenizer input IDs, with the tokenizer pad ID
    data['sn_repr'] = _collate_batch_helper(
        examples=data['sn_repr'],
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.seq_len,
    )
    # pad the sp repr, which is the fixation interest area indices (indices of words in sn) with last possible word idx
    data['sp_repr'] = _collate_batch_helper(
        examples=data['sp_repr'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    # pad the combined input IDs also with the last possible word idx
    data['combined_word_ids'] = _collate_batch_helper(
        examples=data['combined_word_ids'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    # pad the indicces
    data['combined_sn_sp_indices'] = _collate_batch_helper(
        examples=data['combined_sn_sp_indices'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    # pad the mask
    data['mask'] = _collate_batch_helper(
        examples=data['mask'],
        pad_token_id=1,
        max_length=args.seq_len,
    )

    data['mask'] = _collate_batch_helper(
        examples=data['mask'],
        pad_token_id=1,
        max_length=args.seq_len,
    )

    # TODO the following section with the datasets and data splitting is ugly --> maybe change

    if split == 'train':

        dataset = Dataset2.from_dict(data)
        train_dataset = datasets.DatasetDict()
        train_dataset['train'] = dataset
        return train_dataset

    else:

        # flatten the data
        flattened_data = list()
        for i in range(len(data['sn_repr'])):
            flattened_data.append(
                (
                    data['sn_repr'][i],
                    data['sn_word_ids'][i],
                    data['sp_repr'][i],
                    data['sp_word_ids'][i],
                    data['mask'][i],
                    data['combined_word_ids'][i],
                    data['sn_repr_len'][i],
                    data['combined_sn_sp_indices'][i],
                    data['words_for_mapping'][i],
                    #   data['reader_ID'][i],
                    #   data['sn_ID'][i],
                    #   data['combined_reader_sn_ID'][i],
                )
            )

        if split == 'train-test':

            if split_sizes:
                test_size = split_sizes['test_size']
            else:
                test_size = 0.25

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':

                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )

                else:

                    splitting_IDs = splitting_IDs_dict[splitting_criterion]
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
                    for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        test_data = np.array(flattened_data)[test_index].tolist()

            else:
                train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True,
                                                         random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in train_data],
                    'sn_word_ids': [instance[1] for instance in train_data],
                    'sp_repr': [instance[2] for instance in train_data],
                    'sp_word_ids': [instance[3] for instance in train_data],
                    'mask': [instance[4] for instance in train_data],
                    'combined_word_ids': [instance[5] for instance in train_data],
                    'sn_repr_len': [instance[6] for instance in train_data],
                    'combined_sn_sp_indices': [instance[7] for instance in train_data],
                    'words_for_mapping': [instance[8] for instance in train_data],
                    #   'reader_ID': [instance[9] for instance in train_data],
                    #   'sn_ID': [instance[10] for instance in train_data],
                    #   'combined_reader_sn_ID': [instance[11] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in test_data],
                    'sn_word_ids': [instance[1] for instance in test_data],
                    'sp_repr': [instance[2] for instance in test_data],
                    'sp_word_ids': [instance[3] for instance in test_data],
                    'mask': [instance[4] for instance in test_data],
                    'combined_word_ids': [instance[5] for instance in test_data],
                    'sn_repr_len': [instance[6] for instance in test_data],
                    'combined_sn_sp_indices': [instance[7] for instance in test_data],
                    'words_for_mapping': [instance[8] for instance in test_data],
                    #   'reader_ID': [instance[9] for instance in test_data],
                    #   'sn_ID': [instance[10] for instance in test_data],
                    #   'combined_reader_sn_ID': [instance[11] for instance in test_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            return train_data_dict, test_data_dict

        elif split == 'train-val':

            if split_sizes:
                val_size = split_sizes['val_size']
            else:
                val_size = 0.1

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':
                    train_data, val_data, train_reader_IDs, val_reader_IDs, train_sn_IDs, val_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=val_size,
                    )
                else:
                    splitting_IDs = splitting_IDs_dict[splitting_criterion]
                    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
                    for train_index, val_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        val_data = np.array(flattened_data)[val_index].tolist()

            else:
                train_data, val_data = train_test_split(flattened_data, test_size=val_size, shuffle=True,
                                                        random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in train_data],
                    'sn_word_ids': [instance[1] for instance in train_data],
                    'sp_repr': [instance[2] for instance in train_data],
                    'sp_word_ids': [instance[3] for instance in train_data],
                    'mask': [instance[4] for instance in train_data],
                    'combined_word_ids': [instance[5] for instance in train_data],
                    'sn_repr_len': [instance[6] for instance in train_data],
                    'combined_sn_sp_indices': [instance[7] for instance in train_data],
                    'words_for_mapping': [instance[8] for instance in train_data],
                    #   'reader_ID': [instance[9] for instance in train_data],
                    #   'sn_ID': [instance[10] for instance in train_data],
                    #   'combined_reader_sn_ID': [instance[11] for instance in train_data],
                }
            )
            val_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in val_data],
                    'sn_word_ids': [instance[1] for instance in val_data],
                    'sp_repr': [instance[2] for instance in val_data],
                    'sp_word_ids': [instance[3] for instance in val_data],
                    'mask': [instance[4] for instance in val_data],
                    'combined_word_ids': [instance[5] for instance in val_data],
                    'sn_repr_len': [instance[6] for instance in val_data],
                    'combined_sn_sp_indices': [instance[7] for instance in val_data],
                    'words_for_mapping': [instance[8] for instance in val_data],
                    #   'reader_ID': [instance[9] for instance in val_data],
                    #   'sn_ID': [instance[10] for instance in val_data],
                    #   'combined_reader_sn_ID': [instance[11] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, val_data_dict

        elif split == 'train-val-test':

            if split_sizes:
                val_size = split_sizes['val_size']
                test_size = split_sizes['test_size']
            else:
                val_size = 0.1
                test_size = 0.25

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':
                    # split train and test data so that unseen readers and sentences are in the test data
                    # train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                    #     data=flattened_data, splitting_IDs_dict=splitting_IDs_dict, test_size=test_size
                    # )
                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )
                    # randomly split train data into train and validation
                    # TODO maybe change val data also into unseen readers and sentences?
                    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=77,
                                                            shuffle=True)

                else:
                    splitting_IDs = splitting_IDs_dict[splitting_criterion]

                    # split into train and test
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
                    for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        test_data = np.array(flattened_data)[test_index].tolist()
                        train_ids = np.array(splitting_IDs)[train_index].tolist()

                    # split into train and val
                    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
                    for train_index, val_index in gss.split(train_data, groups=train_ids):
                        val_data = np.array(train_data)[val_index].tolist()
                        train_data = np.array(train_data)[train_index].tolist()

            else:
                train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True,
                                                         random_state=77)
                train_data, val_data = train_test_split(train_data, test_size=val_size, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in train_data],
                    'sn_word_ids': [instance[1] for instance in train_data],
                    'sp_repr': [instance[2] for instance in train_data],
                    'sp_word_ids': [instance[3] for instance in train_data],
                    'mask': [instance[4] for instance in train_data],
                    'combined_word_ids': [instance[5] for instance in train_data],
                    'sn_repr_len': [instance[6] for instance in train_data],
                    'combined_sn_sp_indices': [instance[7] for instance in train_data],
                    'words_for_mapping': [instance[8] for instance in train_data],
                    #   'reader_ID': [instance[9] for instance in train_data],
                    #   'sn_ID': [instance[10] for instance in train_data],
                    #   'combined_reader_sn_ID': [instance[11] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in test_data],
                    'sn_word_ids': [instance[1] for instance in test_data],
                    'sp_repr': [instance[2] for instance in test_data],
                    'sp_word_ids': [instance[3] for instance in test_data],
                    'mask': [instance[4] for instance in test_data],
                    'combined_word_ids': [instance[5] for instance in test_data],
                    'sn_repr_len': [instance[6] for instance in test_data],
                    'combined_sn_sp_indices': [instance[7] for instance in test_data],
                    'words_for_mapping': [instance[8] for instance in test_data],
                    #   'reader_ID': [instance[9] for instance in test_data],
                    #   'sn_ID': [instance[10] for instance in test_data],
                    #   'combined_reader_sn_ID': [instance[11] for instance in test_data],
                }
            )
            val_dataset = Dataset2.from_dict(
                {
                    'sn_repr': [instance[0] for instance in val_data],
                    'sn_word_ids': [instance[1] for instance in val_data],
                    'sp_repr': [instance[2] for instance in val_data],
                    'sp_word_ids': [instance[3] for instance in val_data],
                    'mask': [instance[4] for instance in val_data],
                    'combined_word_ids': [instance[5] for instance in val_data],
                    'sn_repr_len': [instance[6] for instance in val_data],
                    'combined_sn_sp_indices': [instance[7] for instance in val_data],
                    'words_for_mapping': [instance[8] for instance in val_data],
                    #   'reader_ID': [instance[9] for instance in val_data],
                    #   'sn_ID': [instance[10] for instance in val_data],
                    #   'combined_reader_sn_ID': [instance[11] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, test_data_dict, val_data_dict


def get_kfold_indices_scanpath(
        splitting_IDs_dict: Dict[str, Union[int, str]],
        n_splits: int = 5,
):
    """ Function to implement the 'random'/'scanpath' split, in which the test set contains both sentences and
    readers that were seen during training. Unfortunately, it is not possible to control for the ratios of both
    readers and sentences (as they are co-dependent), so I will make sure that the data is shuffled and at least
    the readers are stratified. """

    reader_IDs = splitting_IDs_dict['reader']
    sn_IDs = splitting_IDs_dict['sentence']

    tuple_ids = [(idx, reader_ID, sn_ID) for idx, (reader_ID, sn_ID) in enumerate(zip(reader_IDs, sn_IDs))]

    # list of indices of the sn-reader pairs of unique sentences
    unique_sns_ids = [idx for (idx, reader_ID, sn_ID) in tuple_ids if not sn_ID.startswith('en')]
    # unique_sns_indices = {sn_ID: idx for (idx, reader_ID, sn_ID) in tuple_ids if not sn_ID.startswith('en')}
    universal_sn_ids = [idx for (idx, reader_ID, sn_ID) in tuple_ids if sn_ID.startswith('en')]

    # get the reader IDs for the unique and universal sns
    unique_reader_ids = [reader_ID for (idx, reader_ID, sn_ID) in tuple_ids if not sn_ID.startswith('en')]
    universal_reader_ids = [reader_ID for (idx, reader_ID, sn_ID) in tuple_ids if sn_ID.startswith('en')]

    # stratify both the readers for the universal sentences and the ones for the unique sentences
    kfold_unique = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)
    kfold_universal = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)
    train_idx_unique, test_idx_unique = list(), list()
    train_idx_universal, test_idx_universal = list(), list()
    kfold_unique.get_n_splits(unique_sns_ids, groups=unique_reader_ids)
    kfold_universal.get_n_splits(universal_sn_ids, groups=universal_reader_ids)
    for train_idx, test_idx in kfold_unique.split(X=unique_sns_ids, y=unique_reader_ids, groups=unique_reader_ids):
        train_idx_unique.append(train_idx)
        test_idx_unique.append(test_idx)
    for train_idx, test_idx in kfold_universal.split(X=universal_sn_ids, y=universal_reader_ids, groups=universal_reader_ids):
        train_idx_universal.append(train_idx)
        test_idx_universal.append(test_idx)

    # concatenate the respective train and test idx together
    all_train_idx, all_test_idx = list(), list()

    for train_idx_univ, train_idx_uniq in zip(train_idx_universal, train_idx_unique):
        train_idx = np.concatenate([train_idx_univ, train_idx_uniq])
        all_train_idx.append(train_idx)
    for test_idx_univ, test_idx_uniq in zip(test_idx_universal, test_idx_unique):
        test_idx = np.concatenate([test_idx_univ, test_idx_uniq])
        all_test_idx.append(test_idx)

    all_idx = [(train_idx, test_idx) for train_idx, test_idx in zip(all_train_idx, all_test_idx)]
    return all_idx


def get_kfold(
        data: List[Tuple[Any]],
        splitting_IDs_dict: Dict[str, Union[int, str]],
        splitting_criterion: str = 'scanpath',  # 'scanpath', 'reader', 'sentence', 'combined',
        n_splits: int = 5,
):
    if splitting_criterion == 'scanpath':
        # kfold = KFold(n_splits=n_splits, random_state=77, shuffle=True)
        # kfold.get_n_splits(data)
        # return kfold.split(data)
        return (get_kfold_indices_scanpath(
            splitting_IDs_dict=splitting_IDs_dict,
            n_splits=n_splits,
        ))
    elif splitting_criterion in ['reader', 'sentence']:
        splitting_group = splitting_IDs_dict[splitting_criterion]
        kfold = GroupKFold(n_splits=n_splits)
        kfold.get_n_splits(data, groups=splitting_group)
        return kfold.split(data, groups=splitting_group)
    else:  # combined split
        raise NotImplementedError
        # kfold_reader = GroupKFold(n_splits=n_splits)
        # kfold_sentence = GroupKFold(n_splits=n_splits)
        # kfold_reader.get_n_splits(data, groups=splitting_IDs_dict['reader'])
        # kfold_sentence.get_n_splits(data, groups=splitting_IDs_dict['sentence'])


def get_kfold_indices_combined(
        data: List[Tuple[Any]],
        splitting_IDs_dict: Dict[str, Union[int, str]],
        n_splits: int = 5,
):
    kfold_reader = GroupKFold(n_splits=n_splits)
    kfold_sentence = GroupKFold(n_splits=n_splits)
    kfold_reader.get_n_splits(data, groups=splitting_IDs_dict['reader'])
    kfold_sentence.get_n_splits(data, groups=splitting_IDs_dict['sentence'])
    reader_indices, sentence_indices = list(), list()
    for train_idx, test_idx in kfold_reader.split(data, groups=splitting_IDs_dict['reader']):
        reader_indices.append((train_idx, test_idx))
    for train_idx, test_idx in kfold_sentence.split(data, groups=splitting_IDs_dict['sentence']):
        sentence_indices.append((train_idx, test_idx))
    return reader_indices, sentence_indices


def flatten_data_leak(data: Dict[str, List[Any]]):
    flattened_data = list()
    for i in range(len(data['sn_repr'])):
        flattened_data.append(
            (
                data['sn_repr'][i],
                data['sn_word_ids'][i],
                data['sp_repr'][i],
                data['sp_word_ids'][i],
                data['mask'][i],
                data['combined_word_ids'][i],
                data['sn_repr_len'][i],
                data['combined_sn_sp_indices'][i],
                data['words_for_mapping'][i],
            )
        )
    return flattened_data


def unflatten_data_leak(flattened_data: List[Tuple[Any]], split: str):
    dataset = Dataset2.from_dict(
        {
            'sn_repr': [instance[0] for instance in flattened_data],
            'sn_word_ids': [instance[1] for instance in flattened_data],
            'sp_repr': [instance[2] for instance in flattened_data],
            'sp_word_ids': [instance[3] for instance in flattened_data],
            'mask': [instance[4] for instance in flattened_data],
            'combined_word_ids': [instance[5] for instance in flattened_data],
            'sn_repr_len': [instance[6] for instance in flattened_data],
            'combined_sn_sp_indices': [instance[7] for instance in flattened_data],
            'words_for_mapping': [instance[8] for instance in flattened_data],
        }
    )
    data_dict = datasets.DatasetDict()
    data_dict[split] = dataset
    return data_dict


def flatten_data(data: Dict[str, List[Any]]):
    flattened_data = list()
    for i in range(len(data['sp_word_ids'])):
        flattened_data.append(
            (
                #    data['mask'][i],
                #    data['sn_sp_repr'][i],
                data['sn_word_ids'][i],
                data['sp_word_ids'][i],
                data['sn_input_ids'][i],
                data['indices_pos_enc'][i],
                data['sn_repr_len'][i],
                data['words_for_mapping'][i],
                #    data['mask_sn_padding'][i],
                data['mask_transformer_att'][i],
                data['sn_ids'][i],
                data['reader_ids'][i],
            )
        )
    return flattened_data


def unflatten_data(flattened_data: List[Tuple[Any]], split: str):
    dataset = Dataset2.from_dict(
        {
            #    'mask': [instance[0] for instance in flattened_data],
            #    'sn_sp_repr': [instance[1] for instance in flattened_data],
            'sn_word_ids': [instance[0] for instance in flattened_data],
            'sp_word_ids': [instance[1] for instance in flattened_data],
            'sn_input_ids': [instance[2] for instance in flattened_data],
            'indices_pos_enc': [instance[3] for instance in flattened_data],
            'sn_repr_len': [instance[4] for instance in flattened_data],
            'words_for_mapping': [instance[5] for instance in flattened_data],
            #    'mask_sn_padding': [instance[6] for instance in flattened_data],
            'mask_transformer_att': [instance[6] for instance in flattened_data],
            'sn_ids': [instance[7] for instance in flattened_data],
            'reader_ids': [instance[8] for instance in flattened_data],
        }
    )
    data_dict = datasets.DatasetDict()
    data_dict[split] = dataset
    return data_dict


def process_celer(
        sn_list,
        reader_list,
        word_info_df,
        eyemovement_df,
        tokenizer,
        args,
        split: Optional[str] = 'train',
        subset_size: Optional[int] = None,
        split_sizes: Optional[Dict[str, float]] = None,
        splitting_criterion: Optional[str] = 'scanpath',  # 'reader', 'sentence', 'combined'
        inference: Optional[str] = None,
):
    """
    Process the Celer corpus so that it can be used as input to the Diffusion model, where the original sentence (sn)
    is the condition and the scan path (sp) is the target that will be noised.
    :param sn_list:
    :param reader_list:
    :param word_info_df:
    :param eyemovement_df:
    :param tokenizer:
    :param args:
    :param split: 'train', 'train-test', 'train-test-val', 'train-val'
    :param subset_size:
    :param split_sizes:
    :param splitting_criterion: how the data should be split for testing and validation. if 'scanpath', the split is
    just done at random for novel scanpaths; 'reader' will split according to readers, 'sentence' according to
    sentences, and 'combined' according to the combination of reader and sentence
    :param inference:
    """
    SP_ordinal_pos = []
    SP_landing_pos = []
    SP_fix_dur = []

    data = {
        # 'mask': list(),  # 0 for sn, 1 for sp
        # 'sn_sp_repr': list(),  # word IDs of sn and corresponding word IDs of sp (fixated words, interest area IDs) padded with args.seq_len -1
        'sn_word_ids': list(),
        'sp_word_ids': list(),
        'sn_input_ids': list(),  # input IDs of tokenized sentence, padded with pad token ID
        'indices_pos_enc': list(),  # indices from 1 ... len(sn input ids) 1 ... (seq_len - len(sn input ids))
        'sn_repr_len': list(),  # length of sentence in subword tokens
        'words_for_mapping': list(),   # original words of sentence, padded with PAD
        #    'mask_sn_padding': list(),  # masks both the sentence and the padding, for the loss computations
        'mask_transformer_att': list(),  # masks only the padding, for the transformer attention
        'sn_ids': list(),
        'reader_ids': list(),
    }

    max_len = 0

    reader_IDs, sn_IDs = list(), list()

    for sn_id_idx, sn_id in tqdm(enumerate(sn_list)):  # for text/sentence ID

        if subset_size is not None:
            if sn_id_idx == subset_size + 1:
                break

        # subset the fixations report DF to a DF containing only the current sentence/text ID (each sentence appears multiple times)
        sn_df = eyemovement_df[eyemovement_df.sentenceid == sn_id]
        # notice: Each sentence is recorded multiple times in file |word_info_df|.
        # subset the interest area report DF to a DF containing only the current sentence/text ID
        sn = word_info_df[word_info_df.sentenceid == sn_id]
        # sn is a dataframe containing only one sentence (the sentence with the current sentence ID)
        sn = sn[
            sn['list'] == sn.list.values.tolist()[0]]  # list = experimental list number (unique to each participant).
        # compute word length and frequency features for each word
        # BLLIP vocabulary size: 229,538 words.
        # so FREQ_BLLIP are already logs of word frequencies
        # why do we log_10 them again in compute_word_frequency()
        sn_str = sn.sentence.iloc[-1]  # the whole sentence as string
        if sn_id == '1987/w7_019/w7_019.295-3' or sn_id == '1987/w7_036/w7_036.147-43' or sn_id == '1987/w7_091/w7_091.360-6':
            # extra inverted commas at the end of the sentence
            sn_str = sn_str[:-3] + sn_str[-1:]
        if sn_id == '1987/w7_085/w7_085.200-18':
            sn_str = sn_str[:43] + sn_str[44:]
        # sn_len = len(tokenizer.tokenize(sn_str))

        # skip nan values bc they are of type float (np.isnan raises an error)
        if isinstance(sn_str, float):
            continue

        sn_len = len(sn_str.split())

        # add CLS and SEP 'manually' to the sentence so that they receive the word IDs 0 and len(sn)+1
        sn_str = '[CLS] ' + sn_str + ' [SEP]'

        tokenizer.padding_side = 'right'

        for sub_id_idx, sub_id in enumerate(reader_list):

            if sub_id_idx == 5:
                continue

            sub_df = sn_df[sn_df.list == sub_id]
            # remove fixations on non-words
            sub_df = sub_df.loc[
                sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']  # Label for the interest area to which the currentixation is assigned
            if len(sub_df) == 0:
                # no scanpath data found for the subject
                continue

            # prepare decoder input and output
            sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

            # check if recorded fixation duration are within reasonable limits
            # Less than 15ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
            outlier_indx = np.where(sp_fix_dur < 50)[
                0]  # gives indices of the fixations in the fixations list that were shorter than 50ms

            if outlier_indx.size > 0:
                for out_idx in range(len(outlier_indx)):
                    outlier_i = outlier_indx[out_idx]
                    merge_flag = False

                    # outliers are commonly found in the fixation of the last record and the first record, and are removed directly
                    if outlier_i == len(sp_fix_dur) - 1 or outlier_i == 0:
                        merge_flag = True

                    else:
                        if outlier_i - 1 >= 0 and not merge_flag:
                            # try to merge with the left fixation if they landed both on the same interest area
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        if outlier_i + 1 < len(sp_fix_dur) and not merge_flag:
                            # try to merge with the right fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                    # delete the position (interest area ID), the fixation location and the fixation duration from the respective arrays
                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx - 1

            # sanity check
            # scanpath too long, remove outliers, speed up the inference; more than 50 fixations on sentence
            if len(sp_word_pos) > 50:  # 72/10684
                continue
            # scanpath too short for a normal length sentence
            if len(sp_word_pos) <= 1 and sn_len > 10:
                continue

            sp_ordinal_pos = sp_word_pos.astype(int)  # interest area index, i.e., word IDs in fixation report
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_fix_dur.append(sp_fix_dur)
            # preprocess landing position feature
            # assign missing value to 'nan'
            sp_fix_loc = np.where(sp_fix_loc == '.', np.nan, sp_fix_loc)
            # convert string of number of float type
            sp_fix_loc = [float(i) for i in sp_fix_loc]
            # Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
            if np.nanmax(
                    sp_fix_loc) > 35:  # returns fixation outliers (coordinates very off); np.nanmax returns the max value while igonoring nans
                missing_idx = np.where(np.array(sp_fix_loc) > 5)[0]  # array with indices where fix loc greater than 5
                for miss in missing_idx:
                    if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
                        sp_fix_loc[miss] = np.nan
                    else:
                        print('Landing position calculation error. Unknown cause, needs to be checked')
            SP_landing_pos.append(sp_fix_loc)

            encoded_sn = tokenizer.encode_plus(
                sn_str.split(),
                add_special_tokens=False,
                padding=False,
                return_attention_mask=False,
                is_split_into_words=True,
                truncation=False,
            )

            sn_word_ids = encoded_sn.word_ids()
            sp_word_ids = [0] + sp_ordinal_pos.tolist() + [max(encoded_sn.word_ids())]

            sn_input_ids = encoded_sn['input_ids']
            assert len(sn_word_ids) == len(sn_input_ids)

            max_len = max(max_len, len(sn_word_ids) + len(sp_word_ids))

            # truncating
            sep_token_sn_word_ids = sn_word_ids[-1]
            sep_token_sp_word_ids = sp_word_ids[-1]
            sep_token_sn_input_ids = sn_input_ids[-1]

            sn_word_ids = sn_word_ids[:-1]
            sp_word_ids = sp_word_ids[:-1]
            sn_input_ids = sn_input_ids[:-1]

            while len(sp_word_ids) > args.seq_len - 3:
                sp_word_ids.pop()

            # while len(sn_word_ids) + len(sp_word_ids) > args.seq_len - 3:
            #     if len(sn_word_ids) > len(sp_word_ids):
            #         sn_word_ids.pop()
            #         sn_input_ids.pop()
            #     elif len(sp_word_ids) > len(sn_word_ids):
            #         sp_word_ids.pop()
            #     else:
            #         sn_word_ids.pop()
            #         sn_input_ids.pop()
            #         sp_word_ids.pop()

            # add the SEP token word ID and input ID again
            sn_word_ids.append(sep_token_sn_word_ids)
            sp_word_ids.append(sep_token_sp_word_ids)
            sn_input_ids.append(sep_token_sn_input_ids)

            #   sn_sp_repr = sn_word_ids + sp_word_ids

            #   mask = [0] * len(sn_word_ids)
            #   mask_sn_padding = [0] * len(sn_word_ids) + [1] * len(sp_word_ids) + [0] * (args.seq_len - len(sn_word_ids) - len(sp_word_ids))
            mask_transformer_att = [1] * len(sp_word_ids) + [0] * (args.seq_len - len(sp_word_ids))

            #    indices_pos_enc = list(range(0, len(sn_word_ids))) + list(range(0, args.seq_len - len(sn_word_ids)))
            indices_pos_enc = list(range(0, args.seq_len))

            sn_repr_len = len(sn_word_ids)
            words_for_mapping = sn_str.split() + (args.seq_len - len(sn_str.split())) * ['[PAD]']

            #    data['mask'].append(mask)
            #    data['sn_sp_repr'].append(sn_sp_repr)
            data['sn_word_ids'].append(sn_word_ids)
            data['sp_word_ids'].append(sp_word_ids)
            data['sn_input_ids'].append(sn_input_ids)
            data['indices_pos_enc'].append(indices_pos_enc)
            data['sn_repr_len'].append(sn_repr_len)
            data['words_for_mapping'].append(' '.join(words_for_mapping))
            #     data['mask_sn_padding'].append(mask_sn_padding)
            data['mask_transformer_att'].append(mask_transformer_att)
            data['sn_ids'].append(sn_id)
            data['reader_ids'].append(sub_id)

            reader_IDs.append(sub_id)
            sn_IDs.append(sn_id)

    # padding
    # data['mask'] = _collate_batch_helper(
    #     examples=data['mask'],
    #     pad_token_id=1,
    #     max_length=args.seq_len,
    # )
    # data['sn_sp_repr'] = _collate_batch_helper(
    #     examples=data['sn_sp_repr'],
    #     pad_token_id=args.seq_len-1,
    #     max_length=args.seq_len,
    # )
    data['sp_word_ids'] = _collate_batch_helper(
        examples=data['sp_word_ids'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    data['sn_word_ids'] = _collate_batch_helper(
        examples=data['sn_word_ids'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    data['sn_input_ids'] = _collate_batch_helper(
        examples=data['sn_input_ids'],
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.seq_len,
    )

    splitting_IDs_dict = {
        'reader': reader_IDs,
        'sentence': sn_IDs,
    }

    if inference == 'cv':
        return data, splitting_IDs_dict

    # TODO the following section with the datasets and data splitting is ugly --> maybe change

    # if split == 'train':
    #
    #     dataset = Dataset2.from_dict(data)
    #     train_dataset = datasets.DatasetDict()
    #     train_dataset['train'] = dataset
    #     return train_dataset
    #
    # else:
    #
    #     # flatten the data
    #     flattened_data = list()
    #     for i in range(len(data['sn_sp_repr'])):
    #         flattened_data.append(
    #             (
    #                 data['mask'][i],
    #                 data['sn_sp_repr'][i],
    #                 data['sn_input_ids'][i],
    #                 data['indices_pos_enc'][i],
    #                 data['sn_repr_len'][i],
    #                 data['words_for_mapping'][i],
    #                 data['mask_sn_padding'][i],
    #                 data['mask_transformer_att'][i],
    #                 data['sn_ids'][i],
    #                 data['reader_ids'][i],
    #             )
    #         )
    #
    #     if split == 'train-test':
    #
    #         if split_sizes:
    #             test_size = split_sizes['test_size']
    #         else:
    #             test_size = 0.25
    #
    #         if splitting_criterion != 'scanpath':
    #
    #             if splitting_criterion == 'combined':
    #
    #                 train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
    #                     data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
    #                     sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
    #                 )
    #
    #             else:
    #
    #                 splitting_IDs = splitting_IDs_dict[splitting_criterion]
    #                 gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
    #                 for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
    #                     train_data = np.array(flattened_data)[train_index].tolist()
    #                     test_data = np.array(flattened_data)[test_index].tolist()
    #
    #         else:
    #             train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True, random_state=77)
    #
    #         # unflatten the data
    #         train_dataset = Dataset2.from_dict(
    #             {
    #                 'mask': [instance[0] for instance in train_data],
    #                 'sn_sp_repr': [instance[1] for instance in train_data],
    #                 'sn_input_ids': [instance[2] for instance in train_data],
    #                 'indices_pos_enc': [instance[3] for instance in train_data],
    #                 'sn_repr_len': [instance[4] for instance in train_data],
    #                 'words_for_mapping': [instance[5] for instance in train_data],
    #                 'mask_sn_padding': [instance[6] for instance in train_data],
    #                 'mask_transformer_att': [instance[7] for instance in train_data],
    #                 'sn_ids': [instance[8] for instance in train_data],
    #                 'reader_ids': [instance[9] for instance in train_data],
    #             }
    #         )
    #         test_dataset = Dataset2.from_dict(
    #             {
    #                 'mask': [instance[0] for instance in test_data],
    #                 'sn_sp_repr': [instance[1] for instance in test_data],
    #                 'sn_input_ids': [instance[2] for instance in test_data],
    #                 'indices_pos_enc': [instance[3] for instance in test_data],
    #                 'sn_repr_len': [instance[4] for instance in test_data],
    #                 'words_for_mapping': [instance[5] for instance in test_data],
    #                 'mask_sn_padding': [instance[6] for instance in test_data],
    #                 'mask_transformer_att': [instance[7] for instance in test_data],
    #                 'sn_ids': [instance[8] for instance in test_data],
    #                 'reader_ids': [instance[9] for instance in test_data],
    #             }
    #         )
    #         train_data_dict = datasets.DatasetDict()
    #         test_data_dict = datasets.DatasetDict()
    #         train_data_dict['train'] = train_dataset
    #         test_data_dict['test'] = test_dataset
    #         return train_data_dict, test_data_dict
    #
    #     elif split == 'train-val':
    #
    #         if split_sizes:
    #             val_size = split_sizes['val_size']
    #         else:
    #             val_size = 0.1
    #
    #         if splitting_criterion != 'scanpath':
    #
    #             if splitting_criterion == 'combined':
    #                 train_data, val_data, train_reader_IDs, val_reader_IDs, train_sn_IDs, val_sn_IDs = combined_split(
    #                     data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
    #                     sn_IDs=splitting_IDs_dict['sentence'], test_size=val_size,
    #                 )
    #             else:
    #                 splitting_IDs = splitting_IDs_dict[splitting_criterion]
    #                 gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
    #                 for train_index, val_index in gss.split(flattened_data, groups=splitting_IDs):
    #                     train_data = np.array(flattened_data)[train_index].tolist()
    #                     val_data = np.array(flattened_data)[val_index].tolist()
    #
    #         else:
    #             train_data, val_data = train_test_split(flattened_data, test_size=val_size, shuffle=True, random_state=77)
    #
    #         # unflatten the data
    #         train_dataset = Dataset2.from_dict(
    #             {
    #                 'mask': [instance[0] for instance in train_data],
    #                 'sn_sp_repr': [instance[1] for instance in train_data],
    #                 'sn_input_ids': [instance[2] for instance in train_data],
    #                 'indices_pos_enc': [instance[3] for instance in train_data],
    #                 'sn_repr_len': [instance[4] for instance in train_data],
    #                 'words_for_mapping': [instance[5] for instance in train_data],
    #                 'mask_sn_padding': [instance[6] for instance in train_data],
    #                 'mask_transformer_att': [instance[7] for instance in train_data],
    #                 'sn_ids': [instance[8] for instance in train_data],
    #                 'reader_ids': [instance[9] for instance in train_data],
    #             }
    #         )
    #         # unflatten the data
    #         val_dataset = Dataset2.from_dict(
    #             {
    #                 'mask': [instance[0] for instance in val_data],
    #                 'sn_sp_repr': [instance[1] for instance in val_data],
    #                 'sn_input_ids': [instance[2] for instance in val_data],
    #                 'indices_pos_enc': [instance[3] for instance in val_data],
    #                 'sn_repr_len': [instance[4] for instance in val_data],
    #                 'words_for_mapping': [instance[5] for instance in val_data],
    #                 'mask_sn_padding': [instance[6] for instance in val_data],
    #                 'mask_transformer_att': [instance[7] for instance in val_data],
    #                 'sn_ids': [instance[8] for instance in val_data],
    #                 'reader_ids': [instance[9] for instance in val_data],
    #             }
    #         )
    #         train_data_dict = datasets.DatasetDict()
    #         val_data_dict = datasets.DatasetDict()
    #         train_data_dict['train'] = train_dataset
    #         val_data_dict['val'] = val_dataset
    #         return train_data_dict, val_data_dict
    #
    #     elif split == 'train-val-test':
    #
    #         if split_sizes:
    #             val_size = split_sizes['val_size']
    #             test_size = split_sizes['test_size']
    #         else:
    #             val_size = 0.1
    #             test_size = 0.25
    #
    #         if splitting_criterion != 'scanpath':
    #
    #             if splitting_criterion == 'combined':
    #                 # split train and test data so that unseen readers and sentences are in the test data
    #                 # train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
    #                 #     data=flattened_data, splitting_IDs_dict=splitting_IDs_dict, test_size=test_size
    #                 # )
    #                 train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
    #                     data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
    #                     sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
    #                 )
    #                 # randomly split train data into train and validation
    #                 # TODO maybe change val data also into unseen readers and sentences?
    #                 train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=77, shuffle=True)
    #
    #             else:
    #                 splitting_IDs = splitting_IDs_dict[splitting_criterion]
    #
    #                 # split into train and test
    #                 gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
    #                 for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
    #                     train_data = np.array(flattened_data)[train_index].tolist()
    #                     test_data = np.array(flattened_data)[test_index].tolist()
    #                     train_ids = np.array(splitting_IDs)[train_index].tolist()
    #
    #                 # split into train and val
    #                 gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
    #                 for train_index, val_index in gss.split(train_data, groups=train_ids):
    #                     val_data = np.array(train_data)[val_index].tolist()
    #                     train_data = np.array(train_data)[train_index].tolist()
    #
    #         else:
    #             train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True, random_state=77)
    #             train_data, val_data = train_test_split(train_data, test_size=val_size, shuffle=True, random_state=77)
    #
    #         # unflatten the data
    #         train_dataset = Dataset2.from_dict(
    #             {
    #                 'mask': [instance[0] for instance in train_data],
    #                 'sn_sp_repr': [instance[1] for instance in train_data],
    #                 'sn_input_ids': [instance[2] for instance in train_data],
    #                 'indices_pos_enc': [instance[3] for instance in train_data],
    #                 'sn_repr_len': [instance[4] for instance in train_data],
    #                 'words_for_mapping': [instance[5] for instance in train_data],
    #                 'mask_sn_padding': [instance[6] for instance in train_data],
    #                 'mask_transformer_att': [instance[7] for instance in train_data],
    #                 'sn_ids': [instance[8] for instance in train_data],
    #                 'reader_ids': [instance[9] for instance in train_data],
    #             }
    #         )
    #         test_dataset = Dataset2.from_dict(
    #             {
    #                 'mask': [instance[0] for instance in test_data],
    #                 'sn_sp_repr': [instance[1] for instance in test_data],
    #                 'sn_input_ids': [instance[2] for instance in test_data],
    #                 'indices_pos_enc': [instance[3] for instance in test_data],
    #                 'sn_repr_len': [instance[4] for instance in test_data],
    #                 'words_for_mapping': [instance[5] for instance in test_data],
    #                 'mask_sn_padding': [instance[6] for instance in test_data],
    #                 'mask_transformer_att': [instance[7] for instance in test_data],
    #                 'sn_ids': [instance[8] for instance in test_data],
    #                 'reader_ids': [instance[9] for instance in test_data],
    #             }
    #         )
    #         val_dataset = Dataset2.from_dict(
    #             {
    #                 'mask': [instance[0] for instance in val_data],
    #                 'sn_sp_repr': [instance[1] for instance in val_data],
    #                 'sn_input_ids': [instance[2] for instance in val_data],
    #                 'indices_pos_enc': [instance[3] for instance in val_data],
    #                 'sn_repr_len': [instance[4] for instance in val_data],
    #                 'words_for_mapping': [instance[5] for instance in val_data],
    #                 'mask_sn_padding': [instance[6] for instance in val_data],
    #                 'mask_transformer_att': [instance[7] for instance in val_data],
    #                 'sn_ids': [instance[8] for instance in val_data],
    #                 'reader_ids': [instance[9] for instance in val_data],
    #             }
    #         )
    #         train_data_dict = datasets.DatasetDict()
    #         test_data_dict = datasets.DatasetDict()
    #         val_data_dict = datasets.DatasetDict()
    #         train_data_dict['train'] = train_dataset
    #         test_data_dict['test'] = test_dataset
    #         val_data_dict['val'] = val_dataset
    #         return train_data_dict, test_data_dict, val_data_dict


class CelerZucoDataset(Dataset):

    def __init__(
            self,
            dataset,
            data_args,
            split,  # 'train', 'test', 'val'
    ):
        super().__init__()
        self.dataset = dataset
        self.length = len(self.dataset[split])
        self.data_args = data_args
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {
            #    'mask': np.array(self.dataset[self.split][idx]['mask']),
            #    'sn_sp_repr': np.array(self.dataset[self.split][idx]['sn_sp_repr']),
            'sn_word_ids': np.array(self.dataset[self.split][idx]['sn_word_ids']),
            'sp_word_ids': np.array(self.dataset[self.split][idx]['sp_word_ids']),
            'sn_input_ids': np.array(self.dataset[self.split][idx]['sn_input_ids']),
            'indices_pos_enc': np.array(self.dataset[self.split][idx]['indices_pos_enc']),
            'sn_repr_len': np.array(self.dataset[self.split][idx]['sn_repr_len']),
            'words_for_mapping': self.dataset[self.split][idx]['words_for_mapping'],
            #    'mask_sn_padding': np.array(self.dataset[self.split][idx]['mask_sn_padding']),
            'mask_transformer_att': np.array(self.dataset[self.split][idx]['mask_transformer_att']),
            'sn_ids': self.dataset[self.split][idx]['sn_ids'],
            'reader_ids': self.dataset[self.split][idx]['reader_ids'],
        }
        return sample


def process_zuco(
        sn_list,
        reader_list,
        word_info_df,
        eyemovement_df,
        tokenizer,
        args,
        split: Optional[str] = 'train',
        subset_size: Optional[int] = None,
        split_sizes: Optional[Dict[str, float]] = None,
        splitting_criterion: Optional[str] = 'scanpath',  # 'reader', 'sentence', 'combined'
):
    """
        Process the ZuCo corpus so that it can be used as input to the Diffusion model, where the original sentence (sn)
        is the condition and the scan path (sp) is the target that will be noised.
        :param sn_list:
        :param reader_list:
        :param word_info_df:
        :param eyemovement_df:
        :param tokenizer:
        :param args:
        :param split: 'train', 'train-test', 'train-test-val', 'train-val'
        :param subset_size:
        :param split_sizes:
        :param splitting_criterion: how the data should be split for testing and validation. if 'scanpath', the split is
        just done at random for novel scanpaths; 'reader' will split according to readers, 'sentence' according to
        sentences, and 'combined' according to the combination of reader and sentence
        """
    SP_ordinal_pos = []
    SP_landing_pos = []
    SP_fix_dur = []

    data = {
        'mask': list(),  # 0 for sn, 1 for sp
        'sn_sp_repr': list(),  # word IDs of sn and corresponding word IDs of sp (fixated words, interest area IDs),
        # padded with args.seq_len -1
        'sn_input_ids': list(),  # input IDs of tokenized sentence, padded with pad token ID
        'indices_pos_enc': list(),  # indices from 1 ... len(sn input ids) 1 ... (seq_len - len(sn input ids))
        'sn_repr_len': list(),  # length of sentence in subword tokens
        'words_for_mapping': list(),  # original words of sentence, padded with PAD
        'mask_sn_padding': list(),  # masks both the sentence and the padding, for the loss computations
        'mask_transformer_att': list(),  # masks only the padding, for the transformer attention
    }

    max_len = 0
    all_lens = list()

    reader_IDs, sn_IDs = list(), list()

    for sn_id_idx, sn_id in tqdm(enumerate(sn_list)):

        if subset_size is not None:
            if sn_id_idx == subset_size + 1:
                break

        sn_df = eyemovement_df[eyemovement_df.sn == sn_id]
        sn = word_info_df[word_info_df.SN == sn_id]
        sn_str = ' '.join(sn.WORD.values)
        sn_len = len(sn_str.split())

        tokenizer.padding_side = 'right'
        sn_str = '[CLS] ' + sn_str + ' [SEP]'

        for sub_id_idx, sub_id in enumerate(reader_list):

            sub_df = sn_df[sn_df.id == sub_id]
            # remove fixations on non-words
            sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '']
            if len(sub_df) == 0:
                # no scanpath data found for the subject
                continue

            sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.wn.values, sub_df.fl.values, sub_df.dur.values

            # check if recorded fixation duration are within reasonable limits
            # Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
            outlier_indx = np.where(sp_fix_dur < 50)[0]
            if outlier_indx.size > 0:
                for out_idx in range(len(outlier_indx)):
                    outlier_i = outlier_indx[out_idx]
                    merge_flag = False
                    if outlier_i - 1 >= 0 and not merge_flag:
                        # try to merge with the left fixation
                        if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                            sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                    if outlier_i + 1 < len(sp_fix_dur) and not merge_flag:
                        # try to merge with the right fixation
                        if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                            sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx - 1

            # sanity check
            # scanpath too long, remove outliers, speed up the inference
            # if len(sp_word_pos) > 100:
            # continue
            # scanpath too short for a normal length sentence
            if len(sp_word_pos) <= 1 and sn_len > 10:
                continue

            sp_ordinal_pos = sp_word_pos.astype(int)
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_fix_dur.append(sp_fix_dur)

            # preprocess landing position feature
            # assign missing value to 'nan'
            # sp_fix_loc=np.where(sp_fix_loc=='.', np.nan, sp_fix_loc)
            # convert string of number of float type
            sp_fix_loc = [float(i) if isinstance(i, int) or isinstance(i, float) else np.nan for i in sp_fix_loc if
                          isinstance(i, int) or isinstance(i, float)]
            SP_landing_pos.append(sp_fix_loc)

            # encode the sentence
            encoded_sn = tokenizer.encode_plus(
                sn_str.split(),
                add_special_tokens=False,
                padding=False,
                return_attention_mask=False,
                is_split_into_words=True,
                truncation=False,
            )

            sn_word_ids = encoded_sn.word_ids()
            sp_word_ids = [0] + sp_ordinal_pos.tolist() + [max(encoded_sn.word_ids())]

            sn_input_ids = encoded_sn['input_ids']
            assert len(sn_word_ids) == len(sn_input_ids)

            max_len = max(max_len, len(sn_word_ids) + len(sp_word_ids))
            all_lens.append(len(sn_word_ids) + len(sp_word_ids))

            # truncating
            sep_token_sn_word_ids = sn_word_ids[-1]
            sep_token_sp_word_ids = sp_word_ids[-1]
            sep_token_sn_input_ids = sn_input_ids[-1]

            sn_word_ids = sn_word_ids[:-1]
            sp_word_ids = sp_word_ids[:-1]
            sn_input_ids = sn_input_ids[:-1]

            while len(sn_word_ids) + len(sp_word_ids) > args.seq_len - 3:
                if len(sn_word_ids) > len(sp_word_ids):
                    sn_word_ids.pop()
                    sn_input_ids.pop()
                elif len(sp_word_ids) > len(sn_word_ids):
                    sp_word_ids.pop()
                else:
                    sn_word_ids.pop()
                    sn_input_ids.pop()
                    sp_word_ids.pop()

            # add the SEP token word ID and input ID again
            sn_word_ids.append(sep_token_sn_word_ids)
            sp_word_ids.append(sep_token_sp_word_ids)
            sn_input_ids.append(sep_token_sn_input_ids)

            sn_sp_repr = sn_word_ids + sp_word_ids

            mask = [0] * len(sn_word_ids)
            mask_sn_padding = [0] * len(sn_word_ids) + [1] * len(sp_word_ids) + [0] * (args.seq_len - len(sn_word_ids) - len(sp_word_ids))
            mask_transformer_att = [1] * len(sn_word_ids) + [1] * len(sp_word_ids) + [0] * (args.seq_len - len(sn_word_ids) - len(sp_word_ids))

            indices_pos_enc = list(range(0, len(sn_word_ids))) + list(range(0, args.seq_len - len(sn_word_ids)))

            sn_repr_len = len(sn_word_ids)
            words_for_mapping = sn_str.split() + (args.seq_len - len(sn_str.split())) * ['[PAD]']

            data['mask'].append(mask)
            data['sn_sp_repr'].append(sn_sp_repr)
            data['sn_input_ids'].append(sn_input_ids)
            data['indices_pos_enc'].append(indices_pos_enc)
            data['sn_repr_len'].append(sn_repr_len)
            data['words_for_mapping'].append(' '.join(words_for_mapping))
            data['mask_sn_padding'].append(mask_sn_padding)
            data['mask_transformer_att'].append(mask_transformer_att)

            reader_IDs.append(sub_id)
            sn_IDs.append(sn_id)

    # padding
    data['mask'] = _collate_batch_helper(
        examples=data['mask'],
        pad_token_id=1,
        max_length=args.seq_len,
    )
    data['sn_sp_repr'] = _collate_batch_helper(
        examples=data['sn_sp_repr'],
        pad_token_id=args.seq_len - 1,
        max_length=args.seq_len,
    )
    data['sn_input_ids'] = _collate_batch_helper(
        examples=data['sn_input_ids'],
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.seq_len,
    )

    splitting_IDs_dict = {
        'reader': reader_IDs,
        'sentence': sn_IDs,
    }

    # TODO the following section with the datasets and data splitting is ugly --> maybe change

    if split == 'train':

        dataset = Dataset2.from_dict(data)
        train_dataset = datasets.DatasetDict()
        train_dataset['train'] = dataset
        return train_dataset

    else:

        # flatten the data
        flattened_data = list()
        for i in range(len(data['sn_sp_repr'])):
            flattened_data.append(
                (
                    data['mask'][i],
                    data['sn_sp_repr'][i],
                    data['sn_input_ids'][i],
                    data['indices_pos_enc'][i],
                    data['sn_repr_len'][i],
                    data['words_for_mapping'][i],
                    data['mask_sn_padding'][i],
                    data['mask_transformer_att'][i],
                )
            )

        if split == 'train-test':

            if split_sizes:
                test_size = split_sizes['test_size']
            else:
                test_size = 0.25

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':

                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )

                else:

                    splitting_IDs = splitting_IDs_dict[splitting_criterion]
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
                    for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        test_data = np.array(flattened_data)[test_index].tolist()

            else:
                train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'mask': [instance[0] for instance in train_data],
                    'sn_sp_repr': [instance[1] for instance in train_data],
                    'sn_input_ids': [instance[2] for instance in train_data],
                    'indices_pos_enc': [instance[3] for instance in train_data],
                    'sn_repr_len': [instance[4] for instance in train_data],
                    'words_for_mapping': [instance[5] for instance in train_data],
                    'mask_sn_padding': [instance[6] for instance in train_data],
                    'mask_transformer_att': [instance[7] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'mask': [instance[0] for instance in test_data],
                    'sn_sp_repr': [instance[1] for instance in test_data],
                    'sn_input_ids': [instance[2] for instance in test_data],
                    'indices_pos_enc': [instance[3] for instance in test_data],
                    'sn_repr_len': [instance[4] for instance in test_data],
                    'words_for_mapping': [instance[5] for instance in test_data],
                    'mask_sn_padding': [instance[6] for instance in test_data],
                    'mask_transformer_att': [instance[7] for instance in test_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            return train_data_dict, test_data_dict

        elif split == 'train-val':

            if split_sizes:
                val_size = split_sizes['val_size']
            else:
                val_size = 0.1

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':
                    train_data, val_data, train_reader_IDs, val_reader_IDs, train_sn_IDs, val_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=val_size,
                    )
                else:
                    splitting_IDs = splitting_IDs_dict[splitting_criterion]
                    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
                    for train_index, val_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        val_data = np.array(flattened_data)[val_index].tolist()

            else:
                train_data, val_data = train_test_split(flattened_data, test_size=val_size, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'mask': [instance[0] for instance in train_data],
                    'sn_sp_repr': [instance[1] for instance in train_data],
                    'sn_input_ids': [instance[2] for instance in train_data],
                    'indices_pos_enc': [instance[3] for instance in train_data],
                    'sn_repr_len': [instance[4] for instance in train_data],
                    'words_for_mapping': [instance[5] for instance in train_data],
                    'mask_sn_padding': [instance[6] for instance in train_data],
                    'mask_transformer_att': [instance[7] for instance in train_data],
                }
            )
            # unflatten the data
            val_dataset = Dataset2.from_dict(
                {
                    'mask': [instance[0] for instance in val_data],
                    'sn_sp_repr': [instance[1] for instance in val_data],
                    'sn_input_ids': [instance[2] for instance in val_data],
                    'indices_pos_enc': [instance[3] for instance in val_data],
                    'sn_repr_len': [instance[4] for instance in val_data],
                    'words_for_mapping': [instance[5] for instance in val_data],
                    'mask_sn_padding': [instance[6] for instance in val_data],
                    'mask_transformer_att': [instance[7] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, val_data_dict

        elif split == 'train-val-test':

            if split_sizes:
                val_size = split_sizes['val_size']
                test_size = split_sizes['test_size']
            else:
                val_size = 0.1
                test_size = 0.25

            if splitting_criterion != 'scanpath':

                if splitting_criterion == 'combined':
                    # split train and test data so that unseen readers and sentences are in the test data
                    # train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                    #     data=flattened_data, splitting_IDs_dict=splitting_IDs_dict, test_size=test_size
                    # )
                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )
                    # randomly split train data into train and validation
                    # TODO maybe change val data also into unseen readers and sentences?
                    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=77, shuffle=True)

                else:
                    splitting_IDs = splitting_IDs_dict[splitting_criterion]

                    # split into train and test
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=77)
                    for train_index, test_index in gss.split(flattened_data, groups=splitting_IDs):
                        train_data = np.array(flattened_data)[train_index].tolist()
                        test_data = np.array(flattened_data)[test_index].tolist()
                        train_ids = np.array(splitting_IDs)[train_index].tolist()

                    # split into train and val
                    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=77)
                    for train_index, val_index in gss.split(train_data, groups=train_ids):
                        val_data = np.array(train_data)[val_index].tolist()
                        train_data = np.array(train_data)[train_index].tolist()

            else:
                train_data, test_data = train_test_split(flattened_data, test_size=test_size, shuffle=True, random_state=77)
                train_data, val_data = train_test_split(train_data, test_size=val_size, shuffle=True, random_state=77)

            # unflatten the data
            train_dataset = Dataset2.from_dict(
                {
                    'mask': [instance[0] for instance in train_data],
                    'sn_sp_repr': [instance[1] for instance in train_data],
                    'sn_input_ids': [instance[2] for instance in train_data],
                    'indices_pos_enc': [instance[3] for instance in train_data],
                    'sn_repr_len': [instance[4] for instance in train_data],
                    'words_for_mapping': [instance[5] for instance in train_data],
                    'mask_sn_padding': [instance[6] for instance in train_data],
                    'mask_transformer_att': [instance[7] for instance in train_data],
                }
            )
            test_dataset = Dataset2.from_dict(
                {
                    'mask': [instance[0] for instance in test_data],
                    'sn_sp_repr': [instance[1] for instance in test_data],
                    'sn_input_ids': [instance[2] for instance in test_data],
                    'indices_pos_enc': [instance[3] for instance in test_data],
                    'sn_repr_len': [instance[4] for instance in test_data],
                    'words_for_mapping': [instance[5] for instance in test_data],
                    'mask_sn_padding': [instance[6] for instance in test_data],
                    'mask_transformer_att': [instance[7] for instance in test_data],
                }
            )
            val_dataset = Dataset2.from_dict(
                {
                    'mask': [instance[0] for instance in val_data],
                    'sn_sp_repr': [instance[1] for instance in val_data],
                    'sn_input_ids': [instance[2] for instance in val_data],
                    'indices_pos_enc': [instance[3] for instance in val_data],
                    'sn_repr_len': [instance[4] for instance in val_data],
                    'words_for_mapping': [instance[5] for instance in val_data],
                    'mask_sn_padding': [instance[6] for instance in val_data],
                    'mask_transformer_att': [instance[7] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, test_data_dict, val_data_dict
