import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as Dataset2
import datasets
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold, StratifiedKFold
from typing import Optional, List, Tuple, Union, Any, Dict
from CONSTANTS import PATH_TO_IA, PATH_TO_FIX, SUB_METADATA_PATH, path_to_zuco


def load_celer():
    path_to_fix = PATH_TO_FIX
    path_to_ia = PATH_TO_IA
    eyemovement_df = pd.read_csv(path_to_fix, delimiter='\t', low_memory=False)
    eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace('\t(.*)',
                                                                                                               '',
                                                                                                               regex=True)
    word_info_df = pd.read_csv(path_to_ia, delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
    return word_info_df, eyemovement_df


def load_celer_speakers(only_native_speakers: bool = True):
    sub_metadata_path = SUB_METADATA_PATH
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
        padded_examples.append(instance + (max_length-len(instance)) * [pad_token_id])
    return padded_examples


def infinite_loader(data_loader):
    while True:
        yield from data_loader



def celer_zuco_dataset_and_loader(
        data,
        data_args,
        split: str,
        deterministic=False,
        loop=True,
):

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


def load_zuco(task: str=None): # 'zuco11', 'zuco12'
    dir = path_to_zuco
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
  #  unique_sns_indices = {sn_ID: idx for (idx, reader_ID, sn_ID) in tuple_ids if not sn_ID.startswith('en')}
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


def flatten_data(data: Dict[str, List[Any]]):
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
                data['sn_ids'][i],
                data['reader_ids'][i],
            )
        )
    return flattened_data


def unflatten_data(flattened_data: List[Tuple[Any]], split: str):
    dataset = Dataset2.from_dict(
        {
            'mask': [instance[0] for instance in flattened_data],
            'sn_sp_repr': [instance[1] for instance in flattened_data],
            'sn_input_ids': [instance[2] for instance in flattened_data],
            'indices_pos_enc': [instance[3] for instance in flattened_data],
            'sn_repr_len': [instance[4] for instance in flattened_data],
            'words_for_mapping': [instance[5] for instance in flattened_data],
            'mask_sn_padding': [instance[6] for instance in flattened_data],
            'mask_transformer_att': [instance[7] for instance in flattened_data],
            'sn_ids': [instance[8] for instance in flattened_data],
            'reader_ids': [instance[9] for instance in flattened_data],
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
        inference: Optional[str] = None,  # cv, zuco
):
    """
    Process the Celer corpus so that it can be used as input to the Diffusion model, where the original sentence (sn)
    is the condition and the scan path (sp) is the target that will be noised.
    :param sn_list: list of unique sentence IDs in celer
    :param reader_list: list of reader IDs in celer
    :param word_info_df: pd Dataframe with sentence info
    :param eyemovement_df: pd Dataframe with fixation info
    :param tokenizer: BertTokenizer
    :param args:
    :param split: 'train', 'train-test', 'train-test-val', 'train-val'
    :param subset_size: for test runs: to not load whole dataset but specified no. of instances
    :param split_sizes: proportion of data going into train, test and val
    :param splitting_criterion: how the data should be split for testing and validation.
        'reader' = New Reader setting
        'sentence' = New Sentence setting
        'combined' = New Reader/New Sentence setting
        'scanpath' = data is split at random
    :param inference: if inference is cross-validation, the data is returned before splitting into train test val
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
        'sn_repr_len': list(), # length of sentence in subword tokens
        'words_for_mapping': list(),   # original words of sentence, padded with PAD
        'mask_sn_padding': list(),  # masks both the sentence and the padding, for the loss computations
        'mask_transformer_att': list(),  # masks only the padding, for the transformer attention,
        'sn_ids': list(),  # the sentence IDs
        'reader_ids': list(),  # the reader IDs
    }

    max_len = 0

    reader_IDs, sn_IDs = list(), list()

    for sn_id_idx, sn_id in tqdm(enumerate(sn_list), total=len(sn_list)):  # for text/sentence ID

        if subset_size is not None:
            if sn_id_idx == subset_size+1:
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
        sn_word_len = compute_word_length(sn.WORD_LEN.values)
        sn_word_freq = compute_word_frequency(
            sn.FREQ_BLLIP.values)  # FREQ-BLLIP -log2(word frequency) in BLLIP (Charniak et al. 2000).
        sn_str = sn.sentence.iloc[-1]  # the whole sentence as string
        if sn_id == '1987/w7_019/w7_019.295-3' or sn_id == '1987/w7_036/w7_036.147-43' or sn_id == '1987/w7_091/w7_091.360-6':
            # extra inverted commas at the end of the sentence
            sn_str = sn_str[:-3] + sn_str[-1:]
        if sn_id == '1987/w7_085/w7_085.200-18':
            sn_str = sn_str[:43] + sn_str[44:]

        # skip nan values bc they are of type float (np.isnan raises an error)
        if isinstance(sn_str, float):
            continue

        sn_len = len(sn_str.split())
        sn_split = sn_str.split()

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
                        if outlier_i - 1 >= 0 and merge_flag == False:
                            # try to merge with the left fixation if they landed both on the same interest area
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        if outlier_i + 1 < len(sp_fix_dur) and merge_flag == False:
                            # try to merge with the right fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
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
            data['sn_ids'].append(sn_id)
            data['reader_ids'].append(sub_id)

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
        pad_token_id=args.seq_len-1,
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

    if split == 'train':

        dataset = Dataset2.from_dict(data)
        train_dataset = datasets.DatasetDict()
        train_dataset['train'] = dataset
        return train_dataset, splitting_IDs_dict

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
                    data['sn_ids'][i],
                    data['reader_ids'][i],
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
                    'sn_ids': [instance[8] for instance in train_data],
                    'reader_ids': [instance[9] for instance in train_data],
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
                    'sn_ids': [instance[8] for instance in test_data],
                    'reader_ids': [instance[9] for instance in test_data],
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
                    'sn_ids': [instance[8] for instance in train_data],
                    'reader_ids': [instance[9] for instance in train_data],
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
                    'sn_ids': [instance[8] for instance in val_data],
                    'reader_ids': [instance[9] for instance in val_data],
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
                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )
                    # randomly split train data into train and validation
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
                    'sn_ids': [instance[8] for instance in train_data],
                    'reader_ids': [instance[9] for instance in train_data],
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
                    'sn_ids': [instance[8] for instance in test_data],
                    'reader_ids': [instance[9] for instance in test_data],
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
                    'sn_ids': [instance[8] for instance in val_data],
                    'reader_ids': [instance[9] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, test_data_dict, val_data_dict


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
            'mask': np.array(self.dataset[self.split][idx]['mask']),
            'sn_sp_repr': np.array(self.dataset[self.split][idx]['sn_sp_repr']),
            'sn_input_ids': np.array(self.dataset[self.split][idx]['sn_input_ids']),
            'indices_pos_enc': np.array(self.dataset[self.split][idx]['indices_pos_enc']),
            'sn_repr_len': np.array(self.dataset[self.split][idx]['sn_repr_len']),
            'words_for_mapping': self.dataset[self.split][idx]['words_for_mapping'],
            'mask_sn_padding': np.array(self.dataset[self.split][idx]['mask_sn_padding']),
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
        :param sn_list: list of unique sentence IDs in zuco
        :param reader_list: list of reader IDs in zuco
        :param word_info_df: pd Dataframe with sentence info
        :param eyemovement_df: pd Dataframe with fixation info
        :param tokenizer: BertTokenizer
        :param args:
        :param split: 'train', 'train-test', 'train-test-val', 'train-val'
        :param subset_size: for test runs: to not load whole dataset but specified no. of instances
        :param split_sizes: proportion of data going into train, test and val
        :param splitting_criterion: how the data should be split for testing and validation.
            'reader' = New Reader setting
            'sentence' = New Sentence setting
            'combined' = New Reader/New Sentence setting
            'scanpath' = data is split at random
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
        'sn_ids': list(),  # sentence IDs
        'reader_ids': list(),  # reader IDs
    }

    max_len = 0
    all_lens = list()

    reader_IDs, sn_IDs = list(), list()

    for sn_id_idx, sn_id in tqdm(enumerate(sn_list), total=len(sn_list)):

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
                    if outlier_i - 1 >= 0 and merge_flag == False:
                        # try to merge with the left fixation
                        if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                            outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                            sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                    if outlier_i + 1 < len(sp_fix_dur) and merge_flag == False:
                        # try to merge with the right fixation
                        if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                            outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                            sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                            merge_flag = True

                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx - 1

            # sanity check
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
            all_lens.append(len(sn_word_ids)+len(sp_word_ids))

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
            mask_sn_padding = [0] * len(sn_word_ids) + [1] * len(sp_word_ids) + [0] * (
                        args.seq_len - len(sn_word_ids) - len(sp_word_ids))
            mask_transformer_att = [1] * len(sn_word_ids) + [1] * len(sp_word_ids) + [0] * (
                        args.seq_len - len(sn_word_ids) - len(sp_word_ids))

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
            data['sn_ids'].append(sn_id)
            data['reader_ids'].append(sub_id)

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
                    data['sn_ids'][i],
                    data['reader_ids'][i],
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
                    'sn_ids': [instance[8] for instance in train_data],
                    'reader_ids': [instance[9] for instance in train_data],
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
                    'sn_ids': [instance[8] for instance in test_data],
                    'reader_ids': [instance[9] for instance in test_data],
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
                    'sn_ids': [instance[8] for instance in train_data],
                    'reader_ids': [instance[9] for instance in train_data],
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
                    'sn_ids': [instance[8] for instance in val_data],
                    'reader_ids': [instance[9] for instance in val_data],
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
                    train_data, test_data, train_reader_IDs, test_reader_IDs, train_sn_IDs, test_sn_IDs = combined_split(
                        data=flattened_data, reader_IDs=splitting_IDs_dict['reader'],
                        sn_IDs=splitting_IDs_dict['sentence'], test_size=test_size,
                    )
                    # randomly split train data into train and validation
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
                    'sn_ids': [instance[8] for instance in train_data],
                    'reader_ids': [instance[9] for instance in train_data],
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
                    'sn_ids': [instance[8] for instance in test_data],
                    'reader_ids': [instance[9] for instance in test_data],
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
                    'sn_ids': [instance[8] for instance in val_data],
                }
            )
            train_data_dict = datasets.DatasetDict()
            test_data_dict = datasets.DatasetDict()
            val_data_dict = datasets.DatasetDict()
            train_data_dict['train'] = train_dataset
            test_data_dict['test'] = test_dataset
            val_data_dict['val'] = val_dataset
            return train_data_dict, test_data_dict, val_data_dict



