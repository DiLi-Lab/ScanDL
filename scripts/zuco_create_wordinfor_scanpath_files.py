from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pandas as pd
import scipy.io as io
from tqdm import tqdm

import CONSTANTS as C


def ZUCO_read_words(directory: str, task: str) -> None:
    # task: zuco11, zuco12, zuco21
    if task.startswith('zuco1'):
        full_subj = 'ZAB'  # subject with complete data
        # full_subj = "ZKW"
        directory = directory + 'zuco/'
    elif task == 'zuco21':
        full_subj = 'YAG'
        directory = directory + 'zuco2/'
    else:
        raise NotImplementedError(f'{task=} unknown')
    directory = os.path.join(directory, f'task{task[-1]}', 'Matlab_files')
    save_path = directory + '/Word_Infor.csv'
    sub_file_path = {}
    for file in sorted(os.listdir(directory)):
        if not file.endswith('.mat'):
            continue
        subj = file.split('_')[0][-3:]
        fpath = os.path.join(directory, file)
        sub_file_path[subj] = fpath

    # read words
    df = pd.DataFrame([], columns=['SN', 'NW', 'WORD'])
    if task.startswith('zuco1'):
        sentence_data = io.loadmat(
            sub_file_path[full_subj], squeeze_me=True, struct_as_record=False,
        )['sentenceData']
        for sn_idx in range(len(sentence_data)):
            sn_data = sentence_data[sn_idx]
            # print(sn_data._fieldnames)
            word_data = sn_data.word
            for word_idx in range(len(word_data)):
                df_tmp = pd.DataFrame(
                    [[sn_idx + 1, word_idx + 1, word_data[word_idx].content]],
                    columns=['SN', 'NW', 'WORD'],
                )

                df = pd.concat([df, df_tmp])

    else:
        mat = h5py.File(sub_file_path[full_subj])
        sentence_data = mat['sentenceData/word']
        for sn_idx in range(len(sentence_data)):
            sn_data = sentence_data[sn_idx]
            word_data = mat[sn_data[0]]['content']
            for word_idx in range(len(word_data)):
                item = word_data[word_idx]
                word = ''.join(chr(c) for c in mat[item[0]][()].reshape(-1))
                df_tmp = pd.DataFrame(
                    [[sn_idx + 1, word_idx + 1, word]], columns=['SN', 'NW', 'WORD'],
                )
                df = pd.concat([df, df_tmp])
    df.to_csv(save_path, sep='\t', index=False)


def add_word_len(directory, task):
    if task.startswith('zuco1'):
        directory = directory + 'zuco/'
    elif task == 'zuco21':
        directory = directory + 'zuco2/'
    else:
        raise NotImplementedError(f'{task=} unknown')
    directory = os.path.join(directory, f'task{task[-1]}', 'Matlab_files')
    word_infor_path = directory + '/Word_Infor.csv'
    word_infor_df = pd.read_csv(word_infor_path, sep='\t')
    word_infor_df['word_len'] = np.array(
        [len(word) for word in word_infor_df['WORD'].values],
    )
    word_infor_df.to_csv(word_infor_path, sep='\t', index=False)


def ZUCO_read_scanpath(directory, task):
    # TODO: ZUCO2: exclude YMH due to incomplete data because of dyslexia
    # if subject != 'YMH': already cleaned up in the dataset
    # task: zuco11, zuco12, zuco21
    if task.startswith('zuco1'):
        directory = directory + 'zuco/'
    elif task == 'zuco21':
        directory = directory + 'zuco2/'
    else:
        raise NotImplementedError(f'{task=} unknown')
    directory = os.path.join(directory, f'task{task[-1]}', 'Matlab_files')
    word_infor_path = directory + '/Word_Infor.csv'
    word_infor_df = pd.read_csv(word_infor_path, sep='\t')
    save_path = directory + '/scanpath.csv'

    df = pd.DataFrame([], columns=['id', 'sn', 'nw', 'wn', 'fl', 'dur'])
    for file in tqdm(sorted(os.listdir(directory))):
        if not file.endswith('.mat'):
            continue
        subj = file.split('_')[0][-3:]
        fpath = os.path.join(directory, file)
        # read scanpath
        if task.startswith('zuco1'):
            mat_file = io.loadmat(
                fpath, squeeze_me=True, struct_as_record=False,
            )
            sentence_data = mat_file['sentenceData']
            for sn_idx in tqdm(range(len(sentence_data))):
                sn_data = sentence_data[sn_idx]
                # print(sn_data._fieldnames)
                # skip empty scanpath
                if pd.isna(sn_data.allFixations):
                    continue
                # sanity check with word infor file
                assert word_infor_df[
                    word_infor_df['SN'] == (
                        sn_idx + 1
                    )
                ].iloc[0].WORD == sn_data.word[0].content, 'The sentence order is wrong!'
                # get scanpath data for sentence
                scanpath = sn_data.allFixations
                # print(scanpath._fieldnames)
                # check if scanpath is ndarray, otherwise convert scalar to ndarray
                if isinstance(scanpath.x, np.ndarray):
                    x = scanpath.x
                    y = scanpath.y
                    fix_dur = scanpath.duration
                else:
                    x = np.array([scanpath.x])
                    y = np.array([scanpath.y])
                    fix_dur = np.array([scanpath.duration])
                # get word bounding positions
                word_boundary = sn_data.wordbounds
                num_word = word_boundary.shape[0]

                # assign each fixation to the corresponding bounding boxes
                for fix_idx in range(x.shape[0]):
                    word_idx = np.where(
                        (word_boundary[:, 0] <= x[fix_idx]) &
                        (word_boundary[:, 2] >= x[fix_idx]) &
                        (word_boundary[:, 1] <= y[fix_idx]) &
                        (word_boundary[:, 3] >= y[fix_idx]),
                    )
                    assert len(word_idx) == 1, 'more than 1 word is matched!'
                    # skip fixations outside all boundary boxes
                    if word_idx[0].size == 0:
                        continue
                    fl = ((x[fix_idx] - word_boundary[word_idx, 0]) /
                          (word_boundary[word_idx, 2] - word_boundary[word_idx, 0]))[0][0]
                    df_tmp = pd.DataFrame(
                        [[
                            subj, sn_idx + 1, num_word, word_idx[0][0] + 1,
                            fl, fix_dur[fix_idx],
                        ]], columns=['id', 'sn', 'nw', 'wn', 'fl', 'dur'],
                    )
                    df = pd.concat([df, df_tmp])

        else:
            mat = h5py.File(fpath)
            word_bound = mat['sentenceData/wordbounds']
            scanpath_data = mat['sentenceData/allFixations']
            for sn_idx in tqdm(range(len(scanpath_data))):
                # compute word boundary
                word_bound_sn_data = word_bound[sn_idx]
                word_boundary = mat[word_bound_sn_data[0]][()]
                word_boundary = np.swapaxes(word_boundary, 1, 0)
                num_word = word_boundary.shape[0]

                # compute scanpath
                scanpath_sn_data = scanpath_data[sn_idx]
                # skip empty scanpath
                try:
                    x = mat[scanpath_sn_data[0]]['x']
                except KeyError:
                    continue
                y = mat[scanpath_sn_data[0]]['y']
                fix_dur = mat[scanpath_sn_data[0]]['duration']

                # iterate over each fixation, match to the corresponding word in the sentence
                for fix_idx in range(len(x)):
                    word_idx = np.where(
                        (word_boundary[:, 0] <= x[fix_idx]) &
                        (word_boundary[:, 2] >= x[fix_idx]) &
                        (word_boundary[:, 1] <= y[fix_idx]) &
                        (word_boundary[:, 3] >= y[fix_idx]),
                    )
                    assert len(word_idx) == 1, 'more than 1 word is matched!'
                    # skip fixations outside all boundary boxes
                    if word_idx[0].size == 0:
                        continue
                    fl = (
                        (x[fix_idx] - word_boundary[word_idx, 0]) /
                        (word_boundary[word_idx, 2] -
                         word_boundary[word_idx, 0])
                    )[0][0]
                    df_tmp = pd.DataFrame(
                        [[
                            subj, sn_idx + 1, num_word, word_idx[0][0] + 1,
                            fl, fix_dur[fix_idx][0],
                        ]], columns=['id', 'sn', 'nw', 'wn', 'fl', 'dur'],
                    )
                    df = pd.concat([df, df_tmp])

    df.to_csv(save_path, sep='\t', index=False)


def load_zuco_word_and_scanpth_data(directory, task):
    if task.startswith('zuco1'):
        directory = directory + 'zuco/'
    elif task == 'zuco21':
        directory = directory + 'zuco2/'
    else:
        raise NotImplementedError(f'{task=} unknown')
    directory = os.path.join(directory, f'task{task[-1]}', 'Matlab_files')
    word_infor_path = directory + '/Word_Infor.csv'
    word_infor_df = pd.read_csv(word_infor_path, sep='\t')
    scanpath_path = directory + '/scanpath.csv'
    scanpath_df = pd.read_csv(scanpath_path, sep='\t')
    return word_infor_df, scanpath_df


def add_current_fix_interest_area_label_for_sp(task: str = 'zuco12') -> None:
    directory = C.path_to_zuco
    word_info_df, eyemovement_df = load_zuco_word_and_scanpth_data(
        directory=directory, task=task,
    )
    if task.startswith('zuco1'):
        directory = directory + 'zuco/'
    elif task == 'zuco21':
        directory = directory + 'zuco2/'
    else:
        raise NotImplementedError(f'{task=} unknown')
    directory = os.path.join(directory, f'task{task[-1]}', 'Matlab_files')
    scanpath_path = directory + '/scanpath.csv'
    save_path = scanpath_path
    eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = ''
    for idx, row in eyemovement_df.iterrows():
        SN = row.sn
        NW = row.wn
        eyemovement_df.at[idx, 'CURRENT_FIX_INTEREST_AREA_LABEL'] = word_info_df.loc[
            np.logical_and(
                word_info_df.SN == SN, word_info_df.NW == NW,
            )
        ].WORD.values[0]

    eyemovement_df.to_csv(save_path, sep='\t', index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--zuco-task',
        type=str,
        choices=[
            # zuco 1
            'zuco11', 'zuco12',
            # zuco 2
            'zuco21',
        ],
    )
    args = parser.parse_args()
    print(f'Preparing word info for {args.zuco_task}...')
    # create word information for all the sentences in each task, save to csv file in the same task folder
    ZUCO_read_words(directory=C.path_to_zuco, task=args.zuco_task)
    print(f'Preparing scanpath info for {args.zuco_task}...')
    # create scanpath information for all subjects in each task, save to csv file in the same folder
    ZUCO_read_scanpath(  # 12 subject
        directory=C.path_to_zuco,
        task=args.zuco_task,
    )
    print(f'Preparing add fix interest area label for {args.zuco_task}...')
    add_current_fix_interest_area_label_for_sp(task=args.zuco_task)
    print(f'Preparing add word length for {args.zuco_task}...')
    add_word_len(directory=C.path_to_zuco, task=args.zuco_task)
    # load data, task: zuco11, zuco12, zuco21
    word_infor_df, scanpath_df = load_zuco_word_and_scanpth_data(
        directory=C.path_to_zuco,
        task=args.zuco_task,
    )


if __name__ == '__main__':
    raise SystemExit(main())
