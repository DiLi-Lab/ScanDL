"""
Run evaluation on ScanDL output.
"""

import os
import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser
from scripts.scanpath_similarity import levenshtein_distance
from scripts.scanpath_similarity import levenshtein_similarity
from scripts.scanpath_similarity import levenshtein_normalized_distance
from scripts.scanpath_similarity import levenshtein_normalized_similarity


def get_parser():
    parser = ArgumentParser(description='run evaluation of SCAND-L')
    parser.add_argument(
        '--cv',
        action='store_true',
        help='if given, evaluation is done on CV output'
    )
    parser.add_argument(
        '--generation_outputs',
        help='path to dir containing the model dirs with outputs.',
        type=str,
        required=True,
    )
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()
    path_to_outputs = os.path.join('generation_outputs', args.generation_outputs)

    if not args.cv:  # for cross-dataset

        model_output_dirs = os.listdir(path_to_outputs)

        metrics_dict = {
            'model_name': list(),
            'file_name': list(),
            'levenshtein_distance': list(),
            'levenshtein_similarity': list(),
            'normalized_levenshtein_distance': list(),
            'normalized_levenshtein_similarity': list(),
        }

        for dir in model_output_dirs:
            if not dir.endswith('samples'):
                continue
            output_files = os.listdir(os.path.join(path_to_outputs, dir))

            for out_file in output_files:
                if out_file.endswith('running_remove-PAD_rank0.json'):
                    print(f' --- processing {dir}/{out_file} ...')
                    path_to_file = os.path.join(path_to_outputs, dir, out_file)

                    with open(path_to_file, 'rb') as f:
                        model_output = json.load(f)

                        original_sp_ids = model_output['original_sp_ids']
                        predicted_sp_ids = model_output['predicted_sp_ids']

                        # replace empty lists with [PAD] or they will throw an error for levenshtein distance
                        original_sp_ids = [orig_sp_ids if orig_sp_ids != [] else [511] for orig_sp_ids in original_sp_ids]
                        predicted_sp_ids = [pred_sp_ids if pred_sp_ids != [] else [511] for pred_sp_ids in predicted_sp_ids]

                        ld_list, ls_list, nld_list, nls_list = list(), list(), list(), list()

                        for idx, (pred, orig) in enumerate(zip(predicted_sp_ids, original_sp_ids)):

                            print(f'\t --- computing levenshtein metrics for instance {idx} ...')

                            ld_list.append(levenshtein_distance(gt=orig, pred=pred))
                            ls_list.append(levenshtein_similarity(gt=orig, pred=pred))
                            nld_list.append(levenshtein_normalized_distance(gt=orig, pred=pred))
                            nls_list.append(levenshtein_normalized_similarity(gt=orig, pred=pred))

                        ld = np.round(np.mean(np.array(ld_list)), 5).item()
                        ls = np.round(np.mean(np.array(ls_list)), 5).item()
                        nld = np.round(np.mean(np.array(nld_list)), 5).item()
                        nls = np.round(np.mean(np.array(nls_list)), 5).item()

                        metrics_dict['model_name'].append(dir)
                        metrics_dict['file_name'].append(out_file)
                        metrics_dict['levenshtein_distance'].append(ld)
                        metrics_dict['levenshtein_similarity'].append(ls)
                        metrics_dict['normalized_levenshtein_distance'].append(nld)
                        metrics_dict['normalized_levenshtein_similarity'].append(nls)

        metrics_df = pd.DataFrame(metrics_dict)

        if 'fold' in args.generation_outputs:
            metrics_filename = f'metrics_{args.generation_outputs.split("/")[0]}.csv'
        else:
            metrics_filename = f'metrics_{args.generation_outputs}.csv'

        metrics_df.to_csv(os.path.join(path_to_outputs, metrics_filename), sep='\t')

    else:  # for all cross-validation settings

        fold_dirs = [f for f in os.listdir(path_to_outputs) if f.startswith('fold')]
        model_dirs = os.listdir(os.path.join(path_to_outputs, fold_dirs[-1]))
        metrics = ['ls', 'ld', 'nld', 'nls']
        model_dict = dict()

        for model in model_dirs:
            if not model.endswith('samples'):
                continue
            model_dict[model] = dict()
            for metric in metrics:
                model_dict[model][metric] = list()

            for fold in fold_dirs:
                if not fold.startswith('fold'):
                    continue
                print(f' --- processing model {model}, {fold} ...')
                path_to_file = os.path.join(path_to_outputs, fold, model)
                output_files = os.listdir(path_to_file)
                for out_file in output_files:
                    if out_file.endswith('running_remove-PAD_rank0.json'):
                        file_path = os.path.join(path_to_file, out_file)
                        with open(file_path, 'rb') as f:
                            model_output = json.load(f)
                        original_sp_ids = model_output['original_sp_ids']
                        predicted_sp_ids = model_output['predicted_sp_ids']

                        # replace empty lists with [PAD] or they will throw an error for levenshtein distance
                        original_sp_ids = [orig_sp_ids if orig_sp_ids != [] else [511] for orig_sp_ids in
                                           original_sp_ids]
                        predicted_sp_ids = [pred_sp_ids if pred_sp_ids != [] else [511] for pred_sp_ids in
                                            predicted_sp_ids]

                        ld_list, ls_list, nld_list, nls_list = list(), list(), list(), list()

                        for idx, (pred, orig) in enumerate(zip(predicted_sp_ids, original_sp_ids)):
                            print(f'\t --- computing levenshtein metrics for instance {idx} ...')

                            ld_list.append(levenshtein_distance(gt=orig, pred=pred))
                            ls_list.append(levenshtein_similarity(gt=orig, pred=pred))
                            nld_list.append(levenshtein_normalized_distance(gt=orig, pred=pred))
                            nls_list.append(levenshtein_normalized_similarity(gt=orig, pred=pred))

                        ld = np.mean(np.array(ld_list)).item()
                        ls = np.mean(np.array(ls_list)).item()
                        nld = np.mean(np.array(nld_list)).item()
                        nls = np.mean(np.array(nls_list)).item()

                        model_dict[model]['ld'].append(ld)
                        model_dict[model]['ls'].append(ls)
                        model_dict[model]['nld'].append(nld)
                        model_dict[model]['nls'].append(nls)

        metrics_dict = {
            'model_name': list(),
            'ld_mean': list(),
            'ld_std': list(),
            'ls_mean': list(),
            'ls_std': list(),
            'nld_mean': list(),
            'nld_std': list(),
            'nls_mean': list(),
            'nls_std': list(),
        }

        for model in model_dirs:
            if not model.endswith('samples'):
                continue

            metrics_dict['model_name'].append(model)

            metrics_dict['ld_mean'].append(np.mean(np.array(model_dict[model]['ld'])))
            metrics_dict['ld_std'].append(np.std(np.array(model_dict[model]['ld'])))
            metrics_dict['ls_mean'].append(np.mean(np.array(model_dict[model]['ls'])))
            metrics_dict['ls_std'].append(np.std(np.array(model_dict[model]['ls'])))
            metrics_dict['nld_mean'].append(np.mean(np.array(model_dict[model]['nld'])))
            metrics_dict['nld_std'].append(np.std(np.array(model_dict[model]['nld'])))
            metrics_dict['nls_mean'].append(np.mean(np.array(model_dict[model]['nls'])))
            metrics_dict['nls_std'].append(np.std(np.array(model_dict[model]['nls'])))

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(os.path.join(path_to_outputs, f'metrics_{args.generation_outputs}.csv'), sep='\t')


if __name__ == '__main__':
    main()
