from collections import defaultdict
import pandas as pd
import json
import re
import argparse
from glob import glob
from pathlib import Path
import string
import pickle5 as pickle
import os


from wordfreq import word_frequency

parser = argparse.ArgumentParser(description="Extract reading measures and annotate")
parser.add_argument("--model", dest="model", default="None")
parser.add_argument("--setting", dest="setting", default="None")
parser.add_argument("--steps", dest="steps", type=int, default=0)
parser.add_argument("--seed", dest="seed", type=int, default=0)
parser.add_argument("--original", dest="original_data", action="store_true")
parser.set_defaults(original_data=False)

args = parser.parse_args()

STOP_CHARS_SURP = []
BASELINES = ['ez-reader', 'local', 'traindist', 'uniform', 'swift']
Path("./pl_analysis/reading_measures").mkdir(exist_ok=True)


class ResultFiles:
    """
    Container for results files
    """
    def __init__(
        self,
        model,
        steps,
        setting,
        seed,
        original_data,
        root_path: str = "./generation_outputs",
    ):
        self.model = model
        if self.model in BASELINES:
            self.steps = 0
            self.setting = model
        else:
            self.steps = steps
            self.setting = setting
        self.seed = seed
        self.root_path = root_path
        self.original_data = original_data
        # model, steps, prediction
        self.annotations_predicted_data = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.annotations_original_data = defaultdict(pd.DataFrame)
        self.out_path = './pl_analysis/reading_measures'
        if self.original_data:
            self.original_data_file_paths = self.get_original_data_result_files()
        elif self.model in BASELINES:
            self.result_file_paths = self.get_baseline_files()
        else:
            self.result_file_paths = self.get_scandl_result_files()

    def get_baseline_files(self):
        # print(sorted(glob(f'{self.root_path}/{self.model}/*/ps_*.pickle')))
        return sorted(glob(f'{self.root_path}/{self.model}/*/*.pickle'))

    def get_original_data_result_files(self):
        original_data_files = sorted(glob(f'{self.root_path}/original_data/*/*.json'))
        return original_data_files

    def get_scandl_result_files(self):
        if self.setting == 'cross_dataset':
            result_files = sorted(glob(f'scandl/ema_0.9999_{self.steps:06d}.pt.samples/all_output_remove-PAD_seed{self.seed}.json', recursive=True))
        else:
            result_files = sorted(glob(f'{self.root_path}/scandl/fold*/*/output_w_nld.json', recursive=True))
        return result_files

    def get_annotations(self):
        if self.original_data:
            self.get_annotations_original_data()
        else:
            self.get_annotations_predicted_data()

    def get_annotations_original_data(self):
        for data_set_file in self.original_data_file_paths:
            surprisal_path = '/'.join(data_set_file.split('/')[:-1])
            annotated_file = Annotations(
                directory=data_set_file, surprisal_path=surprisal_path, original_data=self.original_data,
                model=self.model, steps=self.steps, setting=self.setting, fold=0
            )
            annotated_file.compute_events()
            annotations = annotated_file.merge_reading_measures_with_linguistic_information(key="original")
            self.annotations_original_data[data_set_file] = annotations

    def get_annotations_predicted_data(self):
        # get result files over all folds
        temp_annotations = []
        for result_file in self.result_file_paths:
            # result files include both predicted and original results
            if self.setting == 'cross_dataset':
                fold = 0
            elif self.model in BASELINES:
                fold = result_file.split('/')[-2]
            else:
                fold = result_file.split('/')[-3]
            print(f'getting annotations for {result_file}')
            surprisal_path = '/'.join(result_file.split('/')[:-1])
            annotated_file = Annotations(
                directory=result_file, surprisal_path=surprisal_path, original_data=self.original_data,
                model=self.model, steps=self.steps, setting=self.setting, fold=fold
            )
            annotated_file.compute_events()
            annotations = annotated_file.merge_reading_measures_with_linguistic_information(key="predicted")
            temp_annotations.append(annotations)
        self.annotations_predicted_data[self.setting][self.steps] = pd.concat(
                temp_annotations, axis=0
            )

    def create_outfiles(self):
        if self.original_data:
            self.create_outfiles_original()
        else:
            self.create_outfiles_predicted()

    def create_outfiles_original(self):
        for test_set,_ in self.annotations_original_data.items():
            dataset = test_set.split('/')[-2]
            self.annotations_original_data[test_set].to_csv(
                f"{self.out_path}/reading_measures_{dataset}.csv"
            )

    def create_outfiles_predicted(self):
        for model,_ in self.annotations_predicted_data.items():
            for steps, _ in self.annotations_predicted_data[model].items():
                self.annotations_predicted_data[model][steps].to_csv(
                    f"{self.out_path}/reading_measures_{model}_{steps}_predicted.csv")


class Annotations:
    """
    Class for annotations
    """
    def __init__(
        self,
        directory,
        surprisal_path,
        original_data,
        model,
        steps,
        setting,
        fold,
    ):
        self.directory = directory
        self.surprisal_path = surprisal_path
        self.original_data = original_data
        self.model = model
        self.steps = steps
        self.fold = fold
        self.setting = setting

        self.predicted_sp_ids = []
        self.original_sp_ids = []
        self.predicted_sp_words = []
        self.original_sp_words = []
        self.original_sn = []
        self.sentence_lengths = []
        self.reader_ids = []
        self.sentence_ids = []
        self.load_results()

        self.surprisal = []
        self.compute_surprisal()
        self.word_length = []
        self.compute_word_length()
        self.lexical_frequency = []
        self.compute_lexical_frequency()

        self.skipped = defaultdict(list)
        self.regressions = defaultdict(list)
        self.n_tot_count = defaultdict(list)
        self.n_firstpass_count = defaultdict(list)

        self.pad_predictions = defaultdict(list)
        self.out_of_sent_predictions = defaultdict(list)

    def load_results(self):
        if self.original_data:
            self.load_results_original()
        elif self.model in BASELINES:
            self.load_baseline_file()
        else:
            self.load_scandl_results()

    def load_baseline_file(self):
        file = open(self.directory, 'rb')
        results = pickle.load(file)
        self.predicted_sp_ids = results["predicted_sp_ids"]
        self.predicted_sp_words = results["predicted_sp_words"]
        self.original_sn = results["original_sn"]
        self.reader_ids = results["reader_ids"]
        self.sentence_ids = results["sn_ids"]
        self.get_sent_lengths()

    def load_swift_ez_file(self):
        file = open(self.directory, 'rb')
        results = pickle.load(file)
        counter = 0
        na_counter = 0
        for _, value in results.items():
            counter += 1
            try:
                if self.model in ['uniform', 'traindist', 'local']:
                    words = value['original_sn'].tolist().split()
                    self.reader_ids.append(value['reader_ids'])
                else:
                    words = value['sentence'].tolist()[0].split()
            except TypeError:
                na_counter += 1
                continue
            except AttributeError:
                na_counter += 1
                continue
            # remove data accidentally in pickle file (single numbers)
            if len(words) == 1:
                ...
            else:
                if self.model in ['uniform', 'traindist', 'local']:
                    self.original_sn.append(value['original_sn'].tolist())
                else:
                    self.original_sn.extend(value['sentence'].tolist())
                self.original_sp_ids.append(value['original_sp_ids'].tolist())  # TODO funktioniert n mehr für swift/ez: orig_ids
                # self.original_sp_words.append(' '.join(value['orig_words'].tolist()))
                self.predicted_sp_ids.append(value['predicted_sp_ids'].tolist())  # TODO funktioniert n mehr für swift/ez: orig_ids
                # self.predicted_sp_words.append(' '.join(value['pred_words'].tolist())) # something weird here
        # self.reader_ids = len(self.original_sn) * [-1]
        self.sentence_ids = len(self.original_sn) * [-1]
        self.get_sent_lengths()
        # print('nas/all: ', na_counter, '/',counter)

    def load_results_original(self):
        with open(self.directory, 'rb') as f:
            results = json.load(f)
            original_sp_ids, original_sp_words, original_sn, reader_ids, sentence_ids = \
                (results[key] for key in
                 ['original_sp_ids', 'original_sp_words', 'original_sn', 'reader_ids', 'sn_ids'])
            self.original_sp_ids = original_sp_ids
            self.original_sp_words = original_sp_words
            self.original_sn = original_sn
            self.reader_ids = reader_ids
            self.sentence_ids = sentence_ids
            self.get_sent_lengths()

    def load_scandl_results(self):
        with open(self.directory, 'rb') as f:
            results = json.load(f)
            predicted_sp_ids, original_sp_ids, predicted_sp_words, original_sp_words, original_sn = \
                (results[key] for key in
                 ['predicted_sp_ids', 'original_sp_ids', 'predicted_sp_words', 'original_sp_words', 'original_sn'])
            self.predicted_sp_ids = predicted_sp_ids
            self.original_sp_ids = original_sp_ids
            self.predicted_sp_words = predicted_sp_words
            self.original_sp_words = original_sp_words
            self.original_sn = original_sn
            self.reader_ids = len(original_sn)*[-1]
            self.sentence_ids = len(original_sn)*[-1]
            self.get_sent_lengths()

        assert len(self.predicted_sp_ids) == \
               len(self.original_sp_ids) == \
               len(self.predicted_sp_words) == \
               len(self.original_sp_words) == \
               len(self.original_sn), 'Not equal length in results files'

    @staticmethod
    def column(matrix, i):
        return [row[i] for row in matrix]

    def get_sent_lengths(self):
        if self.model in BASELINES:
            for sent in self.original_sn:
                self.sentence_lengths.append(len(sent.split(" "))+2)
        else:
            for sent in self.original_sn:
                self.sentence_lengths.append(len(sent.split(" ")))

    def compute_word_length(self):
        for sent in self.original_sn:
            words = sent.split(" ")
            word_lengths = [len(word) for word in words]
            self.word_length.append(word_lengths)

    def compute_lexical_frequency(self):
        for sent in self.original_sn:
            words = sent.split(" ")
            words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
            word_lengths = [word_frequency(word, 'en') for word in words]
            self.lexical_frequency.append(word_lengths)

    def compute_surprisal(self):
        # compute surprisal for each word in each sentence
        # check if surprisal already computed
        if self.setting == "cross_dataset":
            load = os.path.exists(f'{self.surprisal_path}/surprisal.pickle')
        else:
            load = os.path.exists(f'{self.surprisal_path}/surprisal.pickle')
        if load:
            if self.setting == "cross_dataset":
                with open(f'{self.surprisal_path}/surprisal.pickle', 'rb') as surprisal_file:
                    self.surprisal = pickle.load(surprisal_file)
            else:
                with open(f'{self.surprisal_path}/surprisal.pickle', 'rb') as surprisal_file:
                    self.surprisal = pickle.load(surprisal_file)
        else:
            from Surprisal import SurprisalScorer
            S = SurprisalScorer(model_name="gpt")
            for sent in self.original_sn:
                sent = re.sub(r'^\[CLS\]', '', sent)
                sent = re.sub(r'\[SEP\]$', '', sent)
                # replace CLS and SEP token with UNK (for predictions of CLS/SEP mid-scanpath)
                sent = sent.replace("[CLS]", S.tokenizer.unk_token).replace("[SEP]", S.tokenizer.unk_token)
                sent = sent.strip().lstrip()
                words = sent.split(" ")
                probs, offset = S.score(sent)
                surprisal = self.get_per_word_surprisal(offset, probs, sent, words)
                self.surprisal.append(surprisal)
            with open(f'{self.surprisal_path}/surprisal.pickle', 'wb') as surprisal_file:
                pickle.dump(self.surprisal, surprisal_file)

    @staticmethod
    def get_per_word_surprisal(offset, probs, sent, words):
        surprisal = []
        j = 0
        for i in range(0, len(words)):  # i index for reference word list
            try:
                # case 1: tokenized word = white-space separated word
                # print(f'{words[i]} ~ {sent[offset[j][0]:offset[j][1]]}')
                if words[i] == sent[offset[j][0]: offset[j][1]].strip():
                    surprisal += [probs[i]]
                    j += 1
                # case 2: tokenizer split subword tokens: merge subwords and add up surprisal values until the same
                else:
                    concat_token = sent[offset[j][0]: offset[j][1]].strip()
                    concat_surprisal = probs[j]
                    while concat_token != words[i]:
                        j += 1
                        concat_token += sent[
                                        offset[j][0]: offset[j][1]
                                        ].strip()
                        # define characters that should not be added to word surprisal values
                        if (
                                sent[offset[j][0]: offset[j][1]].strip()
                                not in STOP_CHARS_SURP
                        ):
                            concat_surprisal += probs[j]
                        if concat_token == words[i]:
                            surprisal += [concat_surprisal]
                            j += 1
                            break
            except IndexError:
                print(
                    f"Index error in sentence: {sent}, length: {len(sent)}"
                )
                break
        return surprisal

    @staticmethod
    def get_skips(ids, sent_length):
        skips = [0] * sent_length
        # i = original word sequence
        for i in range(0, sent_length):
            # print('looking for skips of ',i)
            if i not in ids:
                # print(i, ' not in ids')
                skips[i] = 1
            else:
                # get index of first instance in ids
                j = ids.index(i)
                # print(j, 'is index of first occurrence of ',i, ' in ids')
                if any(k > ids[j] for k in ids[0:j]):
                    skips[i] = 1
        return skips

    @staticmethod
    def get_regressions(ids, sent_length):
        regression = [0] * sent_length
        for i in range(0, sent_length-1):
            if i not in ids:
                regression[i] = 0
            else:
                j = ids.index(i)
                # find index of next word in sequence
                for k in range(j, len(ids)):
                    if ids[k]>ids[j]:
                        break
                    elif ids[k] == ids[j]:
                        pass
                    elif ids[k] < ids[j]:
                        regression[i] = 1
                        break
        return regression

    @staticmethod
    def get_n_tot_count(ids, sent_length):
        n_tot_count = [ids.count(scanpath_id) for scanpath_id in range(sent_length)]
        return n_tot_count

    @staticmethod
    def get_n_firstpass_count(ids, sent_length):
        n_firstpass = [0] * sent_length
        for i in range(0, sent_length):
            # print('looking for skips of ',i)
            if i not in ids:
                n_firstpass[i] = 0
            else:
                j = ids.index(i)
                for k in range(j, len(ids)):
                    if ids[k] == ids[j]:
                        n_firstpass[i] += 1
                    else:
                        break
        return n_firstpass

    def event_detection(self, list_of_ids, prediction):
        for ids, original_sentence_length in zip(list_of_ids, self.sentence_lengths):
            eos_id = original_sentence_length - 1
            if 127 in ids:
                self.pad_predictions[prediction].append(ids.count(127))
                no_pad_ids = list(filter(lambda a: a != 127, ids))
            else:
                no_pad_ids = ids
            if max(no_pad_ids) > eos_id:
                no_pad_ids = list(filter(lambda a: a != max(no_pad_ids), ids))
                self.out_of_sent_predictions[prediction].append(ids.count(max(ids)))
            if eos_id in set(no_pad_ids):
                ...
                # print(ids, ': SEP token in scanpath')
            if no_pad_ids[0] != 0:
                print('first token not BOS token')
                raise NotImplementedError
            if no_pad_ids.count(0) > 1:
                print('predicted BOS token in middle of scanpath')
                raise NotImplementedError

            self.skipped[prediction].append(self.get_skips(no_pad_ids, original_sentence_length))
            self.regressions[prediction].append(self.get_regressions(no_pad_ids, original_sentence_length))
            self.n_tot_count[prediction].append(self.get_n_tot_count(no_pad_ids, original_sentence_length))
            self.n_firstpass_count[prediction].append(self.get_n_firstpass_count(no_pad_ids, original_sentence_length))

    def compute_events(self):
        if self.original_data:
            self.event_detection(self.original_sp_ids, prediction="original")
        else:
            self.event_detection(self.predicted_sp_ids, prediction="predicted")

    def merge_reading_measures_with_linguistic_information(self, key) -> pd.DataFrame:
        all_frames = []
        for i, (
            reader_id, sentence_id,
            skipped, regressions, n_tot_count, n_firstpass_count,
            word_length, surprisal, lexical_frequency, word,
        ) in enumerate(zip(
            self.reader_ids,
            self.sentence_ids,
            self.skipped[key],
            self.regressions[key],
            self.n_tot_count[key],
            self.n_firstpass_count[key],
            self.word_length,
            self.surprisal,
            self.lexical_frequency,
            self.original_sn,
        )):
            if self.model in BASELINES:
                all_words = word.split(" ")
            else:
                all_words = word.split(" ")[1:-1]
                word_length = word_length[1:-1]
                lexical_frequency = lexical_frequency[1:-1]
            skipped = skipped[1:-1]
            regressions = regressions[1:-1]
            n_firstpass_count = n_firstpass_count[1:-1]
            n_tot_count = n_tot_count[1:-1]

            assert len(skipped) == len(regressions) == len(n_tot_count) == len(n_firstpass_count) \
                    == len(surprisal) == len(lexical_frequency), \
                    f'not equal length {skipped}, {word_length}, {word} {self.original_sn.index(word)}'

            df = pd.DataFrame({
                'generation': key,
                'model': self.model,
                'steps': self.steps,
                'fold': self.fold,
                'reader_id': reader_id,
                'sentence_id': sentence_id,
                'word_number': range(len(skipped)),
                'word': all_words,
                'skipped': skipped,
                'regression': regressions,
                'n_firstpass_count': n_firstpass_count,
                'n_tot_count': n_tot_count,
                'word_length': word_length,
                'surprisal': surprisal,
                'lexical_frequency': lexical_frequency,
            },
                columns=['generation', 'model', 'steps', 'fold',
                         'reader_id', 'sentence_id',
                         'word_number', 'word',
                         'skipped', 'regression', 'n_firstpass_count', 'n_tot_count',
                         'word_length', 'surprisal', 'lexical_frequency']
            )
            all_frames.append(df)
        return pd.concat(all_frames, axis=0, ignore_index=True)


def main() -> int:
    results = ResultFiles(
        model=args.model, steps=args.steps, setting=args.setting, seed=args.seed, original_data=args.original_data)
    results.get_annotations()
    results.create_outfiles()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

