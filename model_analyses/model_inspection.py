from collections import defaultdict
import pandas as pd
import json
from glob import glob
from pathlib import Path

STOP_CHARS_SURP = []
BASELINES = ['ez-reader', 'local', 'traindist', 'uniform', 'swift']
Path("./pl_analysis/reading_measures").mkdir(parents=True, exist_ok=True)


def get_average_over_sentlen(lst, sent_len):
    if sent_len != 0:
        return sum(lst) / sent_len # TODO: -2 because of CLS / SEP token
    else:
        return 0


def get_average_over_fixations(lst):
    if len(lst) != 0:
        return sum(lst) / len(lst)
    else:
        return 0


class ResultFilesCrossModel:
    """
    Container for results files
    """
    def __init__(
            self,
            root_path: str = "./generation_outputs/model_inspection/model_results",
    ):
        self.root_path = root_path
        self.annotations_predicted_data = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.annotations_original_data = defaultdict(pd.DataFrame)
        self.out_path = './pl_analysis/rm_model_inspection/cross_model'
        self.result_file_paths = self.get_files()
        self.model = False

    def get_files(self):
        data_files = sorted(glob(f'{self.root_path}/*/*/*.json'))
        return data_files

    def get_annotations(self):
        all_df = []
        for result_file in self.result_file_paths:
            self.model = result_file.split('/')[-3]
            fold = result_file.split('/')[-2]  # fold for models, dataset for original dataa
            print(f'getting annotations for {result_file}')
            annotated_file = Annotations(
                directory=result_file, original_data=False,
                model=self.model, fold=fold, scandl_only=False,
            )
            annotated_file.compute_events()
            all_df.append(annotated_file.create_crossmodel_df())
        out_df = pd.concat(all_df, axis=0, ignore_index=True)
        out_df.to_csv(
            f"{self.out_path}/reading_measures_crossmodel_combined.csv"
        )


class ResultFilesScandlOnly:
    """
    Container for results files
    """
    def __init__(
            self,
            root_path: str = "./generation_outputs/model_inspection/scandl_only",
    ):
        self.root_path = root_path
        self.model = "scandl"
        self.annotations = defaultdict(pd.DataFrame)
        self.out_path = './pl_analysis/rm_model_inspection'
        self.result_file_paths = self.get_files()

    def get_files(self):
        all_files = sorted(glob(f'{self.root_path}/*/*/*.json'))
        return all_files

    def get_annotations(self):
        for result_file in self.result_file_paths:
            fold = result_file.split('/')[-2]
            dataset = result_file.split('/')[-3]
            print(f'getting annotations for {result_file} fold {fold}')
            annotated_file = Annotations(
                directory=result_file, original_data=False,
                model=self.model, fold=fold, scandl_only=True,
            )
            annotated_file.compute_events()
            df_out = annotated_file.create_nld_df()
            df_out.to_csv(
                f"{self.out_path}/reading_measures_scandl_only_{dataset}_{fold}.csv"
            )


class Annotations:
    """
    Class for annotations
    """
    def __init__(
        self,
        directory,
        original_data,
        model,
        fold,
        scandl_only
    ):
        self.directory = directory
        self.original_data = original_data
        self.model = model
        self.fold = fold

        self.predicted_sp_ids = []
        self.original_sp_ids = []
        self.predicted_sp_words = []
        self.original_sp_words = []
        self.original_sn = []
        self.original_sn_nostok = []
        self.sentence_lengths = []
        self.reader_ids = []
        self.sentence_ids = []
        self.nld = []
        if scandl_only:
            self.load_results_scandl_only()
        else:
            self.load_results_crossmodel()

        # per word measures
        self.skipped = defaultdict(list)
        self.regressions = defaultdict(list)
        self.n_tot_count = defaultdict(list)
        self.n_firstpass_count = defaultdict(list)

        # sentence measures
        self.progressive_saccades = defaultdict(list)
        self.regressive_saccades = defaultdict(list)

        self.pad_predictions = defaultdict(list)
        self.out_of_sent_predictions = defaultdict(list)

        self.avg_firstpass = defaultdict(list)
        self.avg_tft = defaultdict(list)
        self.avg_skips = defaultdict(list)
        self.avg_regs = defaultdict(list)
        self.avg_psacc = defaultdict(list)
        self.avg_rsacc = defaultdict(list)

    def load_results_scandl_only(self):
        with open(self.directory, 'rb') as f:
            results = json.loads(f.read())
            predicted_sp_ids, original_sp_ids, predicted_sp_words, original_sp_words, original_sn, reader_ids,\
                sn_ids, nld = \
                (results[key] for key in
                 ['predicted_sp_ids', 'original_sp_ids', 'predicted_sp_words', 'original_sp_words', 'original_sn',
                  'reader_ids', 'sn_ids', 'nld'])

            self.predicted_sp_ids = predicted_sp_ids
            self.original_sp_ids = original_sp_ids
            self.predicted_sp_words = predicted_sp_words
            self.original_sp_words = original_sp_words
            self.original_sn = original_sn
            self.reader_ids = reader_ids
            self.sentence_ids = sn_ids
            self.nld = nld
            self.get_sent_lengths()

        assert len(self.predicted_sp_ids) == \
               len(self.original_sp_ids) == \
               len(self.predicted_sp_words) == \
               len(self.original_sp_words) == \
               len(self.original_sn), 'Not equal length in results files'

    def load_results_crossmodel(self):
        with open(self.directory, 'rb') as f:
            results = json.load(f)
            predicted_sp_ids, original_sp_ids, original_sn, nld = \
                (results[key] for key in
                 ['predicted_sp_ids', 'original_sp_ids', 'original_sn', 'nld'])
            self.predicted_sp_ids = predicted_sp_ids
            self.original_sp_ids = original_sp_ids
            if self.model in ['scandl', 'original_data']:
                self.original_sn_nostok = [sent.replace("[CLS] ", "").replace(" [SEP]", "") for sent in original_sn]
            else:
                self.original_sn_nostok = original_sn
            self.original_sn = original_sn
            self.get_sent_lengths()
            self.nld = nld

        # replace CLS 50 with highest number
        updated_ids = []
        if self.model not in ['scandl', 'original_data']:
            for sent_len, original_sp_ids, sent in zip(self.sentence_lengths, self.predicted_sp_ids, self.original_sn):
                original_sp_ids[-1] = sent_len-1
                updated_ids.append(original_sp_ids)
            self.predicted_sp_ids = updated_ids

        assert len(self.predicted_sp_ids) == \
               len(self.original_sp_ids) == \
               len(self.original_sn), 'Not equal length in results files'

    @staticmethod
    def column(matrix, i):
        return [row[i] for row in matrix]

    def get_sent_lengths(self):
        # models with CLS in original_sn: original scandl
        # models w/o CLS in original_sn: Eyettention ez-reader traindist uniform
        if self.model in ['scandl', 'original_data']:
            for sent in self.original_sn:
                self.sentence_lengths.append(len(sent.split(" ")))
        else:
            for sent in self.original_sn:
                try:
                    self.sentence_lengths.append(len(sent.split(" ")) + 2)
                except AttributeError:
                    self.sentence_lengths.append(0)

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
                    if ids[k] > ids[j]:
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

    @staticmethod
    # TODO weight this with word length?
    def compute_saccade_lengths(fixation_array):
        saccade_lengths = []
        for i in range(1, len(fixation_array)):
            saccade_lengths.append(fixation_array[i] - fixation_array[i - 1])

        progressive_saccades = [length for length in saccade_lengths if length > 0]
        regressive_saccades = [length for length in saccade_lengths if length < 0]

        return progressive_saccades, regressive_saccades

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
                # print('first token not BOS token')
                pass
            if no_pad_ids.count(0) > 1:
                # print('predicted BOS token in middle of scanpath')
                pass

            self.skipped[prediction].append(self.get_skips(no_pad_ids, original_sentence_length))
            self.regressions[prediction].append(self.get_regressions(no_pad_ids, original_sentence_length))
            self.n_tot_count[prediction].append(self.get_n_tot_count(no_pad_ids, original_sentence_length))
            self.n_firstpass_count[prediction].append(self.get_n_firstpass_count(no_pad_ids, original_sentence_length))
            self.progressive_saccades[prediction].append(self.compute_saccade_lengths(no_pad_ids)[0])
            self.regressive_saccades[prediction].append(self.compute_saccade_lengths(no_pad_ids)[1])

    def compute_events(self):
        if self.original_data:
            self.event_detection(self.original_sp_ids, prediction="original")
        else:
            self.event_detection(self.original_sp_ids, prediction="original")
            self.event_detection(self.predicted_sp_ids, prediction="predicted")
            self.get_summary_rms()

    def get_summary_rms(self):
        for key in ['original', 'predicted']:
            for measure, s_len in zip(self.n_firstpass_count[key], self.sentence_lengths):
                self.avg_firstpass[key].append(get_average_over_sentlen(measure, s_len))
            for measure, s_len in zip(self.n_tot_count[key], self.sentence_lengths):
                self.avg_tft[key].append(get_average_over_sentlen(measure, s_len))
            for measure, s_len in zip(self.progressive_saccades[key], self.sentence_lengths):
                self.avg_psacc[key].append(get_average_over_fixations(measure))
            for measure, s_len in zip(self.regressive_saccades[key], self.sentence_lengths):
                self.avg_rsacc[key].append(get_average_over_fixations(measure))
            for measure, s_len in zip(self.regressions[key], self.sentence_lengths):
                self.avg_regs[key].append(get_average_over_sentlen(measure, s_len))
            for measure, s_len in zip(self.skipped[key], self.sentence_lengths):
                self.avg_skips[key].append(get_average_over_sentlen(measure, s_len))

    def create_crossmodel_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
                'model': self.model,
                'sentence': self.original_sn_nostok,
                's_len': self.sentence_lengths,
                'avg_firstpass_predicted': self.avg_firstpass['predicted'],
                'avg_tft_predicted': self.avg_tft['predicted'],
                'avg_skips_predicted': self.avg_skips['predicted'],
                'avg_regs_predicted': self.avg_regs['predicted'],
                'avg_psacc_predicted': self.avg_psacc['predicted'],
                'avg_rsacc_predicted': self.avg_rsacc['predicted'],
                'avg_firstpass_original': self.avg_firstpass['original'],
                'avg_tft_original': self.avg_tft['original'],
                'avg_skips_original': self.avg_skips['original'],
                'avg_regs_original': self.avg_regs['original'],
                'avg_psacc_original': self.avg_psacc['original'],
                'avg_rsacc_original': self.avg_rsacc['original'],
                'nld': self.nld,
                'fold': self.fold,
            })
        df = df.drop(df[
                         df['s_len'] <= 3].index)
        return df

    def create_nld_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
                'model': self.model,
                'reader_id': self.reader_ids,
                'sentence_id': self.sentence_ids,
                'sentence': self.original_sn,
                's_len': self.sentence_lengths,
                'avg_firstpass_predicted': self.avg_firstpass['predicted'],
                'avg_tft_predicted': self.avg_tft['predicted'],
                'avg_skips_predicted': self.avg_skips['predicted'],
                'avg_regs_predicted': self.avg_regs['predicted'],
                'avg_psacc_predicted': self.avg_psacc['predicted'],
                'avg_rsacc_predicted': self.avg_rsacc['predicted'],
                'avg_firstpass_original': self.avg_firstpass['original'],
                'avg_tft_original': self.avg_tft['original'],
                'avg_skips_original': self.avg_skips['original'],
                'avg_regs_original': self.avg_regs['original'],
                'avg_psacc_original': self.avg_psacc['original'],
                'avg_rsacc_original': self.avg_rsacc['original'],
                'nld': self.nld,
                'fold': self.fold,
            })
        return df


def main() -> int:
    results_inspection = ResultFilesScandlOnly()
    results_inspection.get_annotations()
    results_cross_models = ResultFilesCrossModel()
    results_cross_models.get_annotations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

