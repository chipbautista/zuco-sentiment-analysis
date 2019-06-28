import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_files
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tflearn.data_utils import VocabularyProcessor

from settings import *


class SentimentDataSet():
    """
    Master class. Not actually used for the DataLoader, but holds all
    information about the data set.
    """
    def __init__(self, args):
        self.use_gaze = args.use_gaze

        categories = ['NEGATIVE', 'POSITIVE']
        if args.num_sentiments == 3:
            categories.append('NEUTRAL')

        dataset = load_files(
            container_path=DATASET_DIR, categories=categories,
            load_content=True, encoding='utf-8')
        self.sentences_, self.sentence_numbers, _, self.targets, _ = dataset.values()
        self.sentence_numbers = [int(re.search(r'\d{1,3}', fname).group())
                                 for fname in dataset['filenames']]

        # adopting Hollenstein's method of building the vocab
        self.sentences = [clean_str(s) for s in self.sentences_]
        self.num_sentences = len(self.sentences)
        self.max_sentence_length = max([len(s.split())
                                        for s in self.sentences])
        self.vocab_processor = VocabularyProcessor(self.max_sentence_length)
        self.indexed_sentences = torch.LongTensor(list(
            self.vocab_processor.fit_transform(self.sentences)))
        self.vocabulary = list(
            self.vocab_processor.vocabulary_._mapping.keys())

        print('\n> Data set loaded. Sentiment classes:', categories)
        print('> Max sentence length:', self.max_sentence_length, 'words.')

        if self.use_gaze:
            self.et_features = EyeTrackingFeatures(self.max_sentence_length)
            print('> Loaded eye-tracking features.\n')
        else:
            self.et_features = np.zeros(self.num_sentences)

    def split_cross_val(self, num_folds=10):
        cv = StratifiedKFold(num_folds, shuffle=True, random_state=111)
        splitter = cv.split(np.zeros(self.num_sentences), self.targets)
        for train_indices, test_indices in splitter:
            yield (self._get_dataloader(train_indices),
                   self._get_dataloader(test_indices))

    def _get_dataloader(self, indices):
        # need to do this in order to match the sentences imported by
        # load_files with the ones
        # given in the Matlab files (where we get et_features from)...
        # ONLY WORKS WHEN USING THE WHOLE DATA SET! (num_classes=3)
        indices_ = np.array([self.sentence_numbers.index(i) for i in indices])
        et_features = [self.et_features[i] for i in indices]

        # _sent_lengths = [len(self.sentences[i].split()) for i in indices_]
        # _et_lengths = [self.et_features.sentences_et[i].shape[0] for i in indices]
        # print('SENTENCES MATCH:', np.all(np.array(np.array(_et_lengths) == np.array(_sent_lengths))))
        dataset = SplitDataset(self.indexed_sentences[indices_],
                               self.targets[indices_],
                               et_features)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class SplitDataset(Dataset):
    """Send the train/test indices here and use as input to DataLoader."""
    def __init__(self, indexed_sentences, targets, et_features):
        self.indexed_sentences = indexed_sentences
        self.et_features = et_features
        self.targets = targets

    def __len__(self):
        return len(self.indexed_sentences)

    def __getitem__(self, idx):
        return (self.indexed_sentences[idx],
                self.et_features[idx],
                self.targets[idx])


class EyeTrackingFeatures():
    def __init__(self, max_sentence_length):
        self.max_sentence_length = max_sentence_length
        self.sentences_et = []
        # to normalize each of the selected ET features. "across corpus"
        self.normalizer = StandardScaler()

        self.import_et_features()

    def import_et_features(self):
        sentence_et_features = np.load(ET_FEATURES_DIR, allow_pickle=True)

        for si, sentence in enumerate(sentence_et_features):  # 400 of these
            sentence_et = []
            for wi, word in enumerate(sentence):
                features = np.array([word['nFixations'],
                                     word['FFD'],
                                     word['TRT'],
                                     word['GD'],
                                     word['GPT']])

                if not np.all(np.isnan(features)):
                    # sklearn's StandardScaler reads the 2nd axis as the
                    # features. The input to fit/transform should be (11, 5)
                    self.normalizer.partial_fit(features.T)
                    sentence_et.append(np.nanmean(features, axis=1))
                else:  # when a word does not have any recorded ET feature
                    sentence_et.append(np.array([np.nan] * 5))

            self.sentences_et.append(np.array(sentence_et))

        # We keep the NaN values at first so that it doesn't mess up
        # the normalization process.
        # Let's only convert them to 0s after normalizing.
        self.sentences_et = [np.nan_to_num(self.normalizer.transform(s))
                             for s in self.sentences_et]

    def __getitem__(self, index):
        et_features = torch.Tensor(self.sentences_et[index])
        missing_dims = self.max_sentence_length - et_features.shape[0]
        return torch.nn.functional.pad(et_features,
                                       (0, 0, 0, missing_dims),
                                       mode='constant')


def clean_str(string):
    # mostly copy pasted from Hollenstein's code...
    # had to change some because it messes up the matching of the words
    # in a sentence :fearful:
    # string = string.replace(".", "")
    string = re.sub(r'([\w ])(\.)+', r'\1', string)
    string = string.replace(",", "")
    # this messes up the correspondence of words between cleaned data set and obtained gaze data per word
    # string = string.replace("--", "")
    string = string.replace("`", "")
    string = string.replace("''", "")
    string = string.replace("' ", " ")
    # string = string.replace("*", "")
    string = string.replace("\\", "")
    string = string.replace(";", "")
    # string = string.replace("- ", " ")
    string = string.replace("/", "-")
    string = string.replace("!", "")
    string = string.replace("?", "")

    # added by chip
    # string = re.sub(r"[():]", "", string)
    # string = re.sub(r"-$", "", string)

    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()
