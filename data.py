import re

import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import load_files
from sklearn.preprocessing import StandardScaler

from settings import *


class SentimentDataSet(Dataset):
    def __init__(self, mode='binary'):
        categories = ['NEGATIVE', 'POSITIVE']
        if mode == 'ternary':
            categories.append('NEUTRAL')

        dataset = load_files(
            container_path=DATASET_DIR, categories=categories,
            load_content=True, encoding='utf-8')
        self.sentences, _, _, self.targets, _ = dataset
        self.sentences = [self.clean_str(s) for s in self.sentences]
        self.et_features = EyeTrackingFeatures()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (self.sentences[idx], self.targets[idx], self.et_features[idx])

    def clean_str(self, string):
        # copy pasted from Hollenstein's code...
        # TO-DO: Convert this to regex
        string = string.replace(".", "")
        string = string.replace(",", "")
        string = string.replace("--", "")
        string = string.replace("`", "")
        string = string.replace("''", "")
        string = string.replace("' ", " ")
        string = string.replace("*", "")
        string = string.replace("\\", "")
        string = string.replace(";", "")
        string = string.replace("- ", " ")
        string = string.replace("/", "-")
        string = string.replace("!", "")
        string = string.replace("?", "")
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()


class EyeTrackingFeatures(object):
    def __init__(self):
        # store sentences here. each sentence has length N = number of words.
        self.sentences = []
        # to normalize each of the selected ET features. "across corpus"
        self.normalizer = StandardScaler()

        self.import_et_features()

    def import_et_features(self):
        sentence_et_features = np.load(ET_FEATURES, allow_pickle=True)

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

            self.sentences.append(np.array(sentence_et))
        self.sentences = [self.normalizer.transform(s) for s in self.sentences]

    def __getitem__(self, index):
        return self.sentences[index]
