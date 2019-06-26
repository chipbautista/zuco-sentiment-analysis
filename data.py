
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_files
from sklearn.preprocessing import StandardScaler
from tflearn.data_utils import VocabularyProcessor

from settings import *
from utils import clean_str


class SentimentDataSet(Dataset):
    def __init__(self, mode='binary'):
        categories = ['NEGATIVE', 'POSITIVE']
        if mode == 'ternary':
            categories.append('NEUTRAL')

        self.num_classes = len(categories)

        dataset = load_files(
            container_path=DATASET_DIR, categories=categories,
            load_content=True, encoding='utf-8')
        self.sentences, _, _, self.targets, _ = dataset.values()
        self.sentences = [clean_str(s) for s in self.sentences]

        self.num_sentences = len(self.sentences)
        self.max_sentence_length = max([len(s.split())
                                        for s in self.sentences])

        # adopting Hollenstein's method
        self.vocab_processor = VocabularyProcessor(self.max_sentence_length)
        self.indexed_sentences = torch.LongTensor(list(
            self.vocab_processor.fit_transform(self.sentences)))
        self.vocabulary = list(
            self.vocab_processor.vocabulary_._mapping.keys())

        print('> Data set loaded. Sentiment classes:', categories)
        print('> Max sentence length:', self.max_sentence_length, 'words.')

        self.et_features = EyeTrackingFeatures(self.max_sentence_length)
        print('> Loaded eye-tracking features.')

    def get_train_test_split(self):
        # this isn't real CV though, because i'm just randomizing.
        # fix this later.
        split = int(np.ceil(self.num_sentences * .9))
        indices = np.array(list(range(len(self.sentences))))
        np.random.shuffle(indices)
        train_indices = indices[:split]
        test_indices = indices[split:]
        return (self.get_dataloader(train_indices),
                self.get_dataloader(test_indices))

    def get_dataloader(self, indices):
        et_features = [self.et_features[i] for i in indices]
        dataset = SplitDataset(self.indexed_sentences[indices],
                               et_features,
                               self.targets[indices])
        return DataLoader(dataset, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True)


class SplitDataset(Dataset):
    def __init__(self, indexed_sentences, et_features, targets):
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
        # store sentences here. each sentence has length N = number of words.
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
