import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors

from settings import *


torch.manual_seed(111)
np.random.seed(111)


class BaseModel(torch.nn.Module):
    def __init__(self, lstm_units, max_sentence_length,
                 num_classes, initial_word_embedding, use_gaze):
        super(BaseModel, self).__init__()

        self.use_gaze = use_gaze
        if self.use_gaze:
            et_feature_dim = ET_FEATURE_DIM
        else:
            et_feature_dim = 0

        self.lstm_units = lstm_units
        self.max_sentence_length = max_sentence_length

        self.word_embedding = torch.nn.Embedding.from_pretrained(
            initial_word_embedding, freeze=False)

        self.bi_lstm = torch.nn.LSTM(input_size=WORD_EMBED_DIM + et_feature_dim,
                                     hidden_size=self.lstm_units,
                                     bidirectional=True,
                                     batch_first=True)
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.att_l1 = torch.nn.Linear(in_features=self.lstm_units * 2,
                                      out_features=self.max_sentence_length)
        self.att_l2 = torch.nn.Linear(in_features=self.max_sentence_length,
                                      out_features=2)
        self.hidden_l1 = torch.nn.Linear(in_features=self.lstm_units * 4,
                                         out_features=HIDDEN_LAYER_UNITS)
        self.out = torch.nn.Linear(in_features=HIDDEN_LAYER_UNITS,
                                   out_features=num_classes)

    def forward(self, indices, et_features):
        batch_size = indices.shape[0]
        word_embeddings = self.word_embedding(indices)

        if self.use_gaze:
            x = torch.cat((word_embeddings, et_features), dim=2)
        else:
            x = word_embeddings

        bi_lstm_out, (h_n, c_n) = self.bi_lstm(x)
        bi_lstm_out = self.dropout(bi_lstm_out)

        att_l1_out = self.att_l1(bi_lstm_out)
        att_l1_out_ = torch.tanh(att_l1_out)
        att_l2_out = self.att_l2(att_l1_out_).transpose(1, 2)
        att_weights = F.softmax(att_l2_out, dim=2)

        lstm_embedding = torch.matmul(att_weights, bi_lstm_out)
        lstm_embedding = lstm_embedding.reshape(batch_size, -1)
        hidden_l1_out = self.hidden_l1(lstm_embedding)
        logits = self.out(hidden_l1_out)
        return logits


def init_word_embedding_from_word2vec(vocabulary):
    print('\n> Loading pre-trained word2vec from', WORD_EMBED_MODEL_DIR)
    pretrained_w2v = KeyedVectors.load_word2vec_format(
        WORD_EMBED_MODEL_DIR, binary=True)
    print('> Done. Will now extract embeddings for needed words.')

    embeddings = []
    oov_words = []
    for word in vocabulary:
        try:
            embeddings.append(pretrained_w2v[word])
        except KeyError:
            embeddings.append(np.random.uniform(-0.25, 0.25, WORD_EMBED_DIM))
            oov_words.append(word)

    print('>', len(oov_words), 'words were not found in the pre-trained model.\n')
    return torch.Tensor(embeddings)

# class Model_B(object):
# 	pass
