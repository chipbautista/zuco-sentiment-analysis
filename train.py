import sys

from torch.utils.data import DataLoader

from data import SentimentDataSet
from settings import *


if len(sys.argv) == 1 or sys.argv[1] == 'b':
    # binary classification
    LSTM_UNITS = 300
    HALVE_LR_EVERY_PASSES = 3
    dataset = SentimentDataSet('binary')
else:
    # ternary classification
    LSTM_UNITS = 150
    HALVE_LR_EVERY_PASSES = 9
    dataset = SentimentDataSet('ternary')

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
for i, (sentences, targets) in enumerate(data_loader):
    import pdb; pdb.set_trace()
    words = x.split()

    # get word embedding, probs gensim is most convenient
    # get gaze features. Need to extract them from matlab files first,

    # then save in a format where it's convenient to query from.
