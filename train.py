import sys

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from model import BaseModel
from data import SentimentDataSet
from settings import *

"""
For arg parse

word_embedding_weights
if None: initialize from GoogleNews
"""
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

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True)
model = BaseModel(LSTM_UNITS, dataset.max_sentence_length,
                  dataset.vocabulary, dataset.num_classes)
optimizer = SGD(model.parameters(), lr=INITIAL_LR)
XE_loss = CrossEntropyLoss()

for e in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for i, (indices, et_features, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        logits = model(indices, et_features)
        loss = XE_loss(logits, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print('Epoch', e, 'loss:', epoch_loss)
