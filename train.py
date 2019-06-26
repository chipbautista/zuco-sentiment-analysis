import sys

from torch import softmax
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

from model import BaseModel, init_word_embedding_from_word2vec
from data import SentimentDataSet
from settings import *

"""
For arg parse

word_embedding_weights
if None: initialize from GoogleNews
"""


def iterate(dataloader, mode):
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []
    for i, (indices, et_features, targets) in enumerate(train_loader):
        logits = model(indices, et_features)
        # TO-DO: Add L2 and L1 loss!! Check out Hollenstein's code.
        loss = XE_loss(logits, targets)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_targets.extend(targets)
        all_predictions.extend(softmax(logits, dim=1).argmax(dim=1))
        epoch_loss += loss.item()
    return epoch_loss, accuracy_score(all_targets, all_predictions)


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

initial_word_embedding = init_word_embedding_from_word2vec(dataset.vocabulary)
XE_loss = CrossEntropyLoss()

print('\n> Starting 10-fold CV.')
for k in range(10):
    train_loader, test_loader = dataset.get_train_test_split()
    model = BaseModel(LSTM_UNITS, dataset.max_sentence_length,
                      dataset.num_classes, initial_word_embedding)
    optimizer = SGD(model.parameters(), lr=INITIAL_LR)
    accuracies = []
    for e in range(150):
        train_loss, train_accuracy = iterate(train_loader, 'train')
        # train_loss = 0.0
        # test_
        # all_targets = []
        # all_predictions = []
        # for i, (indices, et_features, targets) in enumerate(train_loader):
        #     optimizer.zero_grad()
        #     logits = model(indices, et_features)
        #     loss = XE_loss(logits, targets)
        #     loss.backward()
        #     optimizer.step()

        #     all_targets.extend(targets)
        #     all_predictions.extend(softmax(logits, dim=1).argmax(dim=1))
        #     train_loss += loss.item()
    test_loss, test_accuracy = iterate(test_loader, 'test')
    accuracies.append(test_accuracy)
    print('Fold', k, 'Train Loss:', train_loss,
          'Train Acc:', train_accuracy, 'Test Loss:', test_loss,
          'Test Acc:', test_accuracy)
