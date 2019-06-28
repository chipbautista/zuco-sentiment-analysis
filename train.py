import sys
from argparse import ArgumentParser

import numpy as np
from torch import softmax
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

from model import BaseModel, init_word_embedding_from_word2vec
from data import SentimentDataSet
from settings import *


def init_metrics():
    return {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }


def print_metrics(metrics, split):
    # just a helper function to cleanly print metrics...
    print('\n[{}]: '.format(split), end='')
    for k, v in metrics.items():
        if type(v) == np.float64:
            print(' {}: {:.2f}'.format(k, v), end='')
        else:
            print(' {}: {:.2f} '.format(k, np.mean(v)), end='')


def get_metrics(targets, predictions):
    return {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average=metric_ave_method),
        'precision': precision_score(targets, predictions, average=metric_ave_method),
        'recall': recall_score(targets, predictions, average=metric_ave_method)
    }


def iterate(dataloader, train=True):
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []
    for i, (indices, et_features, targets) in enumerate(dataloader):
        if USE_CUDA:
            indices = indices.cuda()
            et_features = et_features.cuda()
            targets = targets.cuda()

        logits = model(indices, et_features)
        loss = XE_loss(logits, targets)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer_scheduler.step()

        all_targets.extend(targets.cpu())
        all_predictions.extend(softmax(logits, dim=1).argmax(dim=1).cpu())
        epoch_loss += loss.item()

    return epoch_loss, get_metrics(all_targets, all_predictions)


# Prepping variables for training
parser = ArgumentParser()
parser.add_argument('--num-sentiments', type=int, default=3,
                    help='2: binary classification (broken!), 3: ternary.')
parser.add_argument('--use-gaze', default=True)
args = parser.parse_args()

dataset = SentimentDataSet(args)
lstm_units, halve_lr_every_passes = ((300, 3) if args.num_sentiments == 2
                                     else (150, 9))
if args.num_sentiments == 2:
    lstm_units, halve_lr_every_passes, metric_ave_method = (300, 3, 'binary')
else:
    lstm_units, halve_lr_every_passes, metric_ave_method = (150, 10, 'macro')

# we initialize the word embedding using GoogleNews
initial_word_embedding = init_word_embedding_from_word2vec(dataset.vocabulary)
XE_loss = CrossEntropyLoss()

train_metrics = init_metrics()
test_metrics = init_metrics()

print('--- SETTINGS ---')
print('Number of sentiments to classify:', args.num_sentiments)
print('Learning rate:', INITIAL_LR)
print('Num of epochs per fold:', NUM_EPOCHS)
print('Use gaze features:', args.use_gaze)

print('\n> Starting 10-fold CV.')
for k, (train_loader, test_loader) in enumerate(dataset.split_cross_val(10)):
    # initialize model and optimizer every fold
    model = BaseModel(lstm_units, dataset.max_sentence_length,
                      args.num_sentiments, initial_word_embedding.clone(),
                      args.use_gaze)
    optimizer = SGD(model.parameters(), lr=INITIAL_LR,
                    momentum=0.95, nesterov=True)
    # optimizer = Adam(model.parameters(), 0.001)
    optimizer_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=halve_lr_every_passes, gamma=0.5)
    if USE_CUDA:
        model = model.cuda()

    for e in range(NUM_EPOCHS):
        train_loss, train_results = iterate(train_loader)

    # save the training metrics of last epoch
    for metric, value in train_results.items():
        train_metrics[metric].append(value)

    test_loss, test_results = iterate(test_loader, train=False)
    for metric, value in test_results.items():
        test_metrics[metric].append(value)

    print('\nFold', k, end='')
    print_metrics(train_results, 'TRAIN')
    print_metrics(test_results, 'TEST')

print('\n\n> 10-fold CV done')
print_metrics(train_metrics, 'MEAN TRAIN')
print_metrics(test_metrics, 'MEAN TEST')
