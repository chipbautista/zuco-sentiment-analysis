import scipy.io as scio
import numpy as np

import sys
sys.path.append('.')

from settings import *


L1_KEYS = [
    'content', 'rawData', 'mean_t1', 'mean_t2', 'mean_a1', 'mean_a2',
    'mean_b1', 'mean_b2', 'mean_g1', 'mean_g2', 'mean_t1_sec', 'mean_t2_sec',
    'mean_a1_sec', 'mean_a2_sec', 'mean_b1_sec', 'mean_b2_sec', 'mean_g1_sec',
    'mean_g2_sec', 'mean_t1_diff', 'mean_t2_diff', 'mean_a1_diff',
    'mean_a2_diff', 'mean_b1_diff', 'mean_b2_diff', 'mean_g1_diff',
    'mean_g2_diff', 'mean_t1_diff_sec', 'mean_t2_diff_sec', 'mean_a1_diff_sec',
    'mean_a2_diff_sec', 'mean_b1_diff_sec', 'mean_b2_diff_sec',
    'mean_g1_diff_sec', 'mean_g2_diff_sec', 'word', 'omissionRate',
    'allFixations', 'wordbounds', 'answer_mean_t1', 'answer_mean_t2',
    'answer_mean_a1', 'answer_mean_a2', 'answer_mean_b1', 'answer_mean_b2',
    'answer_mean_g1', 'answer_mean_g2', 'answer_mean_t1_diff',
    'answer_mean_t2_diff', 'answer_mean_a1_diff', 'answer_mean_a2_diff',
    'answer_mean_b1_diff', 'answer_mean_b2_diff', 'answer_mean_g1_diff',
    'answer_mean_g2_diff']

WORD_KEYS = [
    'content', 'fixPositions', 'nFixations', 'meanPupilSize',
    'rawEEG', 'rawET', 'FFD', 'FFD_pupilsize', 'FFD_t1', 'FFD_t2', 'FFD_a1',
    'FFD_a2', 'FFD_b1', 'FFD_b2', 'FFD_g1', 'FFD_g2', 'FFD_t1_diff',
    'FFD_t2_diff', 'FFD_a1_diff', 'FFD_a2_diff', 'FFD_b1_diff', 'FFD_b2_diff',
    'FFD_g1_diff', 'FFD_g2_diff', 'TRT', 'TRT_pupilsize', 'TRT_t1', 'TRT_t2',
    'TRT_a1', 'TRT_a2', 'TRT_b1', 'TRT_b2', 'TRT_g1', 'TRT_g2', 'TRT_t1_diff',
    'TRT_t2_diff', 'TRT_a1_diff', 'TRT_a2_diff', 'TRT_b1_diff', 'TRT_b2_diff',
    'TRT_g1_diff', 'TRT_g2_diff', 'GD', 'GD_pupilsize', 'GD_t1', 'GD_t2',
    'GD_a1', 'GD_a2', 'GD_b1', 'GD_b2', 'GD_g1', 'GD_g2', 'GD_t1_diff',
    'GD_t2_diff', 'GD_a1_diff', 'GD_a2_diff', 'GD_b1_diff', 'GD_b2_diff',
    'GD_g1_diff', 'GD_g2_diff', 'GPT', 'GPT_pupilsize', 'GPT_t1', 'GPT_t2',
    'GPT_a1', 'GPT_a2', 'GPT_b1', 'GPT_b2', 'GPT_g1', 'GPT_g2', 'GPT_t1_diff',
    'GPT_t2_diff', 'GPT_a1_diff', 'GPT_a2_diff', 'GPT_b1_diff', 'GPT_b2_diff',
    'GPT_g1_diff', 'GPT_g2_diff', 'SFD', 'SFD_pupilsize', 'SFD_t1', 'SFD_t2',
    'SFD_a1', 'SFD_a2', 'SFD_b1', 'SFD_b2', 'SFD_g1', 'SFD_g2', 'SFD_t1_diff',
    'SFD_t2_diff', 'SFD_a1_diff', 'SFD_a2_diff', 'SFD_b1_diff', 'SFD_b2_diff',
    'SFD_g1_diff', 'SFD_g2_diff']

WORD_FEATURES = ['fixPositions', 'nFixations', 'meanPupilSize', 'FFD', 'FFD_pupilsize',
                 'TRT', 'TRT_pupilsize', 'GD', 'GD_pupilsize', 'GPT',
                 'GPT_pupilsize', 'SFD', 'SFD_pupilsize']
word_features_indices = [0, 1, 3, 6, 7, 24, 25, 42, 43, 60, 61, 78, 79]
sentences = []

TASK_NUM = 1  # SENTIMENT
task_abbrev = '_SR'
max_sentence_length = 0
for subj in SUBJECTS:
    print('Extracting from subject', subj)
    mat_file = scio.loadmat(MAT_DIR.format(TASK_NUM, subj + task_abbrev))
    for s_num, sentence_data in enumerate(mat_file['sentenceData'][0]):

        # initialize structure
        words = [w[0] for w in sentence_data['word']['content'][0]]

        if (s_num + 1) > len(sentences):
            sentences.append([])
            sentences[s_num] = [{k: [] for k in WORD_FEATURES}
                                for w in words]

        sentence_length = len(words)
        if sentence_length > max_sentence_length:
            max_sentence_length = sentence_length

        for feature in WORD_FEATURES:
            try:
                values = sentence_data['word'][feature][0]
            except ValueError as e:
                print('Subject', subj, 'sentence number', s_num,
                      'has no word-level feature: ', feature)
                values = np.array([[None]] * len(words))

            if len(words) != len(values):
                print('imbalan')

            for w, v in enumerate(values):
                if not v.all() or v.shape[1] == 0:
                    v = np.NaN
                elif feature == 'fixPositions' and v.shape[1] > 0:
                    v = list(v[0])
                else:
                    v = float(v[0])
                sentences[s_num][w][feature].append(v)

np.save(ET_FEATURES, sentences, allow_pickle=True)
print('Done. Saved to:', ET_FEATURES)
print('Max words in sentence:', max_sentence_length)
