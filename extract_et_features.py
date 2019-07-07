import scipy.io as scio
import numpy as np

from settings import *


WORD_FEATURES = ['fixPositions', 'nFixations', 'meanPupilSize', 'FFD',
                 'FFD_pupilsize', 'TRT', 'TRT_pupilsize', 'GD', 'GD_pupilsize',
                 'GPT', 'GPT_pupilsize', 'SFD', 'SFD_pupilsize']
sentences = []
TASK_NUM = 1  # SENTIMENT
task_abbrev = '_SR'
max_sentence_length = 0
vocabulary = set([])

for subj in SUBJECTS:
    print('Extracting from subject', subj)
    mat_file = scio.loadmat(MAT_DIR.format(TASK_NUM, subj + task_abbrev))
    for s_num, sentence_data in enumerate(mat_file['sentenceData'][0]):

        # initialize structure
        try:
            words = [w[0] for w in sentence_data['word']['content'][0]]
        except Exception as e:
            print('Sentence #', s_num, 'for subject', subj, 'not available.')
            continue

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
                print('imbalance!')

            for w, v in enumerate(values):
                if not v.all() or v.shape[1] == 0:
                    v = np.NaN
                elif feature == 'fixPositions' and v.shape[1] > 0:
                    v = list(v[0])
                else:
                    v = float(v[0])
                sentences[s_num][w][feature].append(v)

np.save(ET_FEATURES_DIR, sentences, allow_pickle=True)
print('\n> Done. Eye-tracking features saved to:', ET_FEATURES_DIR)
print('> Max words in sentence:', max_sentence_length)
