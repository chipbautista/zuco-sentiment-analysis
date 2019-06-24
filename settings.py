
# Model configuration, following Hollenstein
NUM_EPOCHS = 10
BATCH_SIZE = 32
INITIAL_LR = 1e-3
DROPOUT = 0.5

WORD_EMBED_DIM = 300
MAX_SENTENCE_LENGTH = 43

# Directories
DATASET_DIR = 'data/sentences/'
WORD_EMBED_MODEL_DIR = 'models/word_embedding/GoogleNews-vectors-negative300.bin'
ET_FEATURES = 'data/task1_sentence_features.npy'
MAT_DIR = '../ZuCo/task{}/Matlab files/results{}.mat'

# General info
SUBJECTS = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH',
            'ZKW', 'ZMG', 'ZPH']  # exclude ZDN for now

ET_FEATURES_TO_USE = ['nFixations', 'FFD', 'TRT', 'GD', 'GPT']
