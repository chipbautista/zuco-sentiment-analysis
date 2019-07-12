# zuco-sentiment-analysis

A PyTorch implementation of the sentiment analysis model from https://arxiv.org/abs/1904.02682 by Hollenstein et al., but with some deviations.

In current settings, ternary classification with gaze features reach 0.61 F1, removing the gaze features reach 0.54 F1 (though this might still benefit from parameter search).

Binary classification is currently broken.
