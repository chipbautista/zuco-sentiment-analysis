import torch.nn as nn
import torch.nn.functional as F


class BaseModel(object):
	def __init__(self):
		self.bi_lstm = nn.LSTM(bidirectional=True, batch_first=True)
		self.attention_l1 = nn.Linear(LSTM_UNITS * 2, 40, bias=False)
		self.attention_l2 = nn.Linear(40, 2, bias=False)

	def forward(self, x):
		bi_lstm_out, h_n, c_n = self.bi_lstm(x)
		attention_l1_out = self.att_l1(h_n)
		attention_l2_out = self.att_l2(attention_l1_out)
		attention = F.softmax(attention_l2_out, dim=1)


class Model_B(object):
	pass
