#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2018-05-03 16:46:15
'''
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def accuracy_score(true_data,pred_data,true_length = None):
	true_data = np.array(true_data)
	pred_data = np.array(pred_data)

	assert true_data.shape == pred_data.shape

	if true_length is not None:
		val_num = np.sum(true_length)
		assert val_num != 0

		res = 0
		for i in range(true_data.shape[0]):
			res += np.sum(true_data[i][:true_length[i]] == pred_data[i][:true_length[i]])
	else:
		val_num = np.prob(true_data.shape)

		assert val_num != 0
		res = np.sum(true_data == pred_data)

	acc = res * 1.0 / val_num

	return acc

def calculateRes(true_data,pred_data,true_length,ignore_labels):
	'''
	true_data,pred_data:[batch_size,steps]	
	'''
	true_cnt = 0
	pred_cnt = 0
	correct_cnt = 0
	
	true_data = np.array(true_data)
	pred_data = np.array(pred_data)
	
	assert len(true_length) == true_data.shape[0]

	assert true_data.shape[1] >= max(true_length)
	assert pred_data.shape[1] >= max(true_length)

	for i in range(true_data.shape[0]):
		tmp_length = true_length[i]

		tmp_true = true_data[i][:tmp_length]
		tmp_pred = pred_data[i][:tmp_length]

		for j in range(len(tmp_true)):
			true_lable = tmp_true[j]
			pred_lable = tmp_pred[j]

			if true_lable not in ignore_labels:
				true_cnt += 1

			if pred_lable not in ignore_labels:
				pred_cnt += 1

				if true_lable == pred_lable:
					correct_cnt += 1
	
	return correct_cnt,pred_cnt,true_cnt
