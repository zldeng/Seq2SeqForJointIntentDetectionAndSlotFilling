#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2018-05-03 10:00:06
'''


import numpy as np
import random


def dataPipeline(ori_data,max_length = 50):
	res_data = []
	for line in ori_data:
		line_list = line.strip().split('\t')

		sentence = line_list[0].strip().split(' ')[1:-1]
		slot_list = line_list[1].strip().split(' ')[1:-1]
		intent = line_list[1].strip().split(' ')[-1]
		
		assert len(sentence) == len(slot_list)

		sen_len = len(sentence)

		if sen_len < max_length:
			pad_list = ['EOS'] + ['PAD'] * (max_length - sen_len - 1)
			sentence.extend(pad_list)
			
			slot_list.extend(pad_list)
			
		else:
			sentence = sentence[:max_length]
			sentence[-1] = 'EOS'

			slot_list = slot_list[:max_length]
			slot_list[-1] = 'EOS'

		res_data.append((sentence,slot_list,intent))

	return res_data


def mapKey2Index(pipeline_res_data):
	word2index = {'SOS':0,'EOS':1,'PAD':2,'UNK':3}
	slot2index = {'SOS':0,'EOS':1,'PAD':2,'UNK':3,'O':4}
	intent2index = {'UNK':0}

	for sentence,slot_list,intent in pipeline_res_data:
		for word in sentence:
			if word not in word2index:
				word2index[word] = len(word2index)
		
		for slot in slot_list:
			if slot not in slot2index:
				slot2index[slot] = len(slot2index)

		if intent not in intent2index:
			intent2index[intent] = len(intent2index)

	index2word = {v:k for k,v in word2index.items()}
	index2slot = {v:k for k,v in slot2index.items()}
	index2intent = {v:k for k,v in intent2index.items()}

	return word2index,slot2index,intent2index,index2word,index2slot,index2intent

def data2Index(pipeline_res_data,word2index,slot2index,intent2index):
	indexed_data = []
	for sentence,slot_list,intent in pipeline_res_data:
		sent_index_list = list(map(lambda word : word2index[word] if word in word2index else word2index['UNK'],sentence))
		
		sent_true_len = sentence.index('EOS')

		slot_index_list = list(map(lambda slot: slot2index[slot] if slot in slot2index else slot2index['UNK'],slot_list))

		intent_index = intent2index[intent] if intent in intent2index else intent2index['UNK']

		indexed_data.append((sent_index_list,sent_true_len,slot_index_list,intent_index))


	return indexed_data


def indexToData(index_list,index2data):
	data_list = []
	for idx in index_list:
		data_list.append(index2data.get(idx,'NULL'))

	return ' '.join(data_list)

def getBatch(batch_size,train_data):
	random.shuffle(train_data)

	sindex = 0
	eindex = batch_size

	while eindex < len(train_data):
		batch = train_data[sindex:eindex]

		sindex = eindex
		eindex = eindex + batch_size

		yield batch



 


