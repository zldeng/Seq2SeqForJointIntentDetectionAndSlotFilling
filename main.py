#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2018-05-03 15:22:25
'''
import tensorflow as tf
import numpy as np
from Seq2SeqModel import Seq2SeqModel
from DataUtil import *
from metrics import *

input_steps = 50
embedding_size = 128
hidden_size = 100
n_layers = 2
batch_size = 16
output_keep_prob = 0.5

epoch_num = 64

train_data_file = './dataset/atis-2.train.w-intent.iob'
test_data_file = './dataset/atis.test.w-intent.iob'

def getModel():

	return model



def main():
	print('Loading data...')
	train_data = open(train_data_file).readlines()

	train_data_pipeline = dataPipeline(train_data,input_steps)


	word2index,slot2index,intent2index,index2word,\
		index2slot,index2intent = mapKey2Index(train_data_pipeline)

	train_indexed_data = data2Index(train_data_pipeline,word2index,slot2index,intent2index)

	test_data = open(test_data_file).readlines()
	test_data_pipeline = dataPipeline(test_data,input_steps)

	test_indexed_data = data2Index(test_data_pipeline,word2index,slot2index,intent2index)

	print('Load data Done!')
	vocab_size = len(word2index)
	slot_size = len(slot2index)
	intent_size = len(intent2index)
	
	print('vocab_size:',vocab_size)
	print('slot_size:',slot_size)
	print('intent_size:',intent_size)

	SOS_slot_index = slot2index['SOS']
	PAD_slot_index = slot2index['PAD']

	model = Seq2SeqModel(input_steps,embedding_size,
			hidden_size,vocab_size,intent_size,
			slot_size,SOS_slot_index,
			PAD_slot_index,output_keep_prob,
			batch_size,n_layers)

	model.build()

	sess = tf.Session()

	sess.run(tf.global_variables_initializer())

	for epoch in range(epoch_num):
		for i,batch in enumerate(getBatch(batch_size,train_indexed_data)):
			_,loss,decoder_prediction,intent,mask = model.step(sess,'train',batch)

			if i % 50 == 0:
				print('train_loss epoch-{} iter-{} loss:{}'.format(epoch,i,loss))

		#slot
		true_cnt_list = []
		pred_cnt_list = []
		correct_cnt_list = []

		intent_total = 0
		intent_correct = 0

		#evaluate after each epoch finished	
		for i,batch in enumerate(getBatch(batch_size,test_indexed_data)):
			decoder_prediction,pred_intent = model.step(sess,'test',batch)
			#print('decoder_prediction:',decoder_prediction.shape)
			#print('pred_intent:',pred_intent.shape)
			
			unziped = list(zip(*batch))
			
			sentence = unziped[0]
			true_len = np.array(unziped[1])
			slot_index_list = np.array(unziped[2])
			intent_index = np.array(unziped[3])
		
			correct_cnt,pred_cnt,true_cnt = calculateRes(slot_index_list,decoder_prediction,
					true_len,set(range(5))) 

			correct_cnt_list.append(correct_cnt)
			pred_cnt_list.append(pred_cnt)
			true_cnt_list.append(true_cnt)
			
			intent_total += intent_index.shape[0]
			intent_correct += np.sum(np.equal(intent_index,pred_intent))
			

		slot_p = sum(correct_cnt_list)/float(sum(pred_cnt_list))
		slot_r = sum(correct_cnt_list) / float(sum(true_cnt_list))
		slot_f = 2 * slot_p * slot_r / (slot_p + slot_r)
		
		intent_acc = intent_correct / float(intent_total)

		print('test epoch-{} slot p:{} r:{} f:{}'.format(epoch,slot_p,slot_r,slot_f))
		print('test epoch-{} intent_acc:{}'.format(epoch,intent_acc))
		

if __name__ == '__main__':
	main()
