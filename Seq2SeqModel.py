#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2018-05-03 10:56:25
'''
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import sys,os
from tensorflow.contrib.rnn import LSTMCell,DropoutWrapper


class Seq2SeqModel(object):
	def __init__(self,
			input_steps,
			embedding_size,
			hidden_size,
			vocab_size,
			intent_size,
			slot_size,
			SOS_slot_index,
			PAD_slot_index,
			output_keep_prob = 0.5,
			batch_size = 16,
			n_layers = 1):
		self.input_steps = input_steps
		self.embedding_size  = embedding_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.slot_size = slot_size
		self.SOS_slot_index = SOS_slot_index
		self.PAD_slot_index = PAD_slot_index
		self.intent_size = intent_size
		self.output_keep_prob = output_keep_prob
		self.batch_size = batch_size
		self.n_layers = n_layers

		self.encoder_inputs = tf.placeholder(tf.int32,shape = [input_steps,batch_size],
				name = 'encoder_inputs')

		self.encoder_inputs_actual_length = tf.placeholder(tf.int32,[batch_size],
				name = 'encoder_inputs_actual_length')

		self.decoder_targets = tf.placeholder(tf.int32,[batch_size,input_steps],
				name = 'decoder_targets')
		
		self.intent_targets = tf.placeholder(tf.int32,[batch_size],name = 'intent_targets')

	def build(self):
		'''
		1. embedding: word_embeddings and slot_embeddings
		2. encoder
		3. decoder	
		'''
		with tf.name_scope('embedding'):
			self.word_embeddings = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],
						-1.0,0.1),dtype = tf.float32,name = 'word_embeddings')

			self.slot_embeddings = tf.Variable(tf.random_uniform([self.slot_size,self.embedding_size],
						-1.0,0.1),dtype = tf.float32,name = 'slot_embeddings')

			self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.word_embeddings,self.encoder_inputs)

			
		#encode
		encoder_fw_cell = DropoutWrapper(LSTMCell(self.hidden_size),output_keep_prob = self.output_keep_prob)
		encoder_bw_cell = DropoutWrapper(LSTMCell(self.hidden_size),output_keep_prob = self.output_keep_prob)

		#use a bi-lstm to encode
		(encoder_fw_outputs,encoder_bw_outpus),(encoder_fw_final_status,encoder_bw_final_status) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw = encoder_fw_cell,
				cell_bw = encoder_bw_cell,
				inputs = self.encoder_inputs_embedded,
				sequence_length = self.encoder_inputs_actual_length,
				dtype = tf.float32,
				time_major = True)

		self.encoder_outputs = tf.concat([encoder_fw_outputs,encoder_bw_outpus],axis = 2)

		self.encoder_final_status_c = tf.concat([encoder_fw_final_status.c,encoder_bw_final_status.c],axis = 1)

		self.encoder_final_status_h = tf.concat([encoder_fw_final_status.h,encoder_bw_final_status.h],axis = 1)

		print('encoder_outputs:',self.encoder_outputs)
		print('encoder_final_status_c:',self.encoder_final_status_c)
		print('encoder_final_status_h:',self.encoder_final_status_h)

		#decoder
		self.decoder_length = self.encoder_inputs_actual_length


		self.intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2,self.intent_size],-0.1,0.1),
				dtype = tf.float32,name = 'intent_W')
		self.intent_b = tf.Variable(tf.zeros([self.intent_size]),dtype = tf.float32,name = 'intent_b')

		intent_use_attention = True
		if intent_use_attention:
			self.intent_context_weight = tf.Variable(tf.truncated_normal([2*self.hidden_size]),name = 'intent_context_weight')

			#对Bi-LSTM的输出进行一次全连接非线性转换编码
			intent_fc = tf.contrib.layers.fully_connected(tf.transpose(self.encoder_outputs,[1,0,2]),self.hidden_size * 2,activation_fn=tf.nn.tanh)

			print('intent_fc:',intent_fc)
			
			multiply = tf.multiply(intent_fc,self.intent_context_weight)

			reduce_sum = tf.reduce_sum(multiply,axis = 2,keep_dims = True)
			print('reduce_sum:',reduce_sum)

			alpha = tf.nn.softmax(reduce_sum,dim = 1)
			print('alpha:',alpha)

			atten_output = tf.reduce_sum(tf.multiply(tf.transpose(self.encoder_outputs,[1,0,2]),alpha),axis = 1)

			print('atten_output:',atten_output)

			intent_logits = tf.add(tf.matmul(atten_output,self.intent_W),self.intent_b)
			print('att_intent_logits:',intent_logits)
		else:
			intent_logits = tf.add(tf.matmul(self.encoder_final_status_h,self.intent_W),self.intent_b)

		self.intent = tf.argmax(intent_logits,axis = 1)

		intent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = intent_logits,
				labels = tf.one_hot(self.intent_targets,self.intent_size,dtype = tf.float32)))

		sos_time_slice = tf.fill([self.batch_size],self.SOS_slot_index)
		sos_step_embedded = tf.nn.embedding_lookup(self.slot_embeddings,sos_time_slice)
		
		#对于pad的位置，全部填充0
		pad_step_embedded = tf.zeros([self.batch_size,self.hidden_size*2 + self.embedding_size],dtype = tf.float32)

		#Custerm Helper
		def initial_fn():
			initial_finished = (0 > self.decoder_length)
			#每个step的输入为：o_{i-1} + h_i
			initial_input = tf.concat((sos_step_embedded,self.encoder_outputs[0]),1)

			return initial_finished,initial_input

		def sample_fn(time,outputs,state):
			prediction_id = tf.to_int32(tf.argmax(outputs,axis = 1))

			return prediction_id

		def next_inputs_fn(time,outputs,state,sample_ids):
			'''
			根据当前step的输出、状态、输出类别得到下一时刻的输入
			'''
			pred_embedded = tf.nn.embedding_lookup(self.slot_embeddings,sample_ids)

			next_input = tf.concat((pred_embedded,self.encoder_outputs[time]),1)

			element_finished = time > self.decoder_length
			all_finished = tf.reduce_all(element_finished)

			next_inputs = tf.cond(all_finished,lambda : pad_step_embedded,lambda: next_input)

			next_state = state

			return element_finished,next_inputs,next_state

		my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn,sample_fn,next_inputs_fn)

		
		def attention_decode(my_helper,name_scope,reuse = None):
			with tf.variable_scope(name_scope,reuse = reuse):
				attention_memory = tf.transpose(self.encoder_outputs,[1,0,2])
				attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
						num_units = self.hidden_size,
						memory = attention_memory,
						memory_sequence_length = self.encoder_inputs_actual_length)
				cell = LSTMCell(num_units = self.hidden_size * 2)
				
				attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell,
						attention_mechanism,attention_layer_size = self.hidden_size)

				out_cell = tf.contrib.rnn.OutputProjectionWrapper(attention_cell,self.slot_size,reuse = reuse)
				
				decoder = tf.contrib.seq2seq.BasicDecoder(cell = out_cell,
						helper = my_helper,
						initial_state = out_cell.zero_state(dtype = tf.float32,batch_size = self.batch_size))
				
				final_outputs,final_state,final_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder = decoder,
						output_time_major = True,
						impute_finished = True,
						maximum_iterations = self.input_steps)

				return final_outputs

		decoder_outputs = attention_decode(my_helper,'decode')

		print('decoder_outputs:',decoder_outputs)
		print('decoder_outputs.rnn_outputs:',decoder_outputs.rnn_output)
		print('decoder_outputs.sample_id:',decoder_outputs.sample_id)

		#decoder_prediction [max_step,batch_size]->[batch_size,max_step]
		self.decoder_prediction = tf.transpose(decoder_outputs.sample_id,[1,0])

		decoder_max_steps,decoder_batch_size,decoder_dim = tf.unstack(tf.shape(decoder_outputs.rnn_output))
		
		print('decoder_max_steps:',decoder_max_steps)
		print('decoder_batch_size:',decoder_batch_size)
		print('decoder_dim:',decoder_dim)

		self.decoder_targets_time_majored = tf.transpose(self.decoder_targets,[1,0])
		self.decoder_targets_with_true_length = self.decoder_targets_time_majored[:decoder_max_steps]

		print('decoder_targets_with_true_length:',self.decoder_targets_with_true_length)

		#标记PAD的位置，计算loss时不考虑
		self.mask = tf.to_float(tf.not_equal(self.decoder_targets_with_true_length,
					self.PAD_slot_index))

		seq_loss = tf.contrib.seq2seq.sequence_loss(logits = decoder_outputs.rnn_output,
				targets = self.decoder_targets_with_true_length,
				weights = self.mask)
		slot_loss = tf.reduce_mean(seq_loss)
		self.loss = intent_loss + slot_loss

		#train
		optimizer = tf.train.AdamOptimizer(name = 'adam_opt')
		self.grads,self.vars = zip(*optimizer.compute_gradients(self.loss))

		self.gradients,_ = tf.clip_by_global_norm(self.grads,5)
		
		self.train_op = optimizer.apply_gradients(zip(self.gradients,self.vars))

	def step(self,sess,mode,train_bacth):
		if mode not in ['train','test']:
			print('mode error:',mode)
			sys.exit(1)

		unziped = list(zip(*train_bacth))

		if mode == 'train':
			output_feeds = [self.train_op,self.loss,self.decoder_prediction,
				self.intent,self.mask]

			feed_dict = {self.encoder_inputs : np.transpose(unziped[0],[1,0]),
				self.encoder_inputs_actual_length : unziped[1],
				self.decoder_targets : unziped[2],
				self.intent_targets : unziped[3]}

		else:
			output_feeds = [self.decoder_prediction,self.intent]
			feed_dict = {self.encoder_inputs : np.transpose(unziped[0],[1,0]),
				self.encoder_inputs_actual_length:unziped[1]}

		results = sess.run(output_feeds,feed_dict = feed_dict)

		return results


				

