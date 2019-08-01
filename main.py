import os
import re
import regex
import sys
import time
import csv
import collections
from collections import Counter, deque
from emoji import UNICODE_EMOJI
from tqdm import tqdm
import pandas as pd

import argparse
import logging as log
from utils import config

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

from model import TagSpace
from preprocess import *

# 0. Basics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.basicConfig(format='%(asctime)s: %(message)s',
	datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

def mkdir(path):
	os.makedirs(path, exist_ok=True)

def handle_arguments(cl_arguments):
	parser = argparse.ArgumentParser(description='')
    # Configuration files
	parser.add_argument('--config_file', '-c', type=str, nargs="+",
		help="Config file(s) (.conf) for model parameters.")
	parser.add_argument('--overrides', '-o', type=str, default=None,
    	help="Parameter overrides, as valid HOCON string.")
	parser.add_argument('--remote_log', '-r', action="store_true",
    	help="If true, enable remote logging on GCP.")

	return parser.parse_args(cl_arguments)

# I. Preprocess input txt data
PADDING = "<PAD>"
UNKNOWN = "<UNK>"
END_SENTENCE = "<END>"
COMMA = "<COM>"

## 1. prepare job and souce data
cl_args = handle_arguments(sys.argv[1:])
args = config.params_from_file(cl_args.config_file, cl_args.overrides)
data_path = args.data_path
load_preprocess = args.load_preprocess
proj_path = args.proj_path ### where one preprocessed data directory is attached to one proj_path, if available
exp_number = str(args.exp_number)
run_number = str(args.run_number)
log_path = args.local_log_path
mkdir(os.path.join(proj_path, exp_number, run_number))


## 2. data cleaning parameters
max_seq_length = 20
max_word_freq = 10**10 ###

preprocess_path = os.path.join(proj_path, 'preprocess_data')
mkdir(preprocess_path)
if load_preprocess:
	try:
		hashtag_pool = torch.load(os.path.join(preprocess_path, 'hashtag_pool.pt'))
		log.info('hashtag_pool loaded')
		word2idx = torch.load(os.path.join(preprocess_path, 'word2idx.pt'))
		log.info('vocab dictionary loaded')
		vocab_size = len(word2idx)
		log.info('%i words found'%(vocab_size))
		with open(os.path.join(preprocess_path, 'trainable_tweets.csv'), 'r') as f:
			trainable_tweets = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
		log.info('trainable_tweets loaded')
	except:
		load_preprocess = False
		log.info('Preprocessed data directory not found. Do the preprocess from scratch.')

if not load_preprocess:
	log.info('Start data preprocessing...')
	raw_tweets = load_txt(data_path)
	start = time.time()
	word_counter, trainable_tweets = clean_tweets(raw_tweets)
	end = time.time()
	log.info('Cleaned tweets in %f min'%((end-start)/60))
	
	start = time.time()
	word2idx, vocab_size, hashtag_pool = build_dict(word_counter)
	end = time.time()
	log.info('Indexed vocab in %f min'%((end-start)/60))
	
	# start = time.time()
	# tweet2seq(word2idx, trainable_tweets, max_seq_length)
	# end = time.time()
	# log.info('Sequenced tweets in %f min'%((end-start)/60))
	
	log.info('Writing preprocessed data into file...')
	# write trainable_tweets, hashtag_pool, word2idx
	keys = trainable_tweets[0].keys()
	with open(os.path.join(preprocess_path, 'trainable_tweets.csv'), 'w+') as csvfile:
		dict_writer = csv.DictWriter(csvfile, keys)
		dict_writer.writeheader()
		dict_writer.writerows(trainable_tweets)
	df = pd.DataFrame(trainable_tweets)
	df.to_csv(os.path.join(preprocess_path, 'trainable_tweets.csv'))
	torch.save(hashtag_pool, os.path.join(preprocess_path, 'hashtag_pool.pt'))
	torch.save(word2idx, os.path.join(preprocess_path, 'word2idx.pt'))
log.info('Preprocessing finished.')

# II. Train TagSpace
## 1. model parameters
load_saved_model = args.load_saved_model
do_pretrain = args.do_pretrain
batch_size = args.batch_size
num_epochs = args.num_epochs
embedding_dim = args.embedding_dim
window_size = args.window_size
hidden_size = args.hidden_size
margin = args.margin
max_iter = args.max_iter
freeze_layer = args.freeze_layer
retrieve_k = args.retrieve_k
learning_rate = args.learning_rate

output_path = os.path.join(proj_path, exp_number, run_number, 'output')
mkdir(output_path)

## 2. split train/dev/test
tagspace_source = list(filter(lambda x: len(x.get('hashtag')) > 0, trainable_tweets)) # tagspace model only input tweets with at least one hashtag:
# train:dev:test = 7:1:2
split = ['train', 'dev', 'test']
for item in tagspace_source:
	item['split'] = split[torch.multinomial(torch.Tensor([7, 2, 1]), 1)] # train:dev:test = 7:2:1
train = list(filter(lambda x: x['split'] == 'train', tagspace_source))
dev = list(filter(lambda x: x['split'] == 'dev', tagspace_source))
test = list(filter(lambda x: x['split'] == 'test', tagspace_source))

## 3. batch generator
def data_iter(source, batch_size):
	"""
	Randomly gets samples for a batch
	-Args:
		@source: list in the form [{'tweet': string, 'hashtag': string, 'tweet_seq': LongTensor, 
		'hash_seq': list, 'split': 'train'/'dev'/'test']
		@batch_size: int
	-Returns:
		None, but builds a generator that gives a subset of source
	"""
	dataset_size = len(source)
	start = -1 * batch_size
	order = list(range(dataset_size))
	random.shuffle(order)

	while True:
		start += batch_size
		if start > dataset_size - batch_size:
			# Start another epoch.
			start = 0
			random.shuffle(order)   
		batch_indices = order[start:start + batch_size]
		batch = [source[index] for index in batch_indices]
#		 log.info('data_iter func')
		yield [source[index] for index in batch_indices]
		
def eval_iter(source, batch_size):
	"""
	A iterator for evaluation, it gives a list of batches to be iterate through
	-Args:
		@source (during evaluation)
		@batch_size: int
	-Returns:
		@batches: generator giving list of batches
	"""
	batches = []
	dataset_size = len(source)
	start = -1 * batch_size
	order = list(range(dataset_size))
	random.shuffle(order)

	while start < dataset_size - batch_size:
		start += batch_size
		batch_indices = order[start:start + batch_size]
		batch = [source[index] for index in batch_indices]
		if len(batch) == batch_size:
			batches.append(batch)
#			 log.info('eval_iter append')
		else:
#			 log.info('eval_iter continue')
			continue	
	return batches

def get_batch(batch):
	"""
	Get relevant field for each batch
	-Args:
		
	-Returns:
		@vectors: LongTensor, stacked index sequence for tweets in the batch
		@labels: list, hashtag index sequence for tweets in the batch
		@pos_list: Long Tensor, stacked positive example for tweets in the batch
	"""
	vectors = []
	labels = []
	pos_list = []
	# batch_dict = trainable_tweets[np.asarray(batch)]
	tweet2seq(word2idx, batch, max_seq_length)
	for dict in batch:
		vectors.append(dict['tweet_seq'])
		labels.append(dict['hashtag_seq'])
		labels_temp = dict['hashtag_seq']
		pos_list.append(torch.LongTensor(labels_temp)[torch.multinomial(torch.ones(len(labels_temp)), 1)])
		# labels.append(dict["hashtag_seq"])
	# del batch_dict
	return torch.stack(vectors).squeeze().to(dtype=torch.long, device=device), labels, torch.stack(pos_list).squeeze().to(dtype=torch.long, device=device)

## 4. evaluation metric
def evaluate(model, data_iter, k, verbose = False):
	model.eval()
	retrieve_k = []
	for i in range(len(data_iter)):
		# print(i)
		rk_temp = 0
		vectors, labels = get_batch(data_iter[i])[: 2]
		
		tweet_emb = model(vectors).to(device)
		hashtag_pool_emb = model.embed_hashtag_pool(hashtag_pool).to(device)

		retrieve_idx = torch.topk(torch.matmul(tweet_emb, hashtag_pool_emb.transpose(0, 1)), k)[1]
		retrieve_hashtag = np.asarray(hashtag_pool)[np.asarray(retrieve_idx.cpu())]
		for j in range(len(vectors)):
#			 log.info(set(labels[j]))
#			 log.info(set(retrieve_hashtag[j].tolist()))
			rk_temp += len(set(labels[j]).intersection(set(retrieve_hashtag[j].tolist())))/k
		retrieve_k.append(rk_temp/len(vectors))
	return sum(retrieve_k)/len(retrieve_k)

## 5. training loop
# logger = SummaryWriter(log_path)
def training_loop(batch_size, num_epochs, 
						model, optimizer, 
						training_iter, train_eval_iter, dev_iter, retrieve_k,
						verbose=True):
	epoch = 0
	step = 0
	total_batches = int(len(train) / batch_size) #remember to modify later
#	 log.info('total_batches: %i'%(total_batches))
	rk = []
	pbar = tqdm(total=num_epochs)
	while epoch <= num_epochs:
		# print('Training...')
		model.train()
		model.zero_grad()
		data_tensor, labels, pos_list = get_batch(next(training_iter))
		
		tweets_emb = model(data_tensor)
		loss = model.WARPLoss(tweets_emb, pos_list, hashtag_pool)
		loss.backward()
		# print('Backward done.')
		torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
		optimizer.step()
		# print('Optim done.')
		# logger.add_scalar('Loss', loss)
		
		if step % total_batches == 0:
			model.eval()
			# print('Evaluating...')
			if epoch%1 == 0:
				train_rk = evaluate(model, train_eval_iter, retrieve_k)
				# print('train rk done')
				eval_rk = evaluate(model, dev_iter, retrieve_k)
				# print('eval rk done')
				rk.append(eval_rk)
				# info = {
						# 'Loss': loss,
						# 'eval_R@%i'%(retrieve_k): eval_rk,
						# 'train_R@%i'%(retrieve_k): train_rk
					# }
				# for tag, value in info.items():
				# 	logger.add_scalar(tag, value, step)
				if verbose:
					log.info("Epoch %i; Step %i; Loss %f; eval_R@%i: %f; train_R@%i: %f" 
						  %(epoch, step, loss, retrieve_k, eval_rk, retrieve_k, train_rk))
			epoch += 1
			pbar.update(1)
		step += 1
	pbar.close()

	return max(rk)

## 6. start training
training_iter = data_iter(train, batch_size)
train_eval_iter = eval_iter(train[:500], batch_size)
dev_iter = eval_iter(dev, batch_size)
######## deal with pretraining
log.info('Start training model...')
tagspace = TagSpace(batch_size, embedding_dim, vocab_size, max_seq_length, margin, max_iter, window_size, hidden_size).to(device)
optimizer = torch.optim.Adam(tagspace.parameters(), lr=learning_rate)
training_loop(batch_size, num_epochs, tagspace, optimizer, training_iter, train_eval_iter, dev_iter, retrieve_k)
log.info('Training finished.')

# III. job output management
model_path = os.path.join(output_path, 'tagspace.pt')
torch.save(tagspace, model_path)
log.info('Model saved in the run direcotry.')


