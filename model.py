import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TagSpace(nn.Module):
	def __init__(self, batch_size=32, embedding_dim=64, 
				 vocab_size=10**5, max_seq_length=20, 
				 margin=0.1, max_iter=1000, 
				 window_size=5, hidden_size=1000):
		super(TagSpace, self).__init__()
		
		self.vocab_size = vocab_size # N in paper
		self.embed = nn.Embedding(vocab_size, embedding_dim, max_norm=1) # or use w2v/wsabie as pretrained lookup
		self.embedding_size = embedding_dim # d in paper
		self.max_seq_length = max_seq_length # l in paper
		self.margin = margin
		self.max_iter = max_iter
		self.window_size = window_size # k in paper
		self.hidden_size = hidden_size # H in paper
		self.batch_size = batch_size
		
		self.conv = nn.Conv1d(in_channels = self.embedding_size, 
						 out_channels = self.hidden_size, 
						 kernel_size = self.window_size, 
						 padding = 1)
		self.maxpool = nn.MaxPool1d(kernel_size = self.max_seq_length)
		self.decoder = nn.Linear(in_features = self.hidden_size, 
								out_features = self.embedding_size, 
								bias=True)
		
	def forward(self, input):
	# 1. get embed
		post_embed = self.embed(input)
#		 print('Size after emebedding:', post_embed.size())
	# 2. convolution + tanh activation
		post_conv = torch.tanh(self.conv(post_embed.permute(0, 2, 1)))
#		 print('Size after convolution layer:', post_conv.size())
	# 3. maxpool + tanh activation
		post_maxpool = torch.tanh(self.maxpool(post_conv).reshape(self.batch_size, self.hidden_size))
#		 print('Size after max pooling:', post_maxpool.size())
	# 4. linear decoder
		tweets_embed = self.decoder(post_maxpool)
#		 print('Size of output:', tweets_embed.size())
		return tweets_embed
		
	
	def WARPLoss(self, tweets_emb, pos_list, hashtag_pool):
		"""
		Compute the averaged WARP loss of a batch of tweets
		-Args:
			@tweets_emb: tensor, 
			@pos_list: list, list of index lists of hashtags found in the tweet batch
			@hashtag_pool: list of hashtags (index) in the training corpus
			@max_iter: max number of iteration to compare with the score of the positive pair
			@margin: int, the margin (default 0.1 as in the paper)
		-Returns:
			@warp_loss: calculated warp loss for each example in the batch
		"""
		# generate positive pairs:
		pos_embs = torch.zeros(self.batch_size, self.embedding_size)
		for i, hashtag_list in enumerate(pos_list):
			pos_embs[i] = self.embed(torch.LongTensor(hashtag_list)[
				torch.multinomial(torch.ones(len(hashtag_list)), 1)])
		pos_scr = torch.bmm(torch.unsqueeze(tweets_emb, 1), torch.unsqueeze(pos_embs, 2)).squeeze()
#		 print('Positive pairs generated.')
		
		# generate negative pairs:
		neg_embs = torch.zeros(self.batch_size, self.embedding_size)
		for b in range(self.batch_size): # loop over tweets in the batch
			neg_samples = torch.LongTensor(hashtag_pool)[
				torch.multinomial(torch.ones(len(hashtag_pool)), self.max_iter)]
			for i, neg_sample in enumerate(neg_samples): # neg_sample is an tensor of one index
				neg_emb = self.embed(neg_sample)
				if torch.dot(tweets_emb[b], neg_emb) >= pos_scr[b] or i == self.max_iter-1:
					neg_embs[b] = neg_emb
#		 print('Negative pairs generated.')
					
		# calculate WARP loss
		neg_scr = torch.bmm(torch.unsqueeze(tweets_emb, 1), 
						 torch.unsqueeze(neg_embs, 2)).squeeze()
		margin_stacked = torch.ones(self.batch_size)*self.margin
		warp_loss = torch.mean(torch.max(torch.zeros(self.batch_size), margin_stacked-pos_scr + neg_scr))
#		 print('Calculated WARP loss:', warp_loss)
		return warp_loss

	def embed_hashtag_pool(self, hashtag_pool):
		hashtag_pool_tensor = torch.Tensor(len(hashtag_pool), self.embedding_size)
		for i in range(len(hashtag_pool)):
			hashtag_pool_tensor[i] = self.embed(hashtag_pool[i])
		return hashtag_pool_tensor