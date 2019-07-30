import os
import re
import regex
import time
import csv
import torch
import collections
from collections import Counter, deque
from emoji import UNICODE_EMOJI


data_path = os.path.join(os.getcwd(), 'data') # should be changed later
PADDING = "<PAD>"
UNKNOWN = "<UNK>"

END_SENTENCE = "<END>"
COMMA = "<COM>"

def load_txt(input_txt):
    l = []
    for i, t in enumerate(open(input_txt).readlines()):
        l.append(t)
    return l

def tokenize(text):
    return text.split()

def is_emoji(string):
    return string in UNICODE_EMOJI

def add_space(text):
    # Wrap emoji with space
    return ''.join(' ' + char + ' ' if is_emoji(char) else char for char in text).strip()

def contain_digit(string):
    return any(char.isdigit() for char in string)

def is_hashtag(string):
    return string[0] == '#'  

def deal_special_char(text, remove_digit = True):
    # remove url
    text = re.sub(r"http\S+", '', text)
    # special char
    for char in ['$', '%', '^', '*', '(', ')', '[', ']', '|', "\\", "/", "`", "~", "<", ">", '\"', '&', '=', '+', '-'
                , '__', '...']:
        text = text.replace(char, ' ')
    # deal with period and comma
    for char in [',', ';', ':']:
        text = text.replace(char, ' '+COMMA)
    for char in ['.', '!', '?']:
        text = text.replace(char, ' '+END_SENTENCE + ' ')
    # deal repeated punctuation
    text = text.replace('<END>   <COM>', '<COM>')
    # deal repeated hashtag
    text = text.replace('#', ' #')
    # remove digits:
    if remove_digit is True:
        text = regex.sub(r"#\S+(*SKIP)(*FAIL)|\d+", "", text)
#     # remove empty hashtags:
#     text = text.replace('# ', '')
    return re.sub(r"\s\s+" , " ", text)
            
def clean_tweets(raw_tweets, deal_emoji = True, remove_digit = True,
             keyword_hashtag = False):
    """
    Generate a dictionary that maps a word to an index and vice versa
    -Args:
        @raw_tweets: list, list of raw tweets in input txt
        @deal_emoji: boolean, treat each emoji individually if set true
        @remove_digit: boolean, remove words containing digit if set true
        @keyword_hashtag**: boolean, whether to applify the keywords in a tweet by labeling them as hashtags
    -Returns:
        @word_counter: collection.Counter object, a word counter of the cleaned tweets
        @trainable_tweets: list, with each item in foramt {'tweet': tweet, 'hashtag': hashtag_list}
    """
    word_counter = collections.Counter()
#     trainable_tweets = {'tweet': [], 'hashtag': []}
    trainable_tweets = []
    for i, tweet in enumerate(raw_tweets):
        if deal_emoji is True:
            tweet = add_space(tweet)
        tweet = deal_special_char(tweet.lower(), remove_digit)
        hashtag = ''
        for w in tokenize(tweet):
            if is_hashtag(w):
                hashtag += (' ' + w)
        word_counter.update(tokenize(tweet))
#         trainable_tweets['tweet'].append(tweet)
#         trainable_tweets['hashtag'].append(hashtag)
        trainable_tweets.append({'tweet': tweet, 'hashtag': hashtag})
    return word_counter, trainable_tweets

def build_dict(word_counter):
    """
    Generate a dictionary that maps a word to an index and vice versa
    -Args:
        @word_counter: collection.Counter object, a word counter of the cleaned tweets
    -Returns:
        @word2idx: dict, {word: idx}
        @vocab_size: int, size of vocab in the input txt
        @hashtag_pool: LongTensor, indices of hashtags
    """
    vocab = set([word for word in word_counter])
#     if remove_digit is True:
#         vocab = set(filter(lambda x: contain_digit(x) is not True, vocab))
#     if len(stop_words) > 0:
#         vocab = vocab.difference(set(stop_words))
    vocab = list(vocab)
    vocab = [PADDING, UNKNOWN] + vocab
    word2idx = dict(zip(vocab, range(len(vocab))))
    vocab_size = len(vocab)
    hashtag_pool = []
    for word in word2idx.keys():
        if word[0] == '#':
            hashtag_pool += [word2idx.get(word)]
    return word2idx, vocab_size, hashtag_pool

def tweet2seq(word2idx, trainable_tweets, max_seq_length=20):
    """
    Annotate tweets with sequence of indices.
    -Args:
        @word2idx: dict, {word: index}
        @tarinable_tweet: list of dicts in the form {tweet: str, hashtag: []}
        @max_seq_length: maximum length of sequence
        **Note: final length of tweet seq will be different for `pad_both_end` settings
    -Returns:
        None, append two keys to original trainable_tweets: 'tweeter_seq'(tensor) and 'hashtag_seq'(list)
    """
    # seq_time, hash_time = 0, 0
    for i, item in enumerate(trainable_tweets):
        # item is {'tweet': str, 'hashtag': []}
        
        # start = time.time()
        tweet_seq_temp = torch.zeros(max_seq_length)
        token_seq = tokenize(item['tweet'])
        for j in range(max_seq_length):
            if j >= len(token_seq):
                tweet_seq_temp[j:] = word2idx[PADDING]
                break
            else:
                try:
                    tweet_seq_temp[j] = word2idx[token_seq[j]]
                except:
                    tweet_seq_temp[j] = word2idx[UNKNOWN]
        item['tweet_seq'] = tweet_seq_temp.long().view(1, -1)
        # end = time.time()
        # seq_time += (end-start)

        # start = time.time()
        item['hashtag_seq'] = []
        if len(item['hashtag']) > 0:
            for h in tokenize(item['hashtag']):
                try:
                    item['hashtag_seq'].append(word2idx.get(h))
                except:
                    pass
        # end = time.time()
        # hash_time += (end-start)
    # print('seq_time: {}, hash_time: {}'.format(seq_time, hash_time))
