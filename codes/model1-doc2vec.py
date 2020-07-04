import os
import pickle
import logging
import functools
import time

import numpy as np
import pandas as pd
import re

from gensim import corpora
from gensim import models

from my_decorators import time_record
from preprocess import *

# dm: 0/1 -> 'distributed bag of words'/'distributed memory'
# window: The maximum distance between the current and predicted word within a sentence.
# min_count: Ignores all words with total frequency lower than this
# max_vocab_size: Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
# alpha: The initial learning rate
# min_alpha: Learning rate will linearly drop to `min_alpha` as training progresses

# 之前跑过一遍corpora.Dictionary，得到词汇量为692795【但是忘了save干】

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

@time_record('model training')
def train(**params):
	model = models.Doc2Vec(**params)
	return model

def train_model():
	path = '../data/raw_reviews_content/'
	model = train(documents=TaggedSentenceGenerator(path), dm=1, vector_size=200, window=8, workers=4, epochs=20)
	model.save('../models/doc2vec.mod')

	# assess


def test_model():
	model = models.Doc2Vec.load('../models/doc2vec.mod')

	print("start testing...")
	tag2track = open('../models/d2v_tag2track.txt').read().splitlines()
	tag2track = dict(map(lambda x:x.split(),tag2track))

	while True:
		words = sent_preprocess(doc_preprocess(input("plz enter hit sentence:\n")))
		vector = model.infer_vector(doc_words=words)
		print("top 10 results:")
		for tag,sim in model.docvecs.most_similar([vector],topn=10):
			print(f"similarity:{sim} - tag:{tag}, track_id:{tag2track[tag]}")
		print()

if __name__ == '__main__':
	train_model()
	test_model()





