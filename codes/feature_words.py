#coding:utf-8
'''
训练词袋模型->计算文档的tf-idf矩阵->topk特征词
'''

import os
import pickle
import json
import csv

import numpy as np
import pandas as pd
import re

import jieba
import jieba.analyse
import jieba.posseg as pseg
from gensim import corpora
from gensim import models
from gensim.models import Word2Vec

#reviews_package_PATH = '/Users/inkding/ML_Database/data/music/网易云/reviews'
reviews_csv_PATH = './files/reviews_topk_extracted.csv'

songid_save_PATH = '../source_tools/song_id.txt'
doc_sent_word_save_PATH = '../source_tools/doc_sent_word.txt'
corpora_dict_save_PATH = '../source_tools/corpora_dict.dict'
tfidif_model_save_PATH = '../source_tools/tfidif_model.mod'
doc_wordbow_save_PATH = '../source_tools/doc_wordbow.txt'
feature_words_database_save_PATH = '../source_tools/feature_words_database.txt'

mydict_PATH = '../source_tools/my_dict.txt'
jieba.load_userdict(mydict_PATH)
stopwords_PATH = '../source_tools/stopwords_list.txt'
stopwords = open(stopwords_PATH,'r').read().splitlines()


#train necessary models and get middle files...
def sent2word(sent):
	words = []
	accept_seg = ['a','ad','an','b','n','nr','ns','nt','nz','z','vn']
	sent = re.sub(r'\r|\n',' ',sent)
	sent = sent.replace(u'\u200b', ' ')
	for word,flag in pseg.cut(sent):
		if word not in stopwords and flag in accept_seg:
			words.append(word)
	words = list(filter(lambda w:len(w)>1,words))
	return words

def get_only_words(doc):
	words = []
	for sent in doc:
		for word in sent:
			words.append(word)
	return words


def split_doc(df,num=0):
	#按文档拆分，每个文档中按句拆分，每个句子中按词拆分
	doc_sent_word_corpus = []
	#按文档拆分，每个文档按词拆分
	if not num: num=df.shape[0]
	for doc in df.contents.values[:num]:
		review = ' '.join(eval(doc))
		review_sent = re.split(r'。|？|\.\.\.|……|~',review)
		review_sent_word = list(map(lambda s:sent2word(s),review_sent))
		review_sent_word = list(filter(lambda s:len(s)>2,review_sent_word))
		doc_sent_word_corpus.append(review_sent_word)

	with open(doc_sent_word_save_PATH,'wb') as f:
		pickle.dump(doc_sent_word_corpus,f)

def get_corpora_dict(doc_word_corpus):
	corpora_dict = corpora.Dictionary(doc_word_corpus)
	#这里的设定就是很tricky，现在调在总文档数的0.25%-3%之间
	corpora_dict.filter_extremes(no_below=30,no_above=0.03) 

	corpora_dict.save(corpora_dict_save_PATH)


def get_wordbow(corpora_dict,doc_word_corpus):
	#按文档拆分，每个文档用词袋表示
	doc_wordbow_corpus = []
	for doc in doc_word_corpus:
		doc_wordbow_corpus.append(corpora_dict.doc2bow(doc))
	with open(doc_wordbow_save_PATH,'wb') as f:
		pickle.dump(doc_wordbow_corpus,f)


def train_tfidf_model(doc_wordbow_corpus,corpora_dict):
	#根据核心corpora训练的tfidf模型
	tfidf_model = models.TfidfModel(corpus=doc_wordbow_corpus, dictionary=corpora_dict)
	tfidf_model.save(tfidif_model_save_PATH)


def train():
	'''
	df = pd.read_csv(reviews_csv_PATH)
	print("df.shape",df.shape)
	split_doc(df) #得到doc_sent_word_corpus
	print("got split doc.")

	with open(songid_save_PATH,'wb') as f:
		pickle.dump(df.song_id.values,f)
	'''
	with open(doc_sent_word_save_PATH,'rb') as f:
		doc_sent_word_corpus = pickle.load(f)
	
	doc_word_corpus = list(map(lambda d:get_only_words(d),doc_sent_word_corpus))

	get_corpora_dict(doc_word_corpus)
	corpora_dict = corpora.Dictionary.load(corpora_dict_save_PATH)
	print("got corpora dictionary.")

	get_wordbow(corpora_dict,doc_word_corpus)
	with open(doc_wordbow_save_PATH,'rb') as f:
		doc_wordbow_corpus = pickle.load(f)
	print("got doc wordbow.")
	
	train_tfidf_model(doc_wordbow_corpus,corpora_dict)
	tfidf_model = models.TfidfModel.load(tfidif_model_save_PATH)
	print("got tfidf model.")

def resort_by_word2vec(topk,words):
	path = '../models/word2vec/w2v_model_all.mod'
	model = Word2Vec.load(path)

	valid_words = []
	for p in words:
		if model.wv.__contains__(p[0]):
			valid_words.append(p)
	words = valid_words

	topk_words = [words[0]]
	words = words[1:]
	while len(topk_words)<topk and len(words)>0:
		for i in range(len(words)):
			word = words[i][0]
			score = words[i][1]
			words[i] = (word,(1-model.wv.similarity(word,topk_words[-1][0]))*score)
		words = list(sorted(words,key=lambda x:x[1],reverse=True))
		topk_words.append(words[0])
		words = words[1:]

	return topk_words




#core function
def get_topk_words(index=None,text=None,topk=10):
	if index and text:
		raise ValueError("can't set query text and index at the same time!")
	if not index and not text:
		raise ValueError("query text and index can't be None at the same time!")
	
	with open(doc_wordbow_save_PATH,'rb') as f:
		doc_wordbow_corpus = pickle.load(f)
	corpora_dict = corpora.Dictionary.load(corpora_dict_save_PATH)
	tfidf_model = models.TfidfModel.load(tfidif_model_save_PATH)
	
	if text:
		words = sent2word(text)
		if len(words)<topk:
			return words
		else:
			wordbow = corpora_dict.doc2bow(words)
	else:
		wordbow = doc_wordbow_corpus[index]

	dict_id2token = {v: k for k, v in corpora_dict.token2id.items()}
	tfidf = tfidf_model[wordbow]
	tfidf = list(sorted(tfidf, key=lambda x:x[1],reverse=True))
	top_words = [(dict_id2token[p[0]],p[1]) for p in tfidf[:2*topk]]
	# print(top_words[:topk])
	topk_words = resort_by_word2vec(topk,top_words)
	# print(topk_words)

	
	return topk_words


#create database for search
def create_feature_words_database():
	feature_words_database = []
	
	#顺序和songids是对应的
	for wordbow in doc_wordbow_corpus:
		feature_words_database.append(get_topk_words(wordbow=wordbow))
	with open(feature_words_database_save_PATH,'wb') as f:
		pickle.dump(feature_words_database,f)


def test():
	# while True:
	# 	index = input('index:')
	# 	get_topk_words(index=int(index))
	# 	print()
	text = open('/Users/inkding/Desktop/NetEase2020/data/breakouts_text/27713922_1.txt').read()
	print([x[0] for x in get_topk_words(text=text)])



if __name__ == '__main__':
	test()


