#coding:utf-8
import re
import os
import sys
import json
import pickle
import numpy as np 
import pandas as pd


import jieba
from collections import Counter

re_emoji_path = '../resources/re_emoji.txt'
stopwords = open('../resources/sup_stopwords.txt').read().splitlines()
gram_stop = open('../resources/gram_stop.txt').read().splitlines()


def replace_noise(text):
	# 对于全文的预处理，统一替换，单个单个句子处理效率太低
	re_emoji = re.compile(u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]',re.UNICODE)
	text = re.sub(re_emoji,'',text)

	# 处理标点符号
	text = re.sub(u'\u3000|\u200b|\xa0',' ',text)
	puncs = open('../resources/punctuations.txt').read().splitlines()
	for p in puncs:
		text = text.replace(p,'')

	return text


def text_butcher(text,ngram=2,english=True):
	text = text.lower()
	slices = []
	words = list(jieba.cut(text))

	for i in range(len(words)-1):
		conflag = 0
		if not re.search('[\u4e00-\u9fa5]',''.join(words[i:i+ngram])): continue

		for word in words[i:i+2]:
			if word in stopwords: conflag=1;break
		if conflag: continue

		if set(words[i])==set(words[i+1]): continue

		gram = ''.join(words[i:i+2])
		conflag = 0
		for w in gram_stop:
			if w in gram: 
				conflag=1;break
		if conflag: continue
		slices.append(gram)

	if english:
		en_words = re.findall(r'[^\u4e00-\u9fa5]+',text)
		for ew in en_words:
			if len(ew.split())>1:
				slices.append(ew)

	return slices


# text中包含换行
def get_topk_grams(text,topk=10,minn2=30):
	grams = []
	topk_grams = []

	# 窗口为2
	grams2 = []
	for line in text.splitlines():
		grams2.extend(text_butcher(line))
	counter2 = Counter(grams2)
	for k,v in counter2.most_common(topk):
		if v>=minn2: 
			conflag = 0
			for w in topk_grams:
				# 防止截断
				if list(jieba.cut(w))[-1]==list(jieba.cut(k))[0] or list(jieba.cut(w))[0]==list(jieba.cut(k))[-1]: 
					conflag = 1;break
				if len(w)>=6:
					conflag = 1;break
			if conflag: continue
			topk_grams.append(k)

	return topk_grams



def run(read_path, write_path, log_num, f_start, f_end):
	flag = 0
	grams_batch = []

	for root,dirs,files in os.walk(read_path):
		for file in files:
			flag+=1
			if flag<f_start or flag>=f_end: continue

			if 'json' not in file: continue
			path = os.path.join(root, file)
			with open(path) as f:
				content = json.load(f)
			text = '\n'.join([x['content'] for x in content])
			grams = get_topk_grams(replace_noise(text),topk=10)
			grams_batch.extend(grams)

			if flag%log_num==0:
				print(f"load {flag} files.")

				with open(write_path,'a') as f:
					f.write('\n'.join(set(grams_batch))+'\n')

				grams_batch = []
			

	# grams = set(open(write_path).read().splitlines())
	# print(len(grams))
	# with open('../resources/grams_u.txt','w') as f:
	# 	f.write('\n'.join(grams))


def test(test_path):
	flag = 0
	for root,dirs,files in os.walk(test_path):
		for file in files:
			flag+=1
			if flag>100: break
			if 'json' not in file: continue
			path = os.path.join(root, file)
			with open(path) as f:
				content = json.load(f)
			text = '\n'.join([x['content'] for x in content])
			text = replace_noise(text)
			# text = open(test_path).read()
			print(get_topk_grams(text,topk=10))



if __name__ == '__main__':
	# main run
	read_path = '/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews'

	worker = int(sys.argv[1])
	f_start, f_end = 5000*(worker-1), 5000*worker
	write_path = '../resources/proxied_grams_{}.txt'.format(worker)

	run(read_path, write_path, 50, f_start=f_start, f_end=f_end)

	# test
	# test_path = '/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews'
	# test(test_path)
