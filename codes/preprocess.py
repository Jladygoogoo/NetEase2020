import os
import jieba
import re
import pickle
import numpy as np 
from collections import Counter 
from gensim import models

from search_kaomojis import search_kaomojis

gram_path = '../resources/proxied_grams.txt'
jieba.load_userdict(gram_path)

skipchars = open('../resources/skipchars.txt').read().splitlines()
stopwords = open('../resources/sup_stopwords.txt').read().splitlines()
rubbish_tags = open('../resources/rubbish_tags.txt').read().splitlines()


def replace_noise(text):
	# 对于全文的预处理，统一替换，单个单个句子处理效率太低
	re_emoji = re.compile(u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]',re.UNICODE)
	text = re.sub(re_emoji,'',text)

	kaomojis = search_kaomojis(text)
	for k in kaomojis:
		text = text.replace(k,'kaomoji')

	# 将连词合为一个
	text = re.sub(r'-|_|——|－','',text)
	text = re.sub(u'\u3000|\u200b|\xa0',' ',text)
	puncs = open('../resources/punctuations.txt').read().splitlines()
	for p in puncs:
		text = text.replace(p,'')

	# 英文词组
	regex = re.compile(r'([a-z]+) ([a-z]+)')
	# 两次才完整（虽然这种做法很丑）
	for i in range(2): text = re.sub(regex,r'\1空\2',text)

	return text


def sent_preprocess(text,deep_clean=False):
	text = text.lower()

	if deep_clean:
		stops = stopwords+rubbish_tags
	else:
		stops = stopwords

	words = []
	for x in jieba.cut(text):

		# 筛除只包含一个字的tag
		if len(x)<2 or len(set(x))==1: continue
		if re.match(r'^\d+$',x): continue
		if '.' in x: continue

		# 对于两个字的tag，用skipchar来筛除
		conti = 0
		if len(x)==2:
			for c in skipchars:
				if c in x: conti = 1;break
		if conti: continue

		# 用停词表筛选
		if  x in stops or '第' in x: continue
		words.append(x)
		
	return words


class W2VSentenceGenerator():
	def __init__(self,path,min_size=2,file_is_sent=False):
		self.path = path
		self.min_size = min_size
		self.file_is_sent = file_is_sent

	def __iter__(self):
		for root,dirs,files in os.walk(self.path):
			for file in files:
				if not '.txt' in file: continue
				text = replace_noise(open(os.path.join(root,file)).read()[:6000])

				# 将整个文档看作一个句子
				if self.file_is_sent:
					file_sent = []
					for line in text.splitlines():
						file_sent.extend(sent_preprocess(line))
					yield file_sent

				# 读取文档中的每一行为一个句子
				else:
					for line in text.splitlines():
						sent = sent_preprocess(line)
						if len(sent)>=self.min_size:
							yield sent


def tags_extractor(text,topk=8,deep_clean=True):
	text = replace_noise(text)

	words = []
	for line in text.splitlines():
		sent = sent_preprocess(line,deep_clean=deep_clean)
		words.extend(sent)
	counter = Counter(words)
	tags = [x[0] for x in counter.most_common(topk)]

	return tags



# 对口doc2vec的句子生成器（加上文档标签和数目限制）
class TaggedSentenceGenerator():
	def __init__(self,path,mode='train'):
		self.path = path
		self.mode = mode

	def __iter__(self):
		MAX_SIZE = 500 # hyperparameters🙋

		flag = 0
		files = os.listdir(self.path)
		try:
			files.remove('.DS_Store')
		except: pass

		if self.mode == 'train':
			with open('../models/d2v_tag2track.pkl','wb') as f:
				tag2track = dict(enumerate(map(lambda x:x[:-4],files)))
				pickle.dump(tag2track,f)

		for file in files:
			text = doc_preprocess(open(self.path+file,'r').read())
			lines = text.splitlines()
			# 控制采样评论数不超过MAX_SIZE
			indices = set(np.round(np.linspace(0,len(lines)-1,MAX_SIZE)))

			# 一个file里面的所有句子属于一个tagged_document
			sents = []
			for i in indices:
				sent = sent_preprocess(lines[int(i)])
				if len(sent)>5:
					sents.extend(sent)

			yield models.doc2vec.TaggedDocument(sents,[str(flag)])
					
			flag += 1
			if flag%100==0:
				print("load {} files in total.".format(flag))


def test():
	test_path = '../data/big_text_100/'
	for sent in W2VSentenceGenerator(test_path,file_is_sent=False):
		print(sent)
	# text = replace_noise("let's dance啦啦啦琅琊榜")
	# print(text)
	# print(list(jieba.cut(text)))
		# print(sent_preprocess(text))
	




def main():
	pass

if __name__ == '__main__':
	# main()
	test()
	


