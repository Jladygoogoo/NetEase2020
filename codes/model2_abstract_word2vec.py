import os
import re
import logging
import sys
from collections import Counter

from gensim.models import Word2Vec
from preprocess import tags_extractor,W2VSentenceGenerator

# 设置logging很重要
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s', datefmt='%m-%d %H:%M:%S')

# stream_log = logging.StreamHandler()
# file_log = logging.FileHandler('logs/proxied_word2vec.log')

# stream_log.setFormatter(formatter)
# file_log.setFormatter(formatter)

# logger.addHandler(stream_log)
# logger.addHandler(file_log)
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s', 
					datefmt='%m-%d %H:%M:%S', filename='logs/proxied_word2vec_2.log')


def train(read_path,save_path,window,iters=6):

	model = Word2Vec(W2VSentenceGenerator(read_path,file_is_sent=True),
					size=300, min_count=20, window=window, iter=iters,
					workers=6, seed=21)
	model.save(save_path)
	print("\nmodel saved.")


# 不行 我之前保存的格式不对 所以无法继续做训练 妇产科
def continue_train(model_path,sentences=None,use_generator=False,file_dir=None):
	model = Word2Vec.load(model_path)
	print("loading from {}".format(model_path))

	if use_generator==True and file_dir:
		model.build_vocab(W2VSentenceGenerator(read_path,file_is_sent=True),update=True)
		model.train(W2VSentenceGenerator(read_path,file_is_sent=True), total_examples=model.corpus_count, epochs=1)
	else:
		model.build_vocab(sentences,update=True)
		model.train(sentences,total_examples=model.corpus_count,epochs=1)

	if '好声音' in model:
		print(model.wv.most_similar('中国好声音'))

def run(model_path):
	model = Word2Vec.load(model_path)
	print("loading from {}".format(model_path))

	while True:
		try: 
			word = input("input: ")
			if not model.wv.__contains__(word):
				print("word not found.")
				continue
			for p in model.wv.most_similar(word):
				print(p)
		except KeyboardInterrupt:
			print("bye.")
			sys.exit(0)


def evaluate(model_path,testing_path,size=100,topk=10):
	model = Word2Vec.load(model_path)
	print("loading from {}".format(model_path))
	print("evaluating scale: ",size)

	def accuracy(filepath):
		words_corpus = []
		text = open(filepath).read()
		tags = tags_extractor(text,topk=topk)
		# print(tags)

		accuracy = 0
		pairs = []
		for i in range(topk-1):
			for j in range(i,topk):
				w1,w2 = tags[i],tags[j]
				if not (model.wv.__contains__(w1) and model.wv.__contains__(w2)):
					continue
				pairs.append((w1,w2))
		for p in pairs:
			w1,w2 = p
			accuracy += model.wv.similarity(w1,w2)
		accuracy /= len(pairs)
		# print(accuracy,'\n')

		return accuracy

	acc = 0
	flag = 0
	for root,dirs,files in os.walk(testing_path):
		breakflag = 0
		for file in files:
			if 'txt' not in file: continue
			acc += accuracy(os.path.join(root,file))
			flag += 1
			if flag==size: breakflag=1; break 
		if breakflag: break

	acc /= size

	return acc



if __name__ == '__main__':
	read_path = '/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews_text/'
	model_path = '../models/word2vec/b1.mod'
	# train(read_path=read_path, save_path=model_path,
	# 	window=2)
	# print(evaluate(model_path,read_path))
	run(model_path)
	# template = '../models/word2vec/abs_word2vec_{}.mod'
	# models_path = [template.format(i) for i in (2,10,20)]
	# models_path.append('../models/word2vec/w2v_model_all.mod')
	# for mp in models_path:m
	# 	print("model",os.path.basename(mp))
	# 	print("accuracy:",evaluate(mp,read_path),'\n')
	


