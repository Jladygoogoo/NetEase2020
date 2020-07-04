# import os

# grams_pool = []
# for file in os.listdir('../resources/grams/'):
# 	if 'DS' in file: continue
# 	grams = open('../resources/grams/'+file).read().splitlines()
# 	grams_pool.extend(grams)

# grams_pool = set(grams_pool)

# stops = open('../resources/grams_stop.txt').read().splitlines()
# new_grams_pool = []
# for w in grams_pool:
# 	add = 1
# 	for stop in stops:
# 		if stop in w:
# 			add = 0
# 			break
# 	if add:
# 		new_grams_pool.append(w)

# print(len(new_grams_pool))
# with open('proxied_grams.txt','w') as f:
# 	f.write('\n'.join(new_grams_pool))

import jieba
for w in jieba.cut('微博来了'):
	print(w)

# from preprocess import W2VSentenceGenerator

# read_path = '/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews_text/'
# flag = 0
# for sent in W2VSentenceGenerator(read_path, file_is_sent=True):
# 	print(sent)
# 	flag += 1
# 	if flag>2:
# 		break
