import os
import re
import json
import sys
import matplotlib.pyplot as plt 
from collections import Counter
import traceback
import pickle

import numpy as np 
import pandas as pd 

import jieba
from gensim.models import Word2Vec
from breakouts_detection import peaks_detection, display_breakouts, detect_start_end
from tags_analysis import ClusetrsSet, TagsCluster
from preprocess import tags_extractor



# 统计cluster_number列的cluster分布（因为项为list，所以不能直接value_counts()）
def clusters_value_counts(clusters_values, topk=30, density=True):
	result_dict = {}
	clusters_pool = []
	for target_clusters in clusters_values:
		if type(target_clusters)==str:
			target_clusters = eval(target_clusters)
		for tc in target_clusters:
			clusters_pool.append((tc[0],tc[1]))
	clusters_pool = Counter(clusters_pool)

	for item,value in clusters_pool.most_common(topk):
		if density:
			value = value/sum(clusters_pool.values())
		result_dict[item] = value
	return result_dict




# =============== genre ===============
def add_col_genre(csv_read_path, csv_write_path):

	import pymysql
	conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='SFpqwnj285798,.', db='NetEase')
	def match_track2genres(track_id):
		cursor = conn.cursor()
		sql = 'SELECT tag FROM tagged_tracks WHERE track_id=%s'
		cursor.execute(sql,(track_id))
		res = cursor.fetchall()

		genres = [t[0] for t in res]
		return genres

	df = pd.read_csv(csv_read_path)
	df['genres'] = df.apply(lambda d:match_track2genres(d['file'][:-4]), axis=1)
	conn.close()

	df.to_csv(csv_write_path, index=False, encoding='utf_8_sig')


# =============== lyrics ===============
def get_lyrics_path(csv_read_path, lyrics_package_path):
	df = pd.read_csv(csv_read_path)
	tracks_id = list(map(lambda x:x[:-4], df['file'].unique()))

	track_id_2_filepath = {}
	for root,dirs,files in os.walk(lyrics_package_path):
		for file in files:
			if file[:-5] in tracks_id:
				filepath = os.path.join(root,file)
				track_id_2_filepath[file[:-5]] = filepath

	print(len(tracks_id)) # 543
	print(len(track_id_2_filepath)) # 145
	with open('../data/track_id_2_filepath.pkl','wb') as f:
		pickle.dump(track_id_2_filepath, f)


def save_lyrics_files():
	from process_lyrics import generate_lyrics
	with open('../data/track_id_2_filepath.pkl','rb') as f:
		track_id_2_filepath = pickle.load(f)

	for track_id,filepath in track_id_2_filepath.items():
		with open('../data/lyrics/{}.txt'.format(track_id), 'w') as f:
			f.write('\n'.join(generate_lyrics(filepath)))

def lyrics_feature_by_cluster(csv_read_path, csv_write_path=None):
	df = pd.read_csv(csv_read_path)
	def has_cluster(text, cluster):
		if cluster in eval(text): return 1
		else: return 0
	for cluster in clusters_value_counts(df['cluster_number'].values).keys():
		if cluster==(0, 'others'): continue
		print(cluster)
		files = list(df.apply(lambda d:d['file'] if has_cluster(d['cluster_number'], cluster) else 'x', axis=1).unique())
		print(files)
		break


	def get_keywords(filepath):
		content = open(filepath).read()
		keywords = tags_extractor(content,topk=10,deep_clean=False)
		return keywords

	for file in df['file'].unique():
		track_id = file[:-4]
		lyrics_filepath = '../data/lyrics/{}.txt'.format(track_id)
		if not os.path.exists(lyrics_filepath): continue
		print(track_id)
		print(get_keywords(lyrics_filepath))




def clusters_related_analysis(read_path):
	topics, events = list(map(lambda x:x.split(),open('../resources/topic_or_event.txt').read().splitlines()))

	df1 = pd.read_csv(read_path)
	df1['lonely'] = 0
	df1['lonely'] = df1.groupby('file').transform(lambda g:1 if g.count()==1 else 0)

	df2 = df1[df1['lonely']==0]

	# 考察只发生一次爆发的歌曲
	def lonely_event_or_topic():
		lonely_dict = clusters_value_counts((df1[df1['lonely']==1]['cluster_number'].values))
		for k,v in lonely_dict.items():
			print(k,v)
		events_rates = np.sum([v for k,v in lonely_dict.items() if str(k[0]) in events])
		topics_rates = np.sum([v for k,v in lonely_dict.items() if str(k[0]) in topics])
		print(events_rates/(events_rates+topics_rates))
		print(topics_rates/(events_rates+topics_rates))

		data = df1[df1['lonely']==1]['reviews_count'].values

		plt.hist(data, bins=20)
		plt.title('lonely breakouts - avg:{:.2f}'.format(np.mean(data)))
		plt.xlabel('reviews_count')
		plt.savefig('../images/lonely_breakouts.png')
		plt.show()


	# 考察发生多次爆发，但是每次的main tag一致的歌曲
	def focus_event_or_topic():
		focus_clusters = []
		for file in df2['file'].unique():
			tmp_df = df2[df2['file']==file]
			cluster_count = clusters_value_counts(tmp_df['cluster_number'].values, density=False)
			for k,v in cluster_count.items():
				if v==len(tmp_df):
					focus_clusters.append(k)
		focus_dict = dict(Counter(focus_clusters).most_common(30)) 
		size = np.sum(list(focus_dict.values()))
		for k in focus_dict:
			focus_dict[k] /= size

		for k,v in focus_dict.items():
			print(k,v)
		events_rates = np.sum([v for k,v in focus_dict.items() if str(k[0]) in events])
		topics_rates = np.sum([v for k,v in focus_dict.items() if str(k[0]) in topics])
		print(events_rates/(events_rates+topics_rates))
		print(topics_rates/(events_rates+topics_rates))


	# 考察发生多次爆发的歌曲中，是否存在tag共现
	def multi_breakouts():
		multi_breakouts = []
		for file in df2['file'].unique():
			tmp_df = df2[df2['file']==file]
			def get_1_cluster(text):
				start = text.index('(')
				end = text.index(')')
				return eval(text[start:end+1])
			clusters = list(map(lambda x:get_1_cluster(x), tmp_df['cluster_number'].values))
			clusters = set(list(filter(lambda x:x!=(0, 'others'), clusters)))
			if len(clusters)<=1: continue
			multi_breakouts.append(clusters)
		for item in multi_breakouts:
			print(item)

		from fp_growth import FPGrowth 
		model = FPGrowth(min_support=1)
		model.train(multi_breakouts)

	# 一次爆发内部，tag的共现
	def breakout_inner_cooccurence():
		data = []
		for item in df1['cluster_number'].values:
			item = list(eval(item))
			item = set(list(filter(lambda x:x!=(0, 'others'), item)))
			if len(item)>1:
				data.append(item)

		from fp_growth import FPGrowth 
		model = FPGrowth(min_support=1)
		model.train(data)


	def focus_interval():
		# nonlocal df2
		df3 = df2[df2['focus']==1]
		intervals = []
		for file in df3['file'].unique():
			tdf = df3[df3['file']==file]
			cluster_number = tdf['cluster_number'].values[0]
			str2date = lambda x:datetime.strptime(x,'%Y-%m-%d')
			starts, ends, centers = list(map(str2date, tdf['start_date'].values)),\
									list(map(str2date, tdf['end_date'].values)),\
									list(map(str2date, tdf['center'].values))
			in_deltas = [(ends[i]-starts[i]).days for i in range(len(starts))]
			ex_intervals = [(centers[i+1]-centers[i]).days for i in range(len(centers)-1)]
			if 'others' not in cluster_number:
				intervals.extend(ex_intervals)

			print(cluster_number)
			print(in_deltas)
			print(ex_intervals,'\n')

		plt.hist(intervals)
		plt.show()


	# lonely_event_or_topic()
	# print('='*20)
	# focus_event_or_topic()
	# focus_interval()
	multi_breakouts()
	# breakout_inner_cooccurence()





if __name__ == '__main__':
	read_path = '/Volumes/nmusic/music/topsongs-reviews/reviews/'
	model_path = '../models/word2vec/a.mod'

	txt_write_path = '../data/breakouts_pucha_gg15.txt'
	csv_write_path = '../data/breakouts_pucha_gg15.csv'
	clusters_model_path = '../models/clusters/100nt10a6_tags_clusters_model.pkl'

	# add_col_genre('../data/breakouts_pucha_gg15_2.csv', '../data/breakouts_pucha_gg15_3.csv')
	# get_lyrics('../data/breakouts_pucha_gg15_2.csv','/Volumes/nmusic/music/all_lyrics')
	# save_lyrics_files()
	# lyrics_feature_by_cluster('../data/breakouts_pucha_gg15_2.csv')

	clusters_related_analysis('../results/pucha/BorgCube2_65b1.csv')
