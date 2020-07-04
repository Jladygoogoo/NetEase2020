import os
import sys
import matplotlib.pyplot as plt 
import warnings
from datetime import datetime
from collections import Counter
import logging
import traceback
import math
import numpy as np 
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

from tags_analysis_plot import power_law_plot, draw_donut, draw_heap


import jieba
from gensim.models import Word2Vec
from preprocess import tags_extractor

from pyecharts.charts import Pie,Bar
from pyecharts import options as opts

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING,format='%(asctime)s: %(message)s',
					filename='../results/results_config.log',filemode='a')

# 类构造：基本组件

class TagsCluster:
	def __init__(self,tag0=None,tag0_num=None):
		if tag0:
			self.tags = Counter([tag0])
			self.count = 1
			self.reviews_nums = [tag0_num]
		else:
			self.tags = Counter([])
			self.count = 0
			self.reviews_nums = []		

	def __add__(self,other):
		new_c = TagsCluster()
		new_c.tags = self.tags + other.tags 
		new_c.count = self.count + other.count
		new_c.reviews_nums = self.reviews_nums + other.reviews_nums
		return new_c

	def get_center(self):
		return self.tags.most_common(1)[0][0]

	def distance(self,tag,model):
		if not model.wv.__contains__(tag):
			return 0
		if tag in self.tags:
			return 1
		else:
			center = self.get_center()
			distance = model.wv.similarity(center,tag)
			return distance

	def add(self,tag,tag_num):
		self.tags.update(Counter([tag]))
		self.count += 1
		self.reviews_nums.append(tag_num)



class ClusetrsSet:
	def __init__(self,model_path,affinity=0.7,min_start_count=3):
		self.model = Word2Vec.load(model_path)
		self.affinity = affinity
		self.min_start_count = min_start_count
		self.input_tags_num = 0
		self.size = 0
		self.clusters = []


	def pruning(self):
		self.clusters = list(filter(lambda x:x.count>self.min_start_count, self.clusters))

	def merging(self):
		merge_num = 0
		roundd = len(self.clusters)
		while roundd>0:
			mergeflag = 0
			current_c = self.clusters[0]
			for j in range(1,len(self.clusters)):
				want_merge = 0
				for t1 in current_c.tags:
					for t2 in self.clusters[j].tags:
						if self.model.wv.similarity(t1,t2)>=self.affinity:
							want_merge += 1
				want_merge /= (len(current_c.tags)*len(self.clusters[j].tags))

				if want_merge>=self.affinity*0.5:
					new_cluster = current_c + self.clusters[j]
					self.clusters.pop(j)
					self.clusters.pop(0)
					self.clusters.append(new_cluster)
					merge_num += 1
					roundd -= 2
					mergeflag = 1

					break
			if not mergeflag:
				self.clusters.append(self.clusters.pop(0))
				roundd -= 1
		print("merging finished, {} cluster(s) being merged.".format(merge_num))


	def growing(self,tags,save_path,save_result=True):
		# 必须是(tag, reviews_num)的形式
		if len(tags[0])!=2:
			raise ValueError("wrong tags input.")
		print("growing clusters set ...")

		start = 0
		while not self.model.wv.__contains__(tags[start][0]):
			start += 1
		self.clusters.append(TagsCluster(tag0=tags[start][0],tag0_num=tags[start][1]))
		self.input_tags_num += 1 
		
		for t,n in tags[start:]:
			distances = [(i,cluster.distance(t,self.model)) for i,cluster in enumerate(self.clusters)]
			if 0 in [d[1] for d in distances]: continue

			max_cindex, max_closeness = 0, 0
			for i,c in distances:
				if c>max_closeness:
					max_closeness = c
					max_cindex = i 
			if max_closeness>self.affinity:
				self.clusters[max_cindex].add(t,n)
			else:
				self.clusters.append(TagsCluster(tag0=t,tag0_num=n))

			self.input_tags_num += 1 
			if self.input_tags_num%1000==0:
				old_size = self.size
				self.pruning()
				self.size = len(self.clusters)
				print("loading {} tags with {} clusters (add {})"\
						.format(self.input_tags_num, self.size, self.size-old_size))
				self.min_start_count += 1

		self.pruning()
		self.merging()
		self.clusters = sorted(self.clusters,key=lambda x:x.count,reverse=True)

		if save_result:
			self.save_clusters_result(save_path=save_path)



	def save_clusters_result(self,save_path):
		with open(save_path,'w') as f:
			for i,c in enumerate(self.clusters,start=1):
				tags = [p[0] for p in c.tags.most_common()]
				f.write("{} count:{} - [{}]".format(i,c.count,', '.join(tags))+'\n')

	def save(self,save_path,pickle_protocol=2):
		if not os.path.exists(os.path.dirname(save_path)):
			os.makedirs(os.path.dirname(save_path))

		with open(save_path,'wb') as f:
			try:
				_pickle.dump(self,f,protocol=pickle_protocol)
				print("successfully saved clusters set.")
			except Exception as e:
				print("failed to save clusters set.")
				print("ERROR:",e)

	@classmethod
	def load(cls,load_path):
		with open(load_path,'rb') as f:
			print("loading {} object from {}".format(cls.__name__,load_path))
			obj = _pickle.load(f)
			print("successfully loaded.")
		return obj


	def classify(self,tags):
		target_clusters = []
		# 每一个tag都选择一个cluster
		for tt in tags:
			if not self.model.wv.__contains__(tt): continue
			clusters_max_simi = []
			max_simi = 0
			for i,c in enumerate(self.clusters):
				focus_range = min(5,len(c.tags))
				cluster_tags = list(zip(*c.tags.most_common()))[0][:focus_range]
				for ct in cluster_tags:
					simi = self.model.wv.similarity(tt,ct)
					if simi > max_simi:
						max_simi = simi
						# max_simi_tags = (tt,ct)
				clusters_max_simi.append(max_simi)
			if max(clusters_max_simi)<0.7:
				continue
			else:
				index = np.argmax(clusters_max_simi)
				cluster_center = self.clusters[index].get_center()
				simi = max_simi
				target_clusters.append((index+1,cluster_center,simi))

		target_clusters = sorted(target_clusters, key=lambda x:x[2], reverse=True)
		unique_target_clusters = set()
		for c in target_clusters:
			c = c[:2]
			unique_target_clusters.add(c)

		if len(unique_target_clusters)==0: 
			unique_target_clusters.add((0,'others'))
		return unique_target_clusters



	def reviews_num_in_clusters(self, image_save_dir):
		if not os.path.exists(image_save_dir): os.makedirs(image_save_dir)
		data = []
		for i,c in enumerate(self.clusters,start=1):
			if c.count <= 15: continue
			beta0, std_sigma, avg = power_law_plot(
							c.reviews_nums, 
							title='cluster-{}'.format(i), 
							# save=True,
							save=False,
							save_path=os.path.join(image_save_dir,'cluster{}.png'.format(i))
						)
			if not beta0: continue
			# print('cluster-{}: beta0={:.2f}, std_sigma={:.2f}, avg={:.2f}'.\
			# 		format(i,beta0,std_sigma,avg))
			data.append([i,beta0,avg,c.count,std_sigma])

		sigma_thres = 0.7
		new_data = list(filter(lambda p:p[-1]<=sigma_thres,data))
		print("keep rate: {:.3f}%".format(len(new_data)/len(data) * 100))

		n_clusters = 6
		method = 'kmeans'
		double_cluster(new_data, n_clusters=n_clusters, method=method, 
				image_save_path='../results/clusters_in_clusters/2cluster_{}_{}_{}_513d.png'.format(method,n_clusters,sigma_thres))
			


	def clusters_in_reviews_rank(self,ranks_list,image_save_path,show_tags_num=8):
		reviews_rank2cluster = {}

		for i,c in enumerate(self.clusters):
			for rn in c.reviews_nums:
				level = rn//100
				for j in range(len(ranks_list)):
					start,end = ranks_list[j]
					if level>=start and level<=end: 
						rank = j
						break

				if rank in reviews_rank2cluster:
					reviews_rank2cluster[rank].update([i])
				else:
					reviews_rank2cluster[rank] = Counter([i])


		reviews_rank2cluster = sorted(reviews_rank2cluster.items(),key=lambda x:x[0])
		all_ranks_tags = []
		for rank,counter in reviews_rank2cluster:
			rank_tags = sorted(counter.most_common(),key=lambda x:x[1],reverse=True)

			pruned_rank_tags = []
			if np.sum([x[1] for x in rank_tags])>20:
				rank_tags = list(filter(lambda x:x[1]>1,rank_tags))
			
			tail_num = np.sum([x[1] for x in rank_tags[show_tags_num:]])
			if tail_num>0:
				rank_tags = rank_tags[:show_tags_num] + [('others',float(tail_num))]
			show_info = lambda x:("{}-{}".format(x[0]+1,self.clusters[int(x[0])].get_center()),x[1]) if x[0]!='others' else x
			rank_tags = list(map(show_info,rank_tags))

			all_ranks_tags.append(rank_tags)

		draw_donut(all_ranks_tags)
		# draw_heap(all_ranks_tags,ranks_list)



# 具体功能实现

# 新建tags数据集
def extract_tags_corpus(read_path,save_path,topk=10):
	tags_pair = []
	for root,dirs,files in os.walk(read_path):
		for file in files:
			if 'DS' in file: continue
			text = open(os.path.join(root,file)).read()
			reviews_num = len(text.split())
			tags = tags_extractor(text,topk=topk)
			tags_pair.extend(list(zip(tags,[reviews_num]*len(tags))))

	with open(save_path,'wb') as f:
		_pickle.dump(tags_pair,f)

# 新建clusters对象
def create_clusters(affinity, tags_path, w2v_model_path):
	# print("tags source", tags_path)
	with open(tags_path,'rb') as f:
		tags_pair = _pickle.load(f)

	name = '{}_{}{}'.format(tags_path.split('/')[-1].split('.')[0], 
							str(affinity).split('.')[1],
							w2v_model_path.split('/')[-1].split('.')[0])
	print("clusters_model name:", name)
	clusters_set = ClusetrsSet(model_path=w2v_model_path,affinity=affinity)
	clusters_set.growing(tags_pair,save_path='../results/tags_clusters/{}.txt'.format(name))
	clusters_set.save(save_path='../models/clusters/{}.pkl'.format(name))

	# logging.warning('clusters saved: ../models/clusters/{}_tags_clusters_model.pkl\n'.format(name))

# 对clusters进行分析
def clusters_analysis(clusters_path, ranks_list):
	name = os.path.basename(clusters_path).split('/')[-1].split('.')[0]
	print(name)
	clusters = ClusetrsSet.load(load_path=clusters_path)
	# clusters.clusters_in_reviews_rank(image_save_path='../results/{}_clusters_in_ranks.html'.format(name),
	# 								ranks_list=ranks_list)
	clusters.reviews_num_in_clusters(image_save_dir='../results/reviewsn_in_clusters/{}/'.format(name))




def double_cluster(data, image_save_path, n_clusters=4, method='kmeans'):
	"""
	:param data: [(beta0,sigma,avg)_1,...]
	"""
	size = len(data)
	# print(size) # 63
	from sklearn.preprocessing import StandardScaler
	from sklearn.cluster import KMeans
	from sklearn.cluster import DBSCAN
	from sklearn.metrics import silhouette_score
	import pandas as pd
	from mpl_toolkits.mplot3d import Axes3D

	fit_data = np.array(data)[:,1:-1]
	scale_model = StandardScaler().fit(fit_data)
	fit_data = scale_model.transform(fit_data)
	model1 = KMeans(n_clusters=n_clusters, init='k-means++', random_state=21).fit(fit_data)
	# model2 = DBSCAN(eps=0.3, min_samples=3).fit(fit_data)
	labels = model1.labels_
	score = silhouette_score(fit_data, labels, metric='euclidean')
	print("model:{} - score:{:.3f}".format(model1.__class__.__name__,score))
	def clusters_report(data,labels):
		cluster_nums, beta0s, avgs, count, sigmas = zip(*data)
		df = pd.DataFrame({
				'cluster_num': cluster_nums,
				'label': labels,
				'beta0': beta0s, 'sigma': sigmas, 'avg': avgs, 'count': count})

		for label in set(labels):
			tdf = df[df['label']==label]
			print('='*20,'label-{}'.format(label),'='*20)
			print('count:',len(tdf))
			print('beta0: mean={:.2f}, std={:.2f}, max={:.2f}, min={:.2f}'.format(
				tdf['beta0'].mean(), tdf['beta0'].std(), tdf['beta0'].max(), tdf['beta0'].min()))
			print('avg: mean={:.2f}, std={:.2f}, max={:.2f}, min={:.2f}'.format(
				tdf['avg'].mean(), tdf['avg'].std(), tdf['avg'].max(), tdf['avg'].min()))
			print('count: mean={:.2f}, std={:.2f}, max={:.2f}, min={:.2f}'.format(
				tdf['count'].mean(), tdf['count'].std(), tdf['count'].max(), tdf['count'].min()))
			print('clusters group:',tdf['cluster_num'].values,'\n')
		
		# ax = plt.subplot(1,1,1)
		# for l in set(labels):
		# 	tmp_df = df[df['label']==l]
		# 	ax.scatter(tmp_df['beta0'].values,
		# 				 tmp_df['avg'].values,
		# 				 label=l, cmap='Blues')

		fig = plt.figure()
		ax = Axes3D(fig)
		for l in set(labels):
			tmp_df = df[df['label']==l]
			ax.scatter(tmp_df['beta0'].values,
						tmp_df['avg'].values,
						tmp_df['count'].values,
						label=l, cmap='Blues')
		
		ax.set_xlabel('beta0')
		ax.set_ylabel('avg-reviews-num')	
		ax.set_zlabel('count')
		ax.view_init(elev=10,azim=None)
		plt.title('{} - {}'.format(method,n_clusters))
		plt.legend()
		plt.savefig(os.path.join(image_save_path))
		plt.show()

	clusters_report(data,labels)


# 得到最新文件
def get_latest_version(dir_name):
	files = sorted([(os.path.getmtime(os.path.join(dir_name,f)),f) for f in os.listdir(dir_name)])
	files = list(filter(lambda x:'.DS_Store' not in x,files))
	latest_file = os.path.join(dir_name,files[-1][1])
	return latest_file
	


if __name__ == '__main__':
	args_size = len(sys.argv)

	if sys.argv[1]=='1':
		# read_path = '../data/new_breakouts_text_100gg7'
		read_path = '../data/[200_15]proxied_breakouts_text'
		topk = 10
		name = sys.argv[2]
		comment = ""
		if args_size==4:
			comment = sys.argv[3]
		with open('../data/tags_pool_name.txt','a') as f:
			f.write("{} {} [min_reviews-{} min_gap-{} topk-{}] {}".format(
											str(datetime.today()).split('.')[0], name,
											200, 15, 10, comment))
		# if args_size==3:
		# 	topk = int(sys.argv[2])
		# if args_size==4:
		# 	topk, read_path = int(sys.argv[2]),sys.argv[3]
		save_path = '../data/tags_pool/{}.pkl'.format(name)

		extract_tags_corpus(read_path,save_path,topk=topk)

	if sys.argv[1]=='2':
		affinity = 0.65
		tags_path = get_latest_version('../data/tags_pool/')
		print("tags_path:",tags_path)
		w2v_model_path = '../models/word2vec/b1.mod'

		# 设定聚类模型的affinity
		if args_size==3:
			affinity = float(sys.argv[3])
		# 设定tags_pool来源
		if args_size==4:
			affinity,tags_path = float(sys.argv[3]),sys.argv[4]
		create_clusters(affinity, tags_path, w2v_model_path)

	if sys.argv[1]=='3':
		clusters_path = get_latest_version('../models/clusters/')
		if args_size==3:
			clusters_path = sys.argv[2]
		ranks_list = [(1,5),(5,10),(10,15),(15,30),(30,60),(60,1000)]
		clusters_analysis(clusters_path, ranks_list)

