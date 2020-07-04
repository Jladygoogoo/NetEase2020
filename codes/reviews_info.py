import os
import re
import json
import time

import pandas as pd 
import numpy

import matplotlib.pyplot as plt 
import seaborn as sns

path = '/Volumes/nmusic/music/topsongs-reviews/reviews/'
new_path = '../data/raw_reviews_content/'


def extract_reviews(file):
	content = []
	for line in open(path+file).read().splitlines():
		try:
			content.append(json.loads(line)['content'])
		except:
			print(line)
	reviews_size.append(len(content))
	print("song:{}, size:{}".format(file[:-4],len(content)))
	with open(new_path+file,'w') as f:
		f.write('\n'.join(content))


def review_size_distribution():
	reviews_text_size = []
	for file in os.listdir(new_path):
		reviews_size.append(len(' '.join(open(new_path+file).read().splitlines()))/1000)

	plt.figure(figsize=(8,8))
	sns.distplot(reviews_size,bins=30,color='#fe828c')
	plt.xlabel('reviews_text_size(k)')
	plt.savefig('../images/reviews_text_size_distribution.png')







