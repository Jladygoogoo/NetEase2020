import os
import json
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import warnings

import numpy as np 
import pandas as pd 
import ruptures as rpt

from my_decorators import time_record

warnings.filterwarnings('ignore')
# sys.stdout = open('../breakouts_keywords.txt','w')



def raw_breakouts_detection(sequence,dates,thres=10):
	breakouts = []
	for i in range(len(sequence)-1):
		if sequence[i+1]/sequence[i]>thres:
			breakouts.append(i+1)

	print(breakouts)

	plt.plot(np.cumsum(sequence))
	for i in breakouts:
		plt.axvline(x=i,ls='--',color='red')

	show_xticks = np.arange(0,len(sequence),200)
	plt.xticks(show_xticks,np.array(dates)[show_xticks])
	plt.xticks(size='small',rotation=30)

	# plt.tight_layout()
	plt.show()

def R_breakouts_detection(points):
	#Changepoint detection with the Pelt search method
	model="rbf"
	algo = rpt.Pelt(model=model).fit(points)
	result = algo.predict(pen=10)
	rpt.display_breakouts(points, result, figsize=(10, 6))
	plt.title('Change Point Detection: Pelt Search Method')
	plt.tight_layout()
	plt.show()  
	    
	#Changepoint detection with the Binary Segmentation search method
	model = "l2"  
	algo = rpt.Binseg(model=model).fit(points)
	my_bkps = algo.predict(n_bkps=10)
	# show results
	rpt.show.display_breakouts(points, my_bkps, figsize=(10, 6))
	plt.title('Change Point Detection: Binary Segmentation Search Method')
	plt.tight_layout()
	plt.show()


def peaks_detection(sequence,k=5,bh=3,ch=1000,method='mean',group_gap=15):
	'''
	param: k: the number of neighbors included
	param: bh/ch: hyper-parameters for peaks identification, b for bottom/c for ceiling
	param: method: 'mean' or 'max'
	return: a list of peaks
	'''
	func_max = lambda left,right: (max(left)+max(right))/2
	func_mean = lambda left,right: np.sum(left+right)/(2*k)
	def func_filter(value,mean,std):
		if value-mean >= bh*std and value-mean <= ch*std:
			return 1
		return 0

	measures = {}
	for i in range(1,len(sequence)-1): # 不考虑两个端点
		start = max(0,i-k)
		end = min(i+k,len(sequence)-1)
		left = [sequence[i]-sequence[j] for j in range(start,i)]
		right = [sequence[i]-sequence[j] for j in range(i+1,end)]
		if method=='mean':
			si = func_mean(left,right)
		elif method=='max':
			si = func_max(left,right)
		measures[i] = si 

	values = list(measures.values())
	mean,std = np.mean(values), np.std(values)
	peaks = list(filter(lambda x:func_filter(x[1],mean,std),list(measures.items())))

	if len(peaks)==0: return None

	# 将临近的peaks合为一个group
	sorted_peaks = list(sorted([x[0] for x in peaks],key=lambda x:sequence[x],reverse=True))
	peaks_group = [[sorted_peaks[0]]]
	peaks_group_head = [sorted_peaks[0]]
	for p in sorted_peaks[1:]:
		join_flag = 0
		for i in range(len(peaks_group_head)):
			if abs(p-peaks_group_head[i])<group_gap:
				peaks_group[i].append(p)
				join_flag = 1
				break 
		if not join_flag:
			peaks_group.append([p])
			peaks_group_head.append(p)

	peaks_group = sorted(peaks_group,key=lambda l:l[0])

	# print("number of peaks: {}".format(len(peaks)))
	# print("number of peaks_group: {}".format(len(peaks_group))
	return peaks_group


def detect_start_end(sequence,b,window_size=15):
	try:
		ori_sequence = sequence
		sequence = sequence[b-window_size:b+window_size+1]
		ori_b = b
		b = window_size

	except: 
		return

	max_former,max_former_point = 0,0
	for i in range(window_size):
		dis = i*sequence[b] - b*sequence[i]
		if dis>max_former: 
			max_former = dis
			max_former_point = i

	max_after,max_after_point = 0,0
	for j in range(window_size+1,2*window_size+1):
		dis = (2*window_size+1-j)*sequence[b] - (2*window_size+1-b)*sequence[j]
		if dis>max_after: 
			max_after = dis
			max_after_point = j

	# display_breakouts(ori_sequence,(ori_b-window_size+max_former_point, ori_b, ori_b-window_size+max_after_point),save=False)

	return (ori_b-window_size+max_former_point, ori_b-window_size+max_after_point)



def display_breakouts(sequence,points,title='peak-heads detection',save=True,show=True,save_path=None):
	fig = plt.figure(figsize=(16,8))
	plt.plot(sequence)
	for p in points:
		plt.plot(p,sequence[p],marker='x')
	plt.title(title)
	if save and save_path:
		plt.savefig(save_path)
	if show:
		plt.show()



if __name__ == '__main__':
	# try:
	# 	run()
	# except KeyboardInterrupt:
	# 	print('aborted by user.')
	test()