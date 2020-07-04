import os
from breakouts_detection import *
from breakouts_analysis import *
from preprocess import tags_extractor
from tags_analysis import ClusetrsSet



def breakouts_influence_data(read_path,window_size,min_reviews):
	write_path = '../data/breakouts_influence_w{}min{}/'.format(window_size,min_reviews)
	if not os.path.exists(write_path): os.makedirs(write_path)

	breakouts_info = []
	for file in os.listdir(read_path):
		if 'txt' not in file: continue
		try:
			df = prep_data(read_path+file)
			reviews_count,dates = get_sequence(df,'reviews_count')

			peaks_group = peaks_detection(reviews_count,group_gap=window_size)
			if not peaks_group: continue

			breakouts = []
			peaks_head = [group[0][0] for group in peaks_group]
			for p in peaks_head:
				# 限制一段上下文的最少评论数
				if p<=window_size or len(reviews_count)-p<=window_size: continue
				if reviews_count[p]>=min_reviews:
					breakouts.append(p)

			flag = 0
			for b in breakouts:
				if b<window_size or len(dates)-1-b<window_size: continue
				b_start, b_end = detect_start_end(reviews_count,b,window_size)
				print(b_start,b,b_end)
				breakout = reviews_count[b_start:b_end+1]
				former = [reviews_count[b_start-i] for i in range(window_size,0,-1)]
				after = [reviews_count[b_end+i] for i in range(1,window_size+1)]

				def identify_vaccum(l):
					if 0 not in l or len(l)<=5: return 0
					i0 = l.index(0)
					if l[i0:i0+5]==[0]*5: 
						return 1
					else:
						return identify_vaccum(l[i0+1:])
				if identify_vaccum(former) or identify_vaccum(after):
					continue

				data = []
				for i,value in enumerate(former):
					data.append((dates[b_start-window_size-i],value,'former'))
				for i,value in enumerate(breakout):
					data.append((dates[b_start+i],value,'breakout'))		
				for i,value in enumerate(after):
					data.append((dates[b_end+i+1],value,'after'))

				flag += 1
				df = pd.DataFrame(data,columns=['date','value','attr'])
				df.to_csv(os.path.join(write_path, '{}-{}.csv'.format(file[:-4],flag)), index=False)
				

		except:
			print(traceback.format_exc())


def breakouts_influence_garch_results(read_path1,read_path2,min_reviews):
	data = []
	for file in os.listdir(read_path2):
		if 'DS' in file: continue

		breakout_details = {'special_type':[],'special_values':[]}

		content = open(os.path.join(read_path2,file)).read()
		try:
			start = content.index('mu')
			end = content.index('Robust')
		except:
			continue

		print(file)
		content = content[start:end].splitlines()[:-1]
		params = list(map(lambda x:[x.split()[i] for i in (0,1,2,4)],content))
		
		mureg1 = params[2]
		vxreg1 = params[-1]


		if float(mureg1[1])>=10 and float(mureg1[3])<=0.05:
			breakout_details['special_type'].append('mu')
			breakout_details['special_values'].extend(mureg1)
		if float(vxreg1[3])<=0.05:
			breakout_details['special_type'].append('vx')
			breakout_details['special_values'].extend(vxreg1)

		if len(breakout_details['special_type'])==0: continue

		
		tmp_df = pd.read_csv(os.path.join(os.path.join('/'.join(read_path2.split('/')[:-2]),file[:-4]+'.csv')))
		breakout_date = tmp_df[(tmp_df['attr']=='breakout') & (tmp_df['value']>=min_reviews)]['date'].values[0]
		# print(breakout_date)

		file,flag = file[:-4].split('-')
		breakout_details['file'], breakout_details['flag'] = file,flag
		df = prep_data(os.path.join(read_path1,file+'.txt'))
		reviews_count,dates = get_sequence(df)
		breakout = dates.index(breakout_date)
		breakout_text = get_breakouts_text(df,dates,[breakout])

		breakout_details['breakout_date'] = breakout_date
		breakout_details['breakout_num'] = reviews_count[breakout]
		breakout_details['breakout_tags'] = tags_extractor(breakout_text[0],topk=10)

		data.append(list(breakout_details.values()))

	df = pd.DataFrame(data,columns=['special_type','special_values','file','flag','breakout_date','breakout_num','breakout_tags'])
	write_path = '/'.join(read_path2.split('/')[:-2])+'.csv'

	df.to_csv(write_path,index=False)


def breakouts_influence_tags_cluster(read_path,w2v_model_path):
	df = pd.read_csv(read_path)
	model_params = read_path[:-4].split('_')[-1]

	mu_tags_raw = df[df['special_type']=="['mu']"]['breakout_tags']
	mu_tags = []
	for x in df['breakout_tags'].values:
		tags = eval(x)
		for t in tags:
			mu_tags.append((t,0))
	mu_clusters_set = ClusetrsSet(model_path=w2v_model_path,affinity=0.6)
	mu_clusters_set.growing(mu_tags,save_path='../results/breakouts_influence/{}_mu_tags_clusters.txt'.format(model_params))

	# vx_tags_raw = df[df['special_type']=="['vx']"]['breakout_tags']
	# vx_tags = []
	# for x in vx_tags_raw:
	# 	tags = eval(x)
	# 	for t in tags:
	# 		vx_tags.append((t,0))
	# vx_clusters_set = ClusetrsSet(model_path=w2v_model_path)
	# vx_clusters_set.growing(vx_tags,save_path='../results/breakouts_influence/{}_vx_tags_clusters.txt'.format(model_params))	


if __name__ == '__main__':
	read_path1 = '/Volumes/nmusic/music/topsongs-reviews/reviews/'
	w2v_model_path = '../models/word2vec/abs_word2vec_nnn2.mod'
	
	# breakouts_influence_data(read_path1,window_size=30,min_reviews=300)
	read_path2 = '../data/breakouts_influence_w30min300/R_results/'
	# breakouts_influence_garch_results(read_path1,read_path2,min_reviews=300)
	# breakouts_influence_tags_cluster('../data/breakouts_influence_w30min300.csv',w2v_model_path)
	df = pd.read_csv('../data/breakouts_influence_w30min300.csv')
	reviews_count = df['breakout_num'].values
	plt.hist(reviews_count)
	plt.title('significant mureg - reviews num distribution')
	plt.xlabel('reviews num')
	plt.ylabel('count')
	plt.savefig('../images/signi_mureg_reviews_num_hist.png')
	plt.show()




