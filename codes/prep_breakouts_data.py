import os
import re
import json
import sys
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import warnings
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

warnings.filterwarnings('ignore')


def std_timestamp(timestamp):
    timestamp = int (timestamp* (10 ** (10-len(str(timestamp)))))
    return timestamp


def assign_path(assigned_flag, prefix):
    dir_size = 100
    first = assigned_flag//(dir_size*dir_size)
    assigned_flag = assigned_flag%(dir_size*dir_size)
    second = assigned_flag//dir_size

    return '{}/{}/{}'.format(prefix, first, second)

def prep_data(filepath, filetype='json'):
    data_corpus = []
    if filetype=='txt':
        for line in open(filepath).read().splitlines():
            try:
                review_dict = json.loads(line)
                data = []
                # 时间戳 -> 13位转化为10位
                data.append(std_timestamp(review_dict['time']))
                # 被赞数目
                data.append(review_dict['likedCount'])
                # 评论长度
                data.append(len(review_dict['content']))
                # 评论内容
                data.append(review_dict['content'].replace('\r',' '))
                data_corpus.append(data)
            except:
                continue
    
    elif filetype=='json':
        with open(filepath) as f:
            content = json.load(f)
            for item in content:
                data = []
                # 时间戳 -> 13位转化为10位
                data.append(std_timestamp(item['time']))
                # 被赞数目
                data.append(item['likedCount'])
                # 评论长度
                data.append(len(item['content']))
                # 评论内容
                data.append(item['content'].replace('\r',' '))
                data_corpus.append(data)

    df = pd.DataFrame(data_corpus,columns=['time','liked_count','size','text'])
    df.drop_duplicates(['time','liked_count'],inplace=True)
    df['datetime'] = df['time'].map(datetime.fromtimestamp)
    df['date'] = df['datetime'].map(lambda x:datetime.strftime(x,'%Y-%m-%d'))
    df.drop(['time','datetime'],axis=1,inplace=True)

    # print(df.head())

    return df


def getEveryDay(begin_date,end_date):
    # 前闭后闭
    date_list = []
    begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += timedelta(days=1)
    return date_list


def get_sequence(df,timebase='date'):
    # if base=='month':
    # 	df['month'] = df['date'].map(lambda x:x[:-3])

    begin_date = df['date'].min()
    end_date = df['date'].max()
    dates_between = getEveryDay(begin_date,end_date)

    date2count_dict = dict(df['date'].value_counts())
    for date in dates_between:
        if date not in date2count_dict:
            date2count_dict[date] = 0
    counts = [x[1] for x in sorted(date2count_dict.items(),key=lambda x:x[0])]

    return counts, dates_between



def get_breakouts(filepath, min_reviews=100, min_start=30, group_gap=15, display=False):
    df = prep_data(filepath)
    reviews_count, dates = get_sequence(df)
    peaks_groups = peaks_detection(reviews_count,group_gap=group_gap)
    if not peaks_groups: 
        return None
    peaks_heads = [group[0] for group in peaks_groups]
    breakouts = []
    for p in peaks_heads:
        if min(p,len(dates)-p)>=min_start and reviews_count[p]>=min_reviews:
            breakouts.append(p)
    if len(breakouts)==0:
        return None

    if display:
        display_breakouts(reviews_count,breakouts,save=False)

    return (df, reviews_count, dates, breakouts)


def get_breakouts_text(df,dates,breakouts):
    btexts = []
    bdates = []
    for i in breakouts:
        date = dates[i]
        btext = df[df['date']==date]['text'].values
        btexts.append('\n'.join(btext))

    return btexts

# 单纯数量多
def get_big_text_asfiles(read_path,num):
    write_template = '/Volumes/nmusic/NetEase2020/data/big_text_{}/'.format(num)

    flag = 0
    dir1, dir2 = 1, 1
    write_path = write_template + '1/1/'
    print("current dir: {}".format(write_path))

    for file in os.listdir(read_path):
        if not os.path.exists(write_path): os.makedirs(write_path)

        if 'txt' not in file: continue
        try:
            df = prep_data(read_path+file)
            reviews_count, dates = get_sequence(df,'reviews_count')
            big_dates = np.arange(len(dates))[np.where(np.array(reviews_count)>=num)]

            # save breakouts text
            btexts = get_breakouts_text(df,dates,big_dates)
            for i,btext in enumerate(btexts,start=1):
                reviews_num = reviews_count[big_dates[i-1]]

                # print("file:{} - peak:{} - reviews_num:{}".format(file,i,reviews_num))
                with open(write_path+'{}_{}.txt'.format(file[:-4],i),'w') as f:
                    f.write(btext)
                    flag += 1
        
            if flag>=1000:
                dir2 += 1
                flag = 0

                if dir2 == 51:
                    dir1 += 1
                    dir2 = 1
                write_path = write_template + '{}/{}/'.format(dir1,dir2)
                print("current dir: {}".format(write_path))
        
        except KeyboardInterrupt:
            print("KeyboardInterrupt.")
            break

        except:
            print(traceback.format_exc())
            print("file:{} - FAILED".format(file))
            continue

# 爆点
def get_breakouts_text_asfiles(read_path, path_prefix, min_reviews, group_gap):
    
    flag = 0 # 记录遍历过的的歌曲数目
    count = 1 # 记录生成的breakouts_text文件数

    breakflag = 0
    for root,dirs,files in os.walk(read_path):
        for file in files:
            try:
                df = prep_data(os.path.join(root,file), filetype='json')
                reviews_count, dates = get_sequence(df,'reviews_count')

                peaks_group = peaks_detection(reviews_count,group_gap=group_gap)
                if not peaks_group: continue

                breakouts = []
                peaks_head = [group[0] for group in peaks_group]
                continue_flag = 0
                for p in peaks_head:
                    # 限制一段上下文的最少评论数
                    if p<=30: continue_flag=1;break
                    if reviews_count[p]>=min_reviews:
                        breakouts.append(p)
                if continue_flag: continue

                # save breakouts text
                btexts = get_breakouts_text(df,dates,breakouts)
                for i,btext in enumerate(btexts,start=1):
                    reviews_num = reviews_count[breakouts[i-1]]

                    print("[{}]file:{} - peak:{} -relevant_date:{} - reviews_num:{}".format(flag, file, i, peaks_head[i-1], reviews_num))

                    write_dir = assign_path(count, path_prefix)
                    if not os.path.exists(write_dir):
                        os.makedirs(write_dir)
                    write_path = os.path.join(write_dir, '{}_{}.txt'.format(file[:-4],i))
                    with open(write_path,'w') as f:
                        f.write(btext)

                    count += 1

                flag += 1

            except KeyboardInterrupt:
                print("Interrupted by keyboard.")
                breakflag = 1
                break
            except:
                print(file)
                print(traceback.format_exc())
                print("file:{} - FAILED".format(file))
                continue

        # interrupted by keyboard
        if breakflag:
            break



def breakouts_pucha(read_path, clusters_model_path):

    clusters_set = ClusetrsSet.load(clusters_model_path)
    name = clusters_model_path.split('/')[-1].split('.')[0]
    txt_write_path = '../results/pucha/{}.txt'.format(name)
    csv_write_path = '../results/pucha/{}.csv'.format(name)
    print(csv_write_path)

    with open(txt_write_path,'w') as f:
        flag = 0
        for root,dirs,files in os.walk(read_path):
            for file in files:
                # flag += 1
                # print(file,flag)
                try:
                    res = get_breakouts(os.path.join(root,file), min_reviews=100,group_gap=15)
                    if res:
                        df,reviews_count,dates,breakouts = res
                    else:
                        continue
                    btexts = get_breakouts_text(df,dates,breakouts)
                    btags = [tags_extractor(btext,topk=5) for btext in btexts]
                    cluster_numbers = [clusters_set.classify(bts) for bts in btags]
                    start_ends = [detect_start_end(reviews_count,b) for b in breakouts]
                    starts, ends = zip(*start_ends)

                    for i,b in enumerate(breakouts, start=1):
                        data = (file,str(i),dates[starts[i-1]], dates[ends[i-1]], dates[b],
                                str(reviews_count[b]), str(btags[i-1]), str(cluster_numbers[i-1]))
                        f.write('  '.join(data)+'\n') # 2*' '
                except:
                    print(traceback.format_exc())

    data_corpus = open(txt_write_path).read()
    data_corpus = data_corpus.replace(', ', ',').splitlines()
    data_corpus = list(map(lambda l:l.split(),data_corpus))
    df = pd.DataFrame(data_corpus, columns=['file','flag','start_date','end_date','center',
                                            'reviews_count','tags','cluster_number'])
    df.to_csv(csv_write_path, index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    ### 使用资源
    model_path = '../models/word2vec/a.mod'
    # 评论信息
    read_path = '/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews' 
    # 日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', 
                    datefmt='%H:%M:%S', filename='./logs/get_proxied_breakouts_text.log')
    
    ### 单条测试
    test_path = '/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews/0/29/4330176.json'
    # get_breakouts(test_path)


    ### 提取爆发点的文本信息
    # 最少评论数为200
    # 组间最小间隔为15天（中心）
    min_reviews = 200
    group_gap = 15
    path_prefix = '../data/[{}_{}]proxied_breakouts_text'.format(min_reviews, group_gap)
    # get_breakouts_text_asfiles(read_path=read_path, path_prefix=path_prefix,
    #                          min_reviews=min_reviews, group_gap=group_gap)

    ### 生成pucha文件 -> 统计全局
    # 上一个是gg15
    clusters_model_path = '../models/clusters/BorgCube2_65b1.pkl'
    breakouts_pucha(read_path, clusters_model_path)

