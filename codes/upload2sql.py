import os
import re
import json
import time
import traceback
import warnings 
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy

import pymysql
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', 
					password='SFpqwnj285798,.', db='NetEase')
cursor = conn.cursor()

# import matplotlib.pyplot as plt 
# import seaborn as sns

path = '/Volumes/nmusic/music/playlists/'
files = os.listdir(path)
playlists_data = []
tracks_data = []
albums_data = []
artists_data = []

def extract_info(content):
	playlist_info = []
	tracks = []
	# playlist
	for c in ('id','name','tags'):
		if c=='tags':
			playlist_info.append(str(tuple(content[c])))
		else:
			playlist_info.append(str(content[c]))
	playlist_info.append(content['creator']['userId'])

	def extract_track_album_artist(content):
		track_info = []
		album_info = []
		artists_info = []

		# track
		for c in ('id','name','duration','score','popularity'):
			if c=='id': tracks.append(str(content[c]))
			track_info.append(str(content[c]))
		track_info.append(str(content['album']['id'])) #补充'album'信息

		# artists
		for artist in content['album']['artists']:
			artists_info.append((str(artist['id']),artist['name']))

		# album
		for c in ('id','size','tags'):
			if c=='tag':
				album_info.append(str(tuple(content['album'][c])))
			else:
				album_info.append(str(content['album'][c]))
		album_info.append(str(tuple(int(x[0]) for x in artists_info))) #补充'artists'信息

		tracks_data.append(tuple(track_info))
		albums_data.append(tuple(album_info))
		artists_data.extend(tuple(artists_info))

	for c in content['tracks']:
		extract_track_album_artist(c)

	# playlist补充track信息
	length = min(len(tracks),400)
	playlist_info.append(str(tuple(tracks)[:length]))
	playlists_data.append(tuple(playlist_info))



flag = 0
t = time.time()
def spendtime(duration):
	m,s = divmod(duration,60)
	h,m = divmod(m,60)
	print("time duration - {:02f}:{:02f}:{:02f}".format(h,m,s))

for file in files:
	try:
		with open(path+file) as f:
			content = json.load(f)
		if type(content)==int: continue
		extract_info(content)
		flag += 1
	except:
		print(traceback.format_exc())

	# 将数据上传至数据库
	if flag%100==0:

		playlists_data = set(playlists_data)
		sql = 'INSERT IGNORE INTO playlists VALUES(%s,%s,%s,%s,%s)'
		cursor.executemany(sql,playlists_data)
		print("upload {} playlists to database.".format(len(playlists_data)))

		tracks_data = set(tracks_data)
		sql = 'INSERT IGNORE INTO tracks VALUES(%s,%s,%s,%s,%s,%s)'
		cursor.executemany(sql,tracks_data)
		print("upload {} tracks to database.".format(len(tracks_data)))

		albums_data = set(albums_data)
		sql = 'INSERT IGNORE INTO albums VALUES(%s,%s,%s,%s)'
		cursor.executemany(sql,albums_data)
		print("upload {} albums to database.".format(len(albums_data)))

		artists_data = set(artists_data)
		sql = 'INSERT IGNORE INTO artists VALUES(%s,%s)'
		cursor.executemany(sql,artists_data)
		print("upload {} artists to database.".format(len(artists_data)))

		print()
		conn.commit()
		

		playlists_data = []
		tracks_data = []
		albums_data = []
		artists_data = []

cursor.close()
conn.close()
spendtime(time.time()-t)

