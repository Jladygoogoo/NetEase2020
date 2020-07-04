import os
import re
import click

import requests
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

import pymysql


def get_track_name(host,user,password):
	cross_tracks = open('../data/cross_tracks.txt').read().splitlines()

	conn = pymysql.connect(host=host,user=user,password=password,database='NetEase')
	cursor = conn.cursor()

	id2name = {}
	for t in cross_tracks:
		sql = 'SELECT name FROM tracks WHERE id={}'.format(t)
		cursor.execute(sql)
		name = cursor.fetchall()[0][0]
		id2name[t] = name

	cursor.close()
	conn.close()
	print(len(id2name))

	return id2name

def download_mp3(track_id,name):
	print("start??")
	name = name.replace(' ','%20')
	bridge_url = 'http://music.ifkdy.com/?name={}&type=netease'.format(name)
	html = requests.get(bridge_url).text

	soup = BeautifulSoup(html,'lxml')
	urlsoups = soup.find(attrs={'class':'am-g am-margin-bottom-sm'}).find_all('a')
	print(urlsoups)
	urls = []
	for s in urlsoups:
		urls.append(s.get('href'))

	neturl,mp3url = urls[0],urls[1]
	print(neturl)
	hit_id = re.search(r'.+id=(\d+)',neturl).group(1)
	if hit_id!=track_id:
		print("fail to match {}-{}".format(track_id,name))
		return None
	try:
		urlretrieve(mp3url,'../data/mp3/{}.mp3'.format(track_id))
	except:
		print("fail to retrieve {}-{}".format(track_id,name))


@click.command()
@click.option(
	'-h','--host',
	default='127.0.0.1',
	help='host name for login mysql')
@click.option(
	'-u','--user',
	help='user name for login mysql',
	prompt=True)
@click.option(
	'-p','--password',
	help='password for login mysql',
	prompt=True,
	hide_input=True)
def run(host,user,password):
	id2name = get_track_name(host,user,password)
	for track_id,name in id2name.items():
		print(track_id,name)
		download_mp3(track_id,name)
		break


def simple_run():
	cross_tracks = open('../data/cross_tracks.txt').read().splitlines()
	for t in cross_tracks:
		mp3url = 'http://music.163.com/song/media/outer/url?id={}.mp3'.format(t)
		try:
			urlretrieve(mp3url,'../data/mp3/{}.mp3'.format(t))
			print("download {} successfully".format(t))
		except:
			print("fail to retrieve {}".format(t))


if __name__ == '__main__':
	simple_run()

	






