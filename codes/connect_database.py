import os
import pymysql
import pickle


def get_track_partners(conn,track_id):
	cursor = conn.cursor()
	sql = 'SELECT playlist_id FROM tracks WHERE id=%s'
	cursor.execute(sql,(track_id,))
	playlists = cursor.fetchone()
	if playlists:
		playlists = playlists[0]

	playlist_2_tracks = {}
	for p in playlists.split(','):
		sql = 'SELECT tracks FROM playlists WHERE id=%s'
		cursor.execute(sql,(p,))
		tracks = cursor.fetchone()[0]
		playlist_2_tracks['p'] = tracks.split(',')+[track_id]
	return playlist_2_tracks


def assign_track_playlist_id(conn):
	cursor = conn.cursor()
	cursor.execute('SELECT COUNT(id) FROM playlists')
	playlists_count = cursor.fetchone()[0]
	print(playlists_count)

	for i in range(int(playlists_count//100)):
		sql = 'SELECT id,tracks FROM playlists LIMIT %s,%s'
		cursor.execute(sql,(i*100,(i+1)*100))
		res = cursor.fetchall()
		for pid,r in res:
			tracks_id = r.split(',')
			# print(tracks_id)
			for tid in tracks_id:
				tid = tid[1:-1]
				sql = 'UPDATE tracks SET playlist_id = CONCAT(playlist_id,%s) WHERE id=%s'
				try:
					cursor.execute(sql,(str(pid)+',',tid))
				except:
					continue
		conn.commit()
		print('[success] processed {} playlists'.format((i+1)*100))
	cursor.close()

def main():
	conn = pymysql.connect(host='127.0.0.1',port=3306,user='root',password='SFpqwnj285798,.',db='NetEase')
	print('[success] connected to db.')
	assign_track_playlist_id(conn)

	# reviewed_tracks = open('../data/tracks_list/have_reviews_tracks.txt').read().splitlines()

	# refined_playlists_dict = {}
	# for track_id in reviewed_tracks:
	# 	for playlist,tracks in get_track_partners(conn,track_id).items():
	# 		if playlist in refined_playlists_dict: continue
	# 			tracks = set(tracks).intersection(set(reviewed_tracks))
	# 			refined_playlists_dict[playlist] = list(tracks)
	# 			print('{}[{}]: {}'.format(playlist,len(tracks),tracks))

	# with open('../data/tracks_list/reviewed_playlists.pkl','wb') as f:
	# 	pickle.dump(refined_playlists_dict,f)
		
if __name__ == '__main__':
	main()


