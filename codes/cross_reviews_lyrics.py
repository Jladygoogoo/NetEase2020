import os

reviewed_tracks = open('../data/have_reviews_tracks.txt').read().splitlines()
cross_tracks = set()

for root,dirs,files in os.walk('/Volumes/nmusic/music/all_lyrics'):
	for file in files:
		track_id = file[:-5]
		if track_id in reviewed_tracks:
			cross_tracks.add(track_id)

print(len(cross_tracks))
with open('../data/cross_reviews_lyrics_tracks.txt','w') as f:
	f.write('\n'.join(cross_tracks))


