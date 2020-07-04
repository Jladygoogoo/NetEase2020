import os
import re
import json


def generate_lyrics(path,timemark=False):
	with open(path) as f:
		content = json.load(f)['lrc']['lyric']
		lines = content.split('\n')

	timemark_r = r'\[\d{2}:\d{2}[:.]*\d{0,3}\]'

	if re.search(timemark_r,content):
		tmark_text = []
		for l in lines:
			tmarks = re.findall(timemark_r,l)

			if len(tmarks)==0: continue
			text = re.search(r'\]([^\[\]]+)',l)
			if text:
				text = text.group(1).strip()
				for tmark in tmarks:
					tmark_text.append((tmark,text))
		tmark_text = sorted(tmark_text,key=lambda x:x[0])

		tmarks = [p[0] for p in tmark_text]
		lyrics = [p[1] for p in tmark_text]
	else:
		tmarks = None
		lyrics = lines

	if timemark:
		return tmarks,lyrics
	else:
		return list(filter(None,lyrics))


def run(read_path):
	for root,dirs,files in os.walk(read_path):
		for file in files:
			path = os.path.join(root,file)
			

def test(test_path):
	lyrics = generate_lyrics(test_path)
	print(lyrics)


if __name__ == '__main__':
	test_path = '/Volumes/nmusic/music/all_lyrics/1/28/475254755.json'
	read_path = '/Volumes/nmusic/music/all_lyrics/1/1/'
	test(test_path)
	# run(read_path)