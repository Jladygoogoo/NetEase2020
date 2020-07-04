import os
import shutil
import traceback

read_path = '/Volumes/nmusic/music/allData_lyrics/'

# 一级目录动态增长，二级目录文件数50个，三级500个
flag = 1
top = 1
print("top:",1)
for file in os.listdir(read_path):
	# 一级目录动态增长
	if flag>top*8000: 
		top+=1
		print("top:",top)
	cur = flag%8000
	second = int((cur-1)/200)+1
	write_path = '/Volumes/nmusic/music/all_lyrics/{}/{}/'.format(top,second)
	if not os.path.exists(write_path):
		os.makedirs(write_path)
	try:
		shutil.move(read_path+file,write_path+file)
		flag += 1
	except:
		print(traceback.exc_format())
