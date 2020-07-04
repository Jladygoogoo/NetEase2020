import re 
import os

raw_kaomojis_path = '../resources/kaomojis.txt'
raw_kaomojis = open(raw_kaomojis_path).read()
path = '../data/raw_reviews_content/'

# 利用粗糙的颜文字集获取单位符号
chars = set()
for s in raw_kaomojis:
	if s not in ('\n',''):
		chars.add(s)
chars.remove('！')
# 重新排序，防止range错误
chars = ''.join(list(sorted(chars,key=ord)))

# 处理re中需要转义的符号
escapes = '* . ? + $ ^ [ ] ( ) { } | / \''.split()
for e in escapes:
	if e in chars:
		chars = chars.replace(e,r'\\'+e)

re_kaomoji = re.compile('['+chars+']{5,}')


def search_kaomojis(text):
	kaomojis = set()
	mojis = re.findall(re_kaomoji,text.lower())
	for m in mojis:
		if re.match(r'.*[\da-z].*',m): continue
		# if m[0]=='[' or m[-1]==']': continue
		
		conti = 0
		for c in m:
			if m.count(c)>=3: 
				conti = 1
				break
		if conti: continue

		kaomojis.add(m)
	return kaomojis



def test():
	all_kaomojis = []
	flag = 0
	for file in os.listdir(path):
		print(file)
		if 'txt' in file:
			kaomojis = search_kaomojis(open(path+file).read())
			all_kaomojis.extend(kaomojis)

			flag += 1
			if flag>10: break

		for k in kaomojis:
			print(k)

	# all_kaomojis = set(all_kaomojis)
	# print(len(all_kaomojis))
	# for k in all_kaomojis:
	# 	print(k)


if __name__ == '__main__':
	test()
	# text = '⊙∀⊙'
	# text = text.lower()
	# # 处理emojis
	# re_emoji = re.compile(u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]',re.UNICODE)
	# text = re.sub(re_emoji,'',text)
	# print(search_kaomojis(text))

