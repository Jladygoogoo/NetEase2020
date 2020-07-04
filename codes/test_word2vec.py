from gensim.models import Word2Vec
import click
import numpy as np 

path = '../models/word2vec/abs_word2vec_2.5.mod'
model = Word2Vec.load(path)

def find_similar_words():
	while True:
		word = input("input: ")
		try:
			hits = model.wv.most_similar(word)
			print("similar words:")
			for p in hits:
				print(p)
		except:
			print("word not found.")

def calc_similarity():
	while True:
		w1,w2 = input("inputs: ").split()
		try:
			simi = model.wv.similarity(w1,w2)
			distance = np.linalg.norm(model.wv.__getitem__(w1) - model.wv.__getitem__(w2))
			print("similarity:", simi)
			print("distance:", distance)
		except:
			print("at least one word not found.")

@click.command()
@click.option('-m','--mode',prompt=True)
def main(mode):
	if mode=='search':
		find_similar_words()
	if mode=='compare':
		calc_similarity()

if __name__ == '__main__':
	main()

