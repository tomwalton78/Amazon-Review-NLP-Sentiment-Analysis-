import pickle
import os
import re
from nltk.corpus import stopwords
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import common_functions


def get_initial_2_lists(path, min_pos_score):
	pos_texts = []
	neg_texts = []
	with open(path, 'r') as f:
		line = f.readline()
		line = f.readline()  # ignores 1st line
		while line:
			split_line = line.split('\t')
			split_line = [x.replace('\n', '') for x in split_line]
			text = split_line[0]
			score = float(split_line[1].replace(' ', ''))
			if score >= min_pos_score:
				pos_texts.append(text)
			else:
				neg_texts.append(text)
			line = f.readline()
	return pos_texts, neg_texts


def import_and_process(f_name, min_pos_score, max_wf):
	currentPath = str(os.path.dirname(os.path.realpath(__file__)))
	rel_folder_path = r'\Datasets\\'
	full_path = currentPath + rel_folder_path + f_name
	pos_texts, neg_texts = get_initial_2_lists(full_path, min_pos_score)

	# Ensures equal no. of pos and neg reviews to train against, to avoid bias
	pos_texts = pos_texts[:len(neg_texts)]
	print('length of negative texts: ' + str(len(neg_texts)) + '\n')

	all_words = []
	documents = []

	#  j is adject, r is adverb, and v is verb
	allowed_word_types = ["J", "R", "V"]
	# allowed_word_types = ["J"]

	try:
		for p in pos_texts:
			documents.append((p, "pos"))
			words = word_tokenize(p)
			pos = nltk.pos_tag(words)
			for w in pos:
				if w[1][0] in allowed_word_types:
					all_words.append(w[0].lower())
	except(LookupError):
		nltk.download()

	for p in neg_texts:
		documents.append((p, "neg"))
		words = word_tokenize(p)
		pos = nltk.pos_tag(words)
		for w in pos:
			if w[1][0] in allowed_word_types:
				all_words.append(w[0].lower())

	all_words = nltk.FreqDist(all_words)
	word_features = list(all_words.keys())[:max_wf]
	return word_features, documents


def process(f_name, min_pos_score, max_word_features=5000):
	# processes data in similar way to sentdex tutorial
	currentPath = str(os.path.dirname(os.path.realpath(__file__)))
	rel_folder_path = r'\Pickles\\'
	pickle_path_doc = currentPath + rel_folder_path + f_name[:-4] + '_doc.pickle'
	pickle_path_wf = currentPath + rel_folder_path + f_name[:-4] + '_wf.pickle'
	if not os.path.isfile(pickle_path_doc) or not os.path.isfile(pickle_path_wf):
		try:
			os.remove(pickle_path_doc)
		except(FileNotFoundError):
			pass
		try:
			os.remove(pickle_path_wf)
		except(FileNotFoundError):
			pass
		word_features, documents = import_and_process(
			f_name, min_pos_score, max_word_features)
		common_functions.pickle_me(pickle_path_doc, documents)
		common_functions.pickle_me(pickle_path_wf, word_features)
	else:
		documents = common_functions.get_pickle(pickle_path_doc)
		word_features = common_functions.get_pickle(pickle_path_wf)
	return word_features, documents


if __name__ == '__main__':
	wf, doc = process('Musical_Instruments_5.tsv', 4, 5000)
