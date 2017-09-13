import pickle
import os
from nltk.tokenize import word_tokenize


def pickle_me(p_path, p_obj):
	pickle_out = open(p_path, "wb")
	pickle.dump(p_obj, pickle_out)
	pickle_out.close()


def get_pickle(p_path):
	pickle_in = open(p_path,"rb")
	obj = pickle.load(pickle_in)
	pickle_in.close()
	return obj


def find_features(document, w_features):
		words = word_tokenize(document)
		features = {}
		for w in w_features:
			features[w] = (w in words)

		return features


def doc_process(f_name, input_docs, w_features):
	# process documents into featuresets
	currentPath = str(os.path.dirname(os.path.realpath(__file__)))
	rel_folder_path = r'\Pickles\\'
	pickle_path_fsets = (
		currentPath + rel_folder_path + f_name[:-4] + '_fsets.pickle')
	if not os.path.isfile(pickle_path_fsets):
		featuresets = (
			[(find_features(rev, w_features), category) for (rev, category) in input_docs])
		pickle_me(pickle_path_fsets, featuresets)
	else:
		featuresets = get_pickle(pickle_path_fsets)
	return featuresets