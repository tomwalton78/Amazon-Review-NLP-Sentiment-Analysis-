import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from nltk.tokenize import word_tokenize
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
import data_importer_v3 as data_importer
import common_functions

# Gets rid of some warnings in output text
import warnings
warnings.filterwarnings("ignore")

file_name = 'Musical_Instruments_5.tsv'

light_mode = False  # True when only using pre-trained classifiers,
# and don't need to re-evaluate accuracy
min_score_for_pos = 4
percent_train = 90
classifiers_to_use = [
	#'NaiveBayesClassifier',
	'MultinomialNB',
	#'BernoulliNB',
	'LogisticRegression',
	'LinearSVC',
	'SGDClassifier']


def tt_split(input_data, percent_tr):
	# train, test split
	change_index = (
		len(input_data) -
		int(len(input_data) * (1 - (percent_tr / 100.0))))
	tr_set = input_data[:change_index]
	te_set = input_data[change_index:]
	return tr_set, te_set


def try_train_classifier(f_name, c_name, tr_set, te_set):
	currentPath = str(os.path.dirname(os.path.realpath(__file__)))
	rel_folder_path = r'\Pickles\\'
	pickle_path_c = (
		currentPath + rel_folder_path + f_name[:-4] + '_' + c_name + '.pickle')
	if not os.path.isfile(pickle_path_c):
		if c_name == 'NaiveBayesClassifier':
			classifier = nltk.NaiveBayesClassifier.train(tr_set)
		else:
			classifier = eval('SklearnClassifier(' + c_name + '())')
			classifier.train(tr_set)
		common_functions.pickle_me(pickle_path_c, classifier)
		accuracy = round(float(nltk.classify.accuracy(classifier, te_set)) * 100, 1)
	else:
		classifier = common_functions.get_pickle(pickle_path_c)
		accuracy = None
	# accuracy = round(float(nltk.classify.accuracy(classifier, te_set)) * 100, 1)
	return classifier, accuracy


def try_train_all_classifiers(f_name, input_c_names, tr_set, te_set):
	c_list = []
	currentPath = str(os.path.dirname(os.path.realpath(__file__)))
	rel_folder_path = r'\Pickles\\'
	pickle_path_acc = (
		currentPath + rel_folder_path + f_name[:-4] + '_acc_dict.pickle')
	if not os.path.isfile(pickle_path_acc):
		acc_dict = {}
	else:
		acc_dict = common_functions.get_pickle(pickle_path_acc)
		os.remove(pickle_path_acc)
	for i in range(len(input_c_names)):
		c, acc = try_train_classifier(f_name, input_c_names[i], tr_set, te_set)
		c_list.append(c)
		if acc is not None:
			acc_dict[input_c_names[i]] = acc
	acc_dict_mod = acc_dict

	common_functions.pickle_me(pickle_path_acc, acc_dict_mod)
	return c_list, acc_dict_mod


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		return mode(votes)

	def classify_w(self, features, acc_weights):
		# version of classify with votes weighted by accuracy of algo
		votes = []
		for i in range(len(self._classifiers)):
			v = self._classifiers[i].classify(features)
			if v == 'pos':
				votes.append(acc_weights[i])
			elif v == 'neg':
				votes.append(acc_weights[i] * -1)
			else:
				print("error in v")
		pos_score, neg_score = 0, 0
		for vote in votes:
			if vote >= 0:
				pos_score += vote
			elif vote < 0:
				neg_score += -1 * vote
		if pos_score > neg_score:
			return 'pos'
		elif neg_score > pos_score:
			return 'neg'
		else:
			print("ERROR: neg, pos scores are equal")
			return 'pos'

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		if votes.count('pos') == votes.count('neg'):
			return 0.5
		else:
			choice_votes = votes.count(mode(votes))
			conf = choice_votes / len(votes)
		return conf


def generate_class_inp_str(c_list):
	result_string = ''
	for i in range(len(c_list)):
		result_string += 'classifiers[' + str(i) + ']'
		if i != len(c_list) - 1:
			result_string += ', '
	return result_string


def find_features(document, w_features):
		words = word_tokenize(document)
		features = {}
		for w in w_features:
			features[w] = (w in words)
		return features


word_features, documents = data_importer.process(
	file_name, min_score_for_pos, max_word_features=5000)

if light_mode:
	training_set, testing_set = [], []
else:
	featuresets = common_functions.doc_process(
		file_name, documents, word_features)
	random.shuffle(featuresets)

	training_set, testing_set = tt_split(featuresets, percent_train)


classifiers, classifier_accuracies = try_train_all_classifiers(
	file_name, classifiers_to_use, training_set, testing_set)
print('\n', classifier_accuracies)
acc_values = [classifier_accuracies[key] for key in classifier_accuracies.keys() if key in classifiers_to_use]
print(acc_values)

voted_classifier = eval(
	'VoteClassifier(' + generate_class_inp_str(classifiers) + ')')


def sentiment(text):
	feats = common_functions.find_features(text, word_features)
	# return voted_classifier.classify(feats), voted_classifier.confidence(feats)
	return voted_classifier.classify_w(feats, acc_values), voted_classifier.confidence(feats)

if __name__ == '__main__':
	from time import time as tt
	t_1 = tt()
	print(sentiment("good item, handy tool, would recommend"))
	print(sentiment("useless, breaky easily, avoid!"))
	print(tt() - t_1, ' s')
