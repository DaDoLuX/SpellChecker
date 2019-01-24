import math
import _pickle as cPickle
import numpy as np
import os

import fuzzy
from weighted_levenshtein import dam_lev

class SpellChecker:

	def __init__(self,
				 word_set,
				 unigrams,
				 k,
				 costs=None,
				 lamda=1,
				 alphabet='abcdefghijklmnopqrstuvwxyz'):

		# Initialize alphabet
		self.alphabet = alphabet

		# Store all known words
		self.dict_words = word_set

		# Build and store valid prefixes
		self.valid_prefixes = set([])
		for word in self.dict_words:
			for i in range(len(word)+1):
				self.valid_prefixes.add(word[:i])

		# Weighting likelihood & prior
		self.lamda = lamda

		# Store unigram probabilities
		self.priors = {}
		self.k = k
		self.N = sum((count for word, count in unigrams)) + k*len(unigrams) + k
		for word, count in unigrams:
			self.priors[word] = math.log(float(count+k) / self.N)

		# Edit Distance Costs
		if costs != None:
			self.insert_costs = costs['ins_costs']
			self.delete_costs = costs['del_costs']
			self.substitute_costs = costs['sub_costs']
			self.transpose_costs  = costs['trans_costs']
		else:
			self.insert_costs = np.ones((128,))
			self.delete_costs = np.ones((128,))
			self.transpose_costs  = np.ones((128,128))
			self.substitute_costs = np.ones((128,128))

		# Build phonetic index - Double Metaphone
		self.dmeta = fuzzy.DMetaphone()
		self.phonetic_buckets = {}

		for word in self.dict_words:
			phonetic_idx = self.dmeta(word)

			if phonetic_idx[0] not in self.phonetic_buckets:
				self.phonetic_buckets[phonetic_idx[0]] = []
			self.phonetic_buckets[phonetic_idx[0]].append(word)

			if phonetic_idx[1] != None:
				if phonetic_idx[1] not in self.phonetic_buckets:
					self.phonetic_buckets[phonetic_idx[1]] = []
				self.phonetic_buckets[phonetic_idx[1]].append(word)

	def __edit_neighbors_1(self, word):
		word_len = len(word)
		deletions  		= [(word[:i]+word[i+1:]) for i in range(word_len)]
		insertions 		= [word[:i]+letter+word[i:] for i in range(word_len+1) for letter in self.alphabet]
		substitutions  	= [word[:i]+letter+word[i+1:] for i in range(word_len) for letter in self.alphabet]
		transpositions 	= [word[:i]+word[i+1]+word[i]+word[i+2:] for i in range(word_len-1)]
		return set(deletions + insertions + substitutions + transpositions)

	def __filter_unknown(self, words):
		return set([word for word in words if word in self.dict_words])

	def __get_words_from_partial_candidates(self, words):
		candidates = set()
		for word in words:
			for dict_word in self.dict_words:
				if word in dict_word:
					candidates.add(dict_word)
					break
		return candidates

	def __fastGenerateNeighbors(self, left, right, max_dist=2):
		# Boundary Conditions
		if max_dist == 0:
			if left+right in self.valid_prefixes:	return [left+right]
			else:									return []

		if len(right) == 0:
			results = []
			if left in self.valid_prefixes:
				results.append(left)
			for letter in self.alphabet:
				if left + letter in self.valid_prefixes:
					results.append(left + letter)
			return list(set(results))

		# Update bounds
		left = left + right[:1]
		right = right[1:]

		# Initialize neighbors
		neighbor_set = []

		# Deletions
		if left[:-1] in self.valid_prefixes:
			neighbor_set += self.__fastGenerateNeighbors(
				left[:-1], right, max_dist-1
			)

		# Insertions
		for letter in self.alphabet:
			if left[:-1]+letter+left[-1:]  in self.valid_prefixes:
				neighbor_set += self.__fastGenerateNeighbors(
					left[:-1]+letter+left[-1:],
					right,
					max_dist-1)

		# Substitutions
		for letter in self.alphabet:
			if left[:-1]+letter in self.valid_prefixes:
				neighbor_set += self.__fastGenerateNeighbors(
					left[:-1]+letter,
					right,
					max_dist - (1 if letter != left[-1] else 0))

		# Transpositions
		if len(right) >= 1:
			if left[:-1] + right[0] + left[-1] in self.valid_prefixes:
				neighbor_set += self.__fastGenerateNeighbors(
					left[:-1]+right[0]+left[-1],
					right[1:],
					max_dist-1)

		return list(set(neighbor_set))

	def generateCandidates(self, wrong_word):
		return self.__generateCandidates(wrong_word)

	def __generateCandidates(self, wrong_word):
		# Edit Distance based candidates
		candidates = self.__fastGenerateNeighbors('', wrong_word, 3)
		known_candidates = self.__filter_unknown(candidates)

		if len(known_candidates) == 0:
			partial_candidates = self.__generate_partial_candidates(candidates)
			print(partial_candidates)
			return self.__get_words_from_partial_candidates(partial_candidates)
		else:
			return known_candidates

		# TODO DMetaphone based candidates
		# metaphone_bkts = self.dmeta(wrong_word)
		# candidates_meta = \
		# 	self.phonetic_buckets.get(metaphone_bkts[0], []) + \
		# 	(self.phonetic_buckets.get(metaphone_bkts[1], [])
		# 	 if metaphone_bkts[1] != None else [])
		# candidates_meta = set(candidates_meta)

		# return (candidates_meta.union(known_candidates ))

	def __generate_partial_candidates(self, candidates):
		list = []
		candidates = sorted(candidates)
		for i, val in enumerate(candidates):
			if i < len(candidates) - 1:
				if not candidates[i] in candidates[i + 1]:
					list.append(candidates[i])
			else:
				list.append(candidates[i])
		return list

	def __score(self, wrong_word, candidate):
		dl_dist = dam_lev(wrong_word,
						  candidate,
						  insert_costs=self.insert_costs,
						  substitute_costs=self.substitute_costs,
						  delete_costs=self.delete_costs,
						  transpose_costs=self.transpose_costs) / \
				  max(len(wrong_word), len(candidate))
		log_prior = self.priors[candidate] if candidate in self.priors \
			else math.log(float(self.k) / self.N)
		return -dl_dist + self.lamda * log_prior

	def __rankCandidates(self, wrong_word, candidates):
		return [(candidate, self.__score(wrong_word, candidate))
				for candidate in candidates]

	def correct(self, wrong_word, top_k=3):
		candidates = self.__generateCandidates(wrong_word)
		scores	   = self.__rankCandidates(wrong_word, candidates)
		return sorted(scores, key= lambda x:-x[1])[:top_k]

def read_list_dict(dictfile):
	"""
	Read word list file.

	:param dictfile:
	:return:
	"""
	with open(dictfile) as fp:
		words = set([line.strip() for line in fp])
	return words

def read_unigram_probs(unigram_file):
	"""
	Reads unigrams and corresponding frequency counts.

	:param unigram_file:
	:return:
	"""
	with open(unigram_file) as fp:
		lines = [[tok.strip() if i==0 else int(tok.strip())
				  for i, tok in enumerate(line.split('\t'))] for line in fp]
	return lines


if __name__ == '__main__':

	FRESH = True
	DEBUG = False

	# First time execution
	if FRESH:

		word_set = read_list_dict('../Data/lista_parole.txt')
		unigrams = read_unigram_probs('../Data/conteggio_catalogo.txt')

		# Read edit costs
		costs = np.load("../Data/costs.npz")
		costs = {
			'ins_costs': costs['ins_costs'],
			'del_costs': costs['del_costs'],
			'sub_costs': costs['sub_costs'],
			'trans_costs': costs['trans_costs'],
		}

		# Build Checker model
		checker = SpellChecker(word_set, unigrams, 1, costs=None, lamda=0.05)
		if not os.path.isfile('../Data/Models/model.pkl'):
			os.mknod('../Data/Models/model.pkl')
		with open('../Data/Models/model.pkl', 'wb') as fp:
			cPickle.dump(checker, fp)

	# Load Checker model
	with open('../Data/Models/model.pkl', 'rb') as fp:
		checker = cPickle.load(fp)


	input_doc = "../in.txt"
	output_doc = "../out.txt"
	# Output results
	with open(input_doc) as fin, open(output_doc, 'w') as fout:
		for line in fin:
			word = line.strip()
			guesses = checker.correct(word, 4)
			if DEBUG == True:
				fout.write('\t'.join([word] + ['(%s,%.2f)'%(guess,score)
											   for guess,score in guesses]) + '\n')
			else:
				fout.write('\t'.join([word] + [guess for guess, score in guesses]) + '\n')