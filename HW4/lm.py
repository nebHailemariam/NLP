#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Ishita <igoyal@andrew.cmu.edu> and Suyash <schavan@andrew.cmu.edu> based on work by Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 4
N-gram Language Model Implementation

Complete the LanguageModel class and other TO-DO methods.
"""

#######################################
# Import Statements
#######################################
from utils import *
from collections import Counter
from itertools import product
import argparse
import random
import math

#######################################
# Helper Functions
#######################################
def flatten(lst):
    """
    Flattens a nested list into a 1D list.
    Args:
        lst: Nested list (2D)
    
    Returns:
        Flattened 1-D list
    """
    def helper():
        from collections.abc import Iterable

        for x in lst:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x
    
    return list(helper())

#######################################
# TO-DO: get_ngrams()
#######################################
def get_ngrams(list_of_words, n):
    """
    Returns a list of n-grams for a list of words.
    Args
    ----
    list_of_words: List[str]
        List of already preprocessed and flattened (1D) list of tokens e.g. ["<s>", "hello", "</s>", "<s>", "bye", "</s>"]
    n: int
        n-gram order e.g. 1, 2, 3
    
    Returns:
        n_grams: List[Tuple]
            Returns a list containing n-gram tuples
    """
    list_of_words = flatten(list_of_words)
    n_grams = []

    for i in range(len(list_of_words)):
        store = list_of_words[i:i+n]
        if len(store) == n:
            n_grams.append(tuple(store))
    

    return n_grams

#######################################
# TO-DO: LanguageModel()
#######################################
class LanguageModel():
    def __init__(self, n, train_data, alpha=1):
        """
        Language model class.
        
        Args
        ____
        n: int
            n-gram order
        train_data: List[List]
            already preprocessed unflattened list of sentences. e.g. [["<s>", "hello", "my", "</s>"], ["<s>", "hi", "there", "</s>"]]
        alpha: float
            Smoothing parameter
        
        Other attributes:
            self.tokens: list of individual tokens present in the training corpus
            self.vocab: vocabulary dict with counts
            self.model: n-gram language model, i.e., n-gram dict with probabilties
            self.n_grams_counts: dictionary for storing the frequency of ngrams in the training data, keys being the tuple of words(n-grams) and value being their frequency
            self.prefix_counts: dictionary for storing the frequency of the (n-1) grams in the data, similar to the self.n_grams_counts
            As an example:
            For a trigram model, the n-gram would be (w1,w2,w3), the corresponding [n-1] gram would be (w1,w2)
        """
        self.n = n
        self.train_data = train_data
        self.n_grams_counts = Counter([])
        self.prefix_counts = Counter([])
        self.alpha = alpha
        
        # Fill in the following two lines of code
        self.tokens = flatten(train_data)
        self.vocab  = Counter(self.tokens)
        self.sum_of_frequencies  = 0

        self.model = self.build()


    def build(self):
        """
        Returns a n-gram dict with their smoothed probabilities. Remember to consider the edge case of n=1 as well
        
        You are expected to update the self.n_grams_counts and self.prefix_counts, and use those calculate the probabilities. 
        """
        # Extract n-grams from the flattened training data [update n_grams_counts]
        
        # Calculate the prefix (n-1 grams) count using the extracted n-grams [update prefix_counts]
        
        # Calculate probabilities using the get_smooth_probabilities function, you need to define the function

        # Return the probabilities
        n_gram_probabilities = Counter()


        n_grams = get_ngrams(self.tokens, n=self.n)
        
        for n_gram in n_grams:
            if n_gram not in self.n_grams_counts: 
                self.n_grams_counts[n_gram] = 1
            else:
                self.n_grams_counts[n_gram] += 1
            if self.n != 1:
                self.prefix_counts[n_gram[0:-1]] += n_gram[0:-1].count(n_gram[0:-1][0])

        self.sum_of_frequencies = sum(self.n_grams_counts.values())
        for n_gram in n_grams:
            if n_gram not in n_gram_probabilities:
                n_gram_probabilities[n_gram] = self.get_smooth_probabilites(n_gram)

        return n_gram_probabilities
    
    def get_smooth_probabilites(self,n_gram):
        """
        Returns the smoothed probability of the n-gram, using Laplace Smoothing. 
        Remember to consider the edge case of  n = 1
        HINT: Use self.n_gram_counts, self.tokens and self.prefix_counts 
        """
        if self.n != 1:
            return (self.n_grams_counts[n_gram] + self.alpha) / (self.prefix_counts[n_gram[0:-1]] + self.alpha * len(self.vocab))
        else:
            return (self.n_grams_counts[n_gram] + self.alpha) / (self.sum_of_frequencies + self.alpha * len(self.vocab))

#######################################
# TO-DO: perplexity()
#######################################
def perplexity(lm, test_data):
    """
    Returns perplexity calculated on the test data.
    Args
    ----------
    test_data: List[List] 
        Already preprocessed nested list of sentences
        
    lm: LanguageModel class object
        To be used for retrieving lm.model, lm.n and lm.vocab

    Returns
    -------
    float
        Calculated perplexity value
    """
    flattened_data = flatten(test_data)
    n_grams = get_ngrams(flattened_data, lm.n)
    perplexity_result = 0
    for n_gram in n_grams:
        perplexity_result -= math.log(lm.get_smooth_probabilites(n_gram))
    
    perplexity_result = (1/len(flattened_data)) * perplexity_result

    return math.exp(perplexity_result)

###############################################
# Method: Most Probable Candidates [Don't Edit]
###############################################
def best_candidate(lm, prev, i, without=[], mode="random"):
    """
    Returns the most probable word candidate after a given sentence.
    """
    blacklist  = ["<UNK>"] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        if(mode=="random"):
            return candidates[random.randrange(len(candidates))]
        else:
            return candidates[0]

def top_k_best_candidates(lm, prev, k, without=[]):
    """
    Returns the K most-probable word candidate after a given n-1 gram.
    Args
    ----
    lm: LanguageModel class object
    
    prev: n-1 gram
        List of tokens n
    """
    blacklist  = ["<UNK>"] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        return candidates[:k]
        
###########################################
# Method: Generate Sentences [Don't Edit]
###########################################
def generate_sentences_from_phrase(lm, num, sent, prob, mode):
    """
    Generate sentences using the trained language model after a
    provided phrase.
    """
    min_len=12
    max_len=24

    sentences = []
    for i in range(num):
        while sent[-1] != "</s>":
            prev = () if lm.n == 1 else tuple(sent[-(lm.n-1):])
            blacklist = sent + (["</s>"] if len(sent) < min_len else [])

            next_token, next_prob = best_candidate(lm, prev, i, without=blacklist, mode=mode)
            sent.append(next_token)
            prob *= next_prob
            
            if len(sent) >= max_len:
                sent.append("</s>")

        sentences.append((' '.join(sent), -1/math.log(prob)))

    return sentences

def generate_sentences(lm, num, mode="random"):
    """
    Generate sentences using the trained language model without any 
    provided phrase to begin with.
    """
    min_len=12
    max_len=24

    sentences = []
    for i in range(num):
        sent, prob = ["<s>"] * max(1, lm.n-1), 1
        while sent[-1] != "</s>":
            prev = () if lm.n == 1 else tuple(sent[-(lm.n-1):])
            blacklist = sent + (["</s>"] if len(sent) < min_len else [])

            next_token, next_prob = best_candidate(lm, prev, i, without=blacklist, mode=mode)
            sent.append(next_token)
            prob *= next_prob
            
            if len(sent) >= max_len:
                sent.append("</s>")

        sentences.append((' '.join(sent), -1/math.log(prob)))

    return sentences

# Copy of main executable script provided locally for your convenience
# This is not executed on autograder, so do what you want with it
if __name__ == '__main__':
    train = "data/sample.txt"
    test = "data/sample.txt"
    n = 2
    alpha = 0

    print("No of sentences in train file: {}".format(len(train)))
    print("No of sentences in test file: {}".format(len(test)))

    print("Raw train example: {}".format(train[2]))
    print("Raw test example: {}".format(test[2]))

    train = preprocess(train, n)
    test = preprocess(test, n)

    print("Preprocessed train example: \n{}\n".format(train[2]))
    print("Preprocessed test example: \n{}".format(test[2]))

    # Language Model
    print("Loading {}-gram model.".format(n))
    lm = LanguageModel(n, train, alpha)

    print("Vocabulary size (unique unigrams): {}".format(len(lm.vocab)))
    print("Total number of unique n-grams: {}".format(len(lm.model)))
    
    # Perplexity
    ppl = perplexity(lm=lm, test_data=test)
    print("Model perplexity: {:.3f}".format(ppl))
    
    # Generating sentences using your model
    print("Generating random sentences.")
    num_to_generate = 5
    for sentence, prob in generate_sentences(lm, num_to_generate):
        print("{} ({:.5f})".format(sentence, prob))