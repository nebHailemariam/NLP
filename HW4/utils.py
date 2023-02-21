#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Ishita <igoyal@andrew.cmu.edu> and Suyash <schavan@andrew.cmu.edu> based on work by Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 4
This file contains the preprocessing and read() functions. Don't edit this file.
"""

import os
import pdb

START = "<s>"
EOS = "</s>"

def read_file(file_path):
    """
    Read a single text file.
    """
    with open(file_path, 'r') as f:
        text = f.readlines()
    return text

def preprocess(sentences, n):
    """
    Args:
        sentences: List of sentences
        n: n-gram value

    Returns:
        preprocessed_sentences: List of preprocessed sentences
    """
    sentences = add_special_tokens(sentences, n)

    preprocessed_sentences = []
    for line in sentences:
        preprocessed_sentences.append([tok.lower() for tok in line.split()])
    
    return preprocessed_sentences

def add_special_tokens(sentences, ngram):
    num_of_start_tokens = ngram - 1 if ngram > 1 else 1
    start_tokens = " ".join([START] * num_of_start_tokens)
    sentences = ['{} {} {}'.format(start_tokens, sent, EOS) for sent in sentences]
    return sentences