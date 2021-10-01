import unicodedata
import re
import math
import psutil
import time
import datetime
from io import open
import random
from random import shuffle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.cuda

"""this line clears sys to allow for argparse to work as gradient clipper"""
import sys; sys.argv=['']; del sys

"""This function converts a Unicode string to plain ASCII 
from https://stackoverflow.com/a/518232/2809427"""
def uniToAscii(sentence):
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
    )


"""Lowercase, trim, and remove non-letter characters (from pytorch)"""
def normalizeString(s):
    s = re.sub(r" ##AT##-##AT## ", r" ", s)
    s = uniToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s   

  
"""Denote patterns that sentences must start with to be kept in dataset. 
Can be changed if desired (from pytorch)"""
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)



"""Filters each input-output pair, keeping sentences that are less than max_length 
if start_filter is true, also filters out sentences that don't start with eng_prefixes"""
def filterPair(p, max_length, start_filter):
    filtered = len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length 
    if start_filter:
        return filtered and p[1].startswith(eng_prefixes)
    else:
        return filtered

"""Filters all of the input-output language pairs in the dataset using filterPair 
for each pair (from pytorch)"""
def filterPairs(pairs, max_length, start_filter):
    return [pair for pair in pairs if filterPair(pair, max_length, start_filter)]


"""start of sentence tag"""
SOS_token = 0

"""end of sentence tag"""
EOS_token = 1

"""unknown word tag (this is used to handle words that are not in our Vocabulary)"""
UNK_token = 2


"""Lang class, used to store the vocabulary of each language"""
class Lang:
    def _init_(self, language):
        self.language_name = language
        self.word_to_index = {"SOS":SOS_token, "EOS":EOS_token, "<UNK>":UNK_token}
        self.word_to_count = {}
        self.index_to_word = {SOS_token: "SOS", EOS_token: "EOS", UNK_token: "<UNK>"}
        self.vocab_size = 3
        self.cutoff_point = -1


    def countSentence(self, sentence):
        for word in sentence.split(' '):
            self.countWords(word)

    """counts the number of times each word appears in the dataset"""
    def countWords(self, word):
        if word not in self.word_to_count:
            self.word_to_count[word] = 1
        else:
            self.word_to_count[word] += 1

    """if the number of unique words in the dataset is larger than the
    specified max_vocab_size, creates a cutoff point that is used to
    leave infrequent words out of the vocabulary"""
    def createCutoff(self, max_vocab_size):
        word_freqs = list(self.word_to_count.values())
        word_freqs.sort(reverse=True)
        if len(word_freqs) > max_vocab_size:
            self.cutoff_point = word_freqs[max_vocab_size]

    """assigns each unique word in a sentence a unique index"""
    def addSentence(self, sentence):
        new_sentence = ''
        for word in sentence.split(' '):
            unk_word = self.addWord(word)
            if not new_sentence:
                new_sentence =unk_word
            else:
                new_sentence = new_sentence + ' ' + unk_word
        return new_sentence

    """assigns a word a unique index if not already in vocabulary
    and it appeaars often enough in the dataset
    (self.word_to_count is larger than self.cutoff_point)"""
    def addWord(self, word):
        if self.word_to_count[word] > self.cutoff_point:
            if word not in self.word_to_index:
                self.word_to_index[word] = self.vocab_size
                self.index_to_word[self.vocab_size] = word
                self.vocab_size += 1
            return word
        else:
            return self.index_to_word[2]

'''prepares both the input and output Lang classes from the passed dataset'''

def prepareLangs(lang1, lang2, file_path, reverse=False):
    print("Reading lines...")

    if len(file_path) == 2:
        lang1_lines = open(file_path[0], encoding='utf-8').\
            read().strip().split('\n')

        lang2_lines = open(file_path[1], encoding='utf-8').\
            read().strip().split('\n')

        if len(lang1_lines) != len(lang2_lines):
            print("Input and output text sizes do not align")
            print("Number of lang1 lines: %s " %len(lang1_lines))
            print("Number of lang2 lines: %s " %len(lang2_lines))
            quit()

        pairs = []

        for line in range(len(lang1_lines)):
            pairs.append([normalizeString(lang1_lines[line]),
                          normalizeString(lang2_lines[line])])            


    elif len(file_path) == 1:
        lines = open(file_path[0], encoding='utf-8').\
    	read().strip().split('\n')
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs'''prepares both the input and output Lang classes from the passed dataset'''

def prepareLangs(lang1, lang2, file_path, reverse=False):
    print("Reading lines...")

    if len(file_path) == 2:
        lang1_lines = open(file_path[0], encoding='utf-8').\
            read().strip().split('\n')

        lang2_lines = open(file_path[1], encoding='utf-8').\
            read().strip().split('\n')

        if len(lang1_lines) != len(lang2_lines):
            print("Input and output text sizes do not align")
            print("Number of lang1 lines: %s " %len(lang1_lines))
            print("Number of lang2 lines: %s " %len(lang2_lines))
            quit()

        pairs = []

        for line in range(len(lang1_lines)):
            pairs.append([normalizeString(lang1_lines[line]),
                          normalizeString(lang2_lines[line])])            


    elif len(file_path) == 1:
        lines = open(file_path[0], encoding='utf-8').\
    	read().strip().split('\n')
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

"""completely prepares both input and output languages 
and returns cleaned and trimmed train and test pairs"""

def prepareData(lang1, lang2, file_path, max_vocab_size=50000, 
                reverse=False, trim=0, start_filter=False, perc_train_set=0.9, 
                print_to=None):
    
    input_lang, output_lang, pairs = prepareLangs(lang1, lang2, 
                                                  file_path, reverse)
    
    print("Read %s sentence pairs" % len(pairs))
    
    if print_to:
        with open(print_to,'a') as f:
            f.write("Read %s sentence pairs \n" % len(pairs))
    
    if trim != 0:
        pairs = filterPairs(pairs, trim, start_filter)
        print("Trimmed to %s sentence pairs" % len(pairs))
        if print_to:
            with open(print_to,'a') as f:
                f.write("Read %s sentence pairs \n" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.countSentence(pair[0])
        output_lang.countSentence(pair[1])


    input_lang.createCutoff(max_vocab_size)
    output_lang.createCutoff(max_vocab_size)

    pairs = [(input_lang.addSentence(pair[0]),output_lang.addSentence(pair[1])) 
             for pair in pairs]

    shuffle(pairs)
    
    train_pairs = pairs[:math.ceil(perc_train_set*len(pairs))]
    test_pairs = pairs[math.ceil(perc_train_set*len(pairs)):]

    print("Train pairs: %s" % (len(train_pairs)))
    print("Test pairs: %s" % (len(test_pairs)))
    print("Counted Words -> Trimmed Vocabulary Sizes (w/ EOS and SOS tags):")
    print("%s, %s -> %s" % (input_lang.language_name, len(input_lang.word_to_count),
                            input_lang.vocab_size,))
    print("%s, %s -> %s" % (output_lang.language_name, len(output_lang.word_to_count), 
                            output_lang.vocab_size))
    print()

    if print_to:
        with open(print_to,'a') as f:
            f.write("Train pairs: %s" % (len(train_pairs)))
            f.write("Test pairs: %s" % (len(test_pairs)))
            f.write("Counted Words -> Trimmed Vocabulary Sizes (w/ EOS and SOS tags):")
            f.write("%s, %s -> %s" % (input_lang.language_name, 
                                      len(input_lang.word_to_count),
                                      input_lang.vocab_size,))
            f.write("%s, %s -> %s \n" % (output_lang.language_name, len(output_lang.word_to_count), 
                            output_lang.vocab_size))
        
    return input_lang, output_lang, train_pairs, test_pairs
