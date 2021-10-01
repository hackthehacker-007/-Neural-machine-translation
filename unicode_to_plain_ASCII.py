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
