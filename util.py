from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import torch.nn as nn


def weights_init(m):
    def linear_init(weight):
        fan_out, fan_in = weight.size()
        weight.data.normal_(0.0, np.sqrt(2.0 / (fan_in + fan_out)))
    if isinstance(m, nn.Linear):
        linear_init(m.weight)
    elif isinstance(m, nn.GRUCell):
        linear_init(m.weight_ih)
        linear_init(m.weight_hh)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.size()) == 2:
                linear_init(param)
                linear_init(param)
    elif isinstance(m, nn.Embedding):
        m.weight.data.uniform_()


def get_toy_data_words(batch_size, seq_len, vocab_size):
    '''Generate very simple toy training data. Generates sequences of integers where a 'word' is
       consecutive increasing integers and 0 separates words.'''
    batch = np.zeros([batch_size, seq_len], dtype=np.int)
    cur_word = np.random.randint(1, vocab_size, size=batch_size, dtype=np.int)
    batch[:, 0] = cur_word
    for i in xrange(1, seq_len):
        zero_mask = cur_word == 0
        cur_word += 1
        cur_word[cur_word > vocab_size-1] = 0
        cur_word *= np.random.binomial(np.ones(batch_size, dtype=np.int), 0.75)
        cur_word[zero_mask] = np.random.randint(1, vocab_size, size=np.sum(zero_mask),
                                                dtype=np.int)
        batch[:, i] = cur_word
    return batch


def get_toy_data_longterm(batch_size, seq_len, vocab_size):
    '''Generate simple toy training data where two tokens appear separated by large number
       of 0's.'''
    batch = np.zeros([batch_size, seq_len], dtype=np.int)
    batch[:, int(0.33 * seq_len)] = np.random.randint(1, vocab_size, size=batch_size, dtype=np.int)
    batch[:, int(0.8 * seq_len)] = np.random.randint(1, vocab_size, size=batch_size, dtype=np.int)
    return batch
