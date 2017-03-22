from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from six.moves import xrange

import numpy as np
import torch.nn as nn


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, generations, corrections):
        for i in xrange(generations.shape[0]):
            self._push(generations[i], corrections[i])

    def _push(self, generation, correction):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (generation, correction)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        generations, corrections = zip(*random.sample(self.memory, batch_size))
        return np.array(generations), np.array(corrections)

    def __len__(self):
        return len(self.memory)


class ExponentialReplayMemory(object):
    def __init__(self, capacity, half):
        self.capacity = capacity
        self.memory = []
        exp_lambda = np.log(2) / half
        self.probs = np.array([exp_lambda * np.exp(-exp_lambda * i) for i in xrange(capacity)])

    def push(self, generations, corrections):
        for i in xrange(generations.shape[0]):
            self._push(generations[i], corrections[i])

    def _push(self, generation, correction):
        self.memory.insert(0, (generation, correction))
        if len(self.memory) > self.capacity:
            self.memory.pop()

    def sample(self, batch_size):
        probs = self.probs[:len(self.memory)]
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), size=batch_size, replace=False, p=probs)
        generations, corrections = zip(*[self.memory[i] for i in indices])
        return np.array(generations), np.array(corrections)

    def __len__(self):
        return len(self.memory)


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


class Task(object):
    def __init__(self, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def get_data(self):
        '''Get a batch of data'''
        raise NotImplementedError

    def solved(self, data):
        '''Return true if the task has been solved, according to data'''
        return False


class WordsTask(Task):
    def __init__(self, seq_len, vocab_size):
        super(WordsTask, self).__init__(seq_len, vocab_size)

    def get_data(self, batch_size):
        '''Generate very simple toy training data. Generates sequences of integers where a 'word' is
           consecutive increasing integers and 0 separates words.'''
        batch = np.zeros([batch_size, self.seq_len], dtype=np.int)
        cur_word = np.random.randint(1, self.vocab_size, size=batch_size, dtype=np.int)
        batch[:, 0] = cur_word
        for i in xrange(1, self.seq_len):
            zero_mask = cur_word == 0
            cur_word += 1
            cur_word[cur_word > self.vocab_size-1] = 0
            cur_word *= np.random.binomial(np.ones(batch_size, dtype=np.int), 0.75)
            cur_word[zero_mask] = np.random.randint(1, self.vocab_size, size=np.sum(zero_mask),
                                                    dtype=np.int)
            batch[:, i] = cur_word
        return batch

    def solved(self, generated):
        return False  # TODO


class LongtermTask(Task):
    def __init__(self, seq_len, vocab_size):
        super(LongtermTask, self).__init__(seq_len, vocab_size)

    def get_data(self, batch_size):
        '''Generate simple toy training data where two tokens appear separated by large number
           of 0's.'''
        batch = np.zeros([batch_size, self.seq_len], dtype=np.int)
        batch[:, int(0.33 * self.seq_len)] = np.random.randint(1, self.vocab_size,
                                                               size=batch_size, dtype=np.int)
        batch[:, int(0.5 * self.seq_len)] = np.random.randint(1, self.vocab_size // 2,
                                                              size=batch_size, dtype=np.int)
        batch[:, int(0.8 * self.seq_len)] = np.random.randint(1, self.vocab_size,
                                                              size=batch_size, dtype=np.int)
        return batch

    def solved(self, avgprobs):
        # avgprobs size: (seq_len, vocab_size)
        assert avgprobs.shape[0] == self.seq_len
        indices = set([int(0.33 * self.seq_len), int(0.8 * self.seq_len)])
        half_indices = set([int(0.5 * self.seq_len)])
        for i in xrange(self.seq_len):
            probs = avgprobs[i]
            if i in indices:
                if probs[0] > 0.05:
                    return False
                meanprob = 1 / (self.vocab_size - 1)
                for j in xrange(1, self.vocab_size):
                    if np.abs(probs[j] - meanprob) > 0.01 * (self.vocab_size - 1):
                        return False
            elif i in half_indices:
                if probs[0] > 0.05:
                    return False
                meanprob = 1 / ((self.vocab_size // 2) - 1)
                for j in xrange(1, self.vocab_size // 2):
                    if np.abs(probs[j] - meanprob) > 0.02 * (self.vocab_size // 2 - 1):
                        return False
                for j in xrange(self.vocab_size // 2, self.vocab_size):
                    if probs[j] > 0.025:
                        return False
            else:
                if probs[0] < 0.95:
                    return False
        return True
