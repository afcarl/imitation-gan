from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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


def gradient_norm(parameters, norm_type=2):
    # remove this method once pytorch is updated, clip_grad_norn will return the original total norm
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


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

    def display(self, data):
        print(data)


class LMTask(Task):
    def __init__(self, data_dir, seq_len):
        super(LMTask, self).__init__(seq_len, 0)
        self.word2idx = {}
        self.idx2word = []
        self.add_word('<s>')  # zero_input is padded to the front in the model
        self.add_word('<p>')  # to pad after eos
        self.add_word('<e>')  # eos
        self.splits = {}
        for s in ['train', 'valid', 'test']:
            self.splits[s] = self.tokenize(os.path.join(data_dir, s + '.txt'), seq_len)
        random.shuffle(self.splits['train'])
        self.vocab_size = len(self.idx2word)
        self.current = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def tokenize(self, path, max_seq_len):
        """Tokenizes a text file and returns a list of sentences."""
        assert os.path.exists(path)
        ret = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<e>']
                if max_seq_len > 0:
                    words = words[:max_seq_len]
                ids = []
                for word in words:
                    ids.append(self.add_word(word))
                self.seq_len = max(self.seq_len, len(ids))
                ret.append(ids)
        return ret

    def get_data(self, batch_size):
        data = self.splits['train']
        assert len(data) >= batch_size
        if self.current + batch_size > len(data):
            self.current = 0
            random.shuffle(data)
        data = data[self.current:self.current+batch_size]
        self.current += batch_size
        batch = np.ones([batch_size, self.seq_len], dtype=np.int) * self.word2idx['<p>']
        for i, s in enumerate(data):
            batch[i, :len(s)] = s
        return batch

    def display(self, data):
        print(data)
        for s in data:
            print('=>', end=' ')
            for w in s:
                print(self.idx2word[w], end=' ')
            print()


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
                if probs[0] > min(0.05, 1 / (2 * self.vocab_size)):
                    return False
                meanprob = 1 / (self.vocab_size - 1)
                for j in xrange(1, self.vocab_size):
                    if probs[j] < 1 / (2 * self.vocab_size):
                        return False
                    if np.abs(probs[j] - meanprob) > 0.01 * (self.vocab_size - 1):
                        return False
            elif i in half_indices:
                if probs[0] > min(0.05, 1 / (2 * self.vocab_size)):
                    return False
                meanprob = 1 / ((self.vocab_size // 2) - 1)
                for j in xrange(1, self.vocab_size // 2):
                    if probs[j] < 1 / self.vocab_size:
                        return False
                    if np.abs(probs[j] - meanprob) > 0.02 * (self.vocab_size // 2 - 1):
                        return False
                for j in xrange(self.vocab_size // 2, self.vocab_size):
                    if probs[j] > min(0.025, 1 / (2 * self.vocab_size)):
                        return False
            else:
                if probs[0] < 0.95:
                    return False
        return True
