from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from six.moves import xrange
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from main import Critic
import util


def get_fake_toy_data_words(batch_size, seq_len, vocab_size, strategy='real'):
    batch = np.zeros([batch_size, seq_len], dtype=np.int)
    if strategy == 'real':  # TODO more strategies
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


def get_fake_toy_data_longterm(batch_size, seq_len, vocab_size, strategy='real'):
    batch = np.zeros([batch_size, seq_len], dtype=np.int)
    if strategy == 'zeros':
        return batch
    elif strategy == 'real':
        batch[:, int(0.33 * seq_len)] = np.random.randint(1, vocab_size, size=batch_size,
                                                          dtype=np.int)
        batch[:, int(0.8 * seq_len)] = np.random.randint(1, vocab_size, size=batch_size,
                                                         dtype=np.int)
        return batch
    elif strategy == 'close':
        r1 = max(min(np.random.normal(loc=0.33, scale=0.01), 0.999), 0.0)
        batch[:, int(r1 * seq_len)] = np.random.randint(1, vocab_size, size=batch_size,
                                                          dtype=np.int)
        r2 = max(min(np.random.normal(loc=0.8, scale=1e-3), 0.999), 0.0)
        batch[:, int(r2 * seq_len)] = np.random.randint(1, vocab_size, size=batch_size,
                                                         dtype=np.int)
        return batch
    elif strategy == 'random':
        batch = np.random.randint(1, vocab_size, size=batch.shape, dtype=np.int)
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seq_len', type=int, default=15, help='toy sequence length')
    parser.add_argument('--vocab_size', type=int, default=6,
                        help='character vocab size for toy data')
    parser.add_argument('--emb_size', type=int, default=32, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=128, help='RNN hidden size')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--clamp_limit', type=float, default=1.0)
    parser.add_argument('--critic_iters', type=int, default=5,
                        help='number of critic iters per turn')
    parser.add_argument('--task', type=str, default='longterm', help='longterm or words')
    parser.add_argument('--strategy', type=str, default='real', help='fake data strategy')
    parser.add_argument('--print_every', type=int, default=500,
                        help='print losses every these many steps')
    opt = parser.parse_args()
    opt.gamma = 1.0
    print(opt)

    cudnn.benchmark = True
    np.set_printoptions(precision=4, threshold=10000, linewidth=200, suppress=True)

    if opt.task == 'words':
        get_data = util.get_toy_data_words
        get_fake_data = get_fake_toy_data_words
    elif opt.task == 'longterm':
        get_data = util.get_toy_data_longterm
        get_fake_data = get_fake_toy_data_longterm
    else:
        print('error: invalid task name:', opt.task)
        sys.exit(1)

    critic = Critic(opt)
    critic.cuda()

    one = torch.cuda.FloatTensor([1])
    mone = one * -1

    optimizer = optim.RMSprop(critic.parameters(), lr=opt.learning_rate)

    Wdists = []
    err_r = []
    err_f = []
    for epoch in xrange(opt.niter):
        for param in critic.parameters():
            param.data.clamp_(-opt.clamp_limit, opt.clamp_limit)
        critic.zero_grad()

        fake = torch.from_numpy(get_fake_data(opt.batch_size, opt.seq_len,
                                              opt.vocab_size, strategy=opt.strategy)).cuda()
        E_generated = critic(fake).sum() / opt.batch_size
        E_generated.backward(mone)

        real = torch.from_numpy(get_data(opt.batch_size, opt.seq_len,
                                         opt.vocab_size)).cuda()
        E_real = critic(real).sum() / opt.batch_size
        E_real.backward(one)

        optimizer.step()
        Wdist = (E_generated - E_real).data[0]
        Wdists.append(Wdist)
        err_r.append(E_real.data[0])
        err_f.append(E_generated.data[0])

        if epoch % opt.print_every == 0:
            print(epoch, ':\tWdist:', np.array(Wdists).mean(), '\terr R: ', np.array(err_r).mean(),
                  '\terr F: ', np.array(err_f).mean())
            Wdists = []
            err_r = []
            err_f = []
