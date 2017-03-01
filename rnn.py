from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util


class RNN(nn.Module):
    '''The RNN model.'''

    def __init__(self, opt):
        super(RNN, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.cell = nn.GRUCell(opt.emb_size, opt.hidden_size)
        self.dist = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.zero_input = torch.LongTensor(opt.batch_size, 1).zero_().cuda()
        self.zero_state = torch.zeros([opt.batch_size, opt.hidden_size]).cuda()

    def forward(self, inputs):
        logprobs = []
        hidden = Variable(self.zero_state)
        inputs = torch.cat([Variable(self.zero_input), inputs], 1)
        for i in xrange(inputs.size(1) - 1):
            emb = self.embedding(inputs[:, i])
            hidden = self.cell(emb, hidden)
            dist = F.log_softmax(self.dist(hidden)).unsqueeze(1)
            logprobs.append(dist)
        return torch.cat(logprobs, 1)

    def sample(self):
        outputs = []
        hidden = Variable(self.zero_state)
        inputs = self.embedding(Variable(self.zero_input.squeeze(1), volatile=True))
        for out_i in xrange(self.opt.seq_len):
            hidden = self.cell(inputs, hidden)
            dist = F.log_softmax(self.dist(hidden))
            _, sampled = torch.max(dist.data -
                                   torch.log(-torch.log(torch.rand(*dist.size()).cuda())), 1)
            sampled = Variable(sampled, requires_grad=False)
            outputs.append(sampled)
            if out_i < self.opt.seq_len - 1:
                inputs = self.embedding(sampled.squeeze(1))
        return torch.cat(outputs, 1)


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
    parser.add_argument('--clamp_limit', type=float, default=-1)
    parser.add_argument('--task', type=str, default='longterm', help='longterm or words')
    parser.add_argument('--print_every', type=int, default=50,
                        help='print losses every these many steps')
    parser.add_argument('--gen_every', type=int, default=50,
                        help='generate sample every these many steps')
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True
    np.set_printoptions(precision=4, threshold=10000, linewidth=200, suppress=True)

    if opt.task == 'words':
        get_data = util.get_toy_data_words
    elif opt.task == 'longterm':
        get_data = util.get_toy_data_longterm
    else:
        print('error: invalid task name:', opt.task)
        sys.exit(1)

    model = RNN(opt).apply(util.weights_init)
    model.cuda()
    criterion = torch.nn.NLLLoss()

    optimizer = optim.RMSprop(model.parameters(), lr=opt.learning_rate)

    print('\nReal examples:')
    print(get_data(opt.batch_size, opt.seq_len, opt.vocab_size), '\n')
    for epoch in xrange(opt.niter):
        if opt.clamp_limit > 0.0:
            for param in model.parameters():
                param.data.clamp_(-opt.clamp_limit, opt.clamp_limit)
        model.zero_grad()
        real = Variable(torch.from_numpy(get_data(opt.batch_size, opt.seq_len,
                                                  opt.vocab_size)).cuda())
        logprobs = model(real)
        loss = criterion(logprobs.view(-1, opt.vocab_size), real.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % opt.print_every == 0:
            print(epoch, ': Loss:', loss.data.cpu().numpy()[0])
        if epoch % opt.gen_every == 0:
            sampled = model.sample()
            print('Generated:')
            print(sampled.data.cpu().numpy(), '\n')
            if opt.task == 'longterm':
                probs = torch.exp(logprobs).mean(0).squeeze(0).data.cpu().numpy()
                print('Batch-averaged step-wise probs:')
                print(probs, '\n')
