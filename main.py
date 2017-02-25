from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from six.moves import xrange

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    '''The imitation GAN policy network.'''

    def __init__(self, opt):
        super(Actor, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.cell = nn.GRUCell(opt.emb_size, opt.hidden_size)
        self.dist = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.step = 0  # for eps decay

    def forward(self):
        outputs = []
        corrections = []
        logprobs = []
        hidden = Variable(torch.zeros([self.opt.batch_size, self.opt.hidden_size]).cuda())
        inputs = self.embedding(Variable(torch.LongTensor(self.opt.batch_size).zero_().cuda()))
        for out_i in xrange(self.opt.seq_len):
            hidden = self.cell(inputs, hidden)
            dist = F.log_softmax(self.dist(hidden))
            # this has to be a clone of dist, since we modify this later but return original dist:
            dist_numpy = dist.data.cpu().numpy()
            # decide the current eps threshold based on the number of steps so far
            eps_threshold = self.opt.eps_end + (self.opt.eps_start - self.opt.eps_end) * \
                                                  np.exp(-4. * self.step / self.opt.eps_decay_steps)
            draw_randomly = eps_threshold >= np.random.random_sample([self.opt.batch_size])
            # set uniform (log) probability with eps_threshold probability
            dist_numpy[draw_randomly, :] = -np.log(self.opt.vocab_size)

            # for explanation of below, see https://github.com/tensorflow/tensorflow/issues/456
            sampled = np.argmax(dist_numpy -
                                np.log(-np.log(np.random.random_sample(dist_numpy.shape))), axis=1)
            sampled = torch.from_numpy(sampled).cuda()
            sampled_unsq = sampled.unsqueeze(1)
            logprob = dist.gather(1, Variable(sampled_unsq))
            onpolicy_prob = torch.exp(logprob.detach())
            offpolicy_prob = torch.exp(torch.from_numpy(dist_numpy).cuda().gather(1, sampled_unsq))
            offpolicy_prob.clamp_(1e-3, 1.0)
            outputs.append(sampled_unsq)
            # use importance sampling to correct for eps sampling
            corrections.append(onpolicy_prob / Variable(offpolicy_prob))
            logprobs.append(logprob)
            if out_i < self.opt.seq_len - 1:
                inputs = self.embedding(Variable(sampled))
        return torch.cat(outputs, 1), torch.cat(corrections, 1), torch.cat(logprobs, 1)


class Critic(nn.Module):
    '''The imitation GAN discriminator/critic.'''

    def __init__(self, opt):
        super(Critic, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.rnn = nn.GRU(input_size=opt.emb_size, hidden_size=opt.hidden_size, num_layers=1,
                          batch_first=True)
        self.cost = nn.Linear(opt.hidden_size, 1)

    def forward(self, actions):
        actions = torch.cat([torch.LongTensor(self.opt.batch_size, 1).zero_().cuda(), actions],
                            1)
        actions = Variable(actions)
        inputs = self.embedding(actions)
        outputs, _ = self.rnn(inputs,
                              Variable(torch.zeros([1, self.opt.batch_size,
                                                    self.opt.hidden_size]).cuda()))
        outputs = outputs.contiguous()
        flattened = outputs.view(-1, self.opt.hidden_size)
        flat_costs = self.cost(flattened)
        costs = flat_costs.view(self.opt.batch_size, -1)
        costs = costs[:, 1:]  # ignore costs of the padded input token
        return costs


def weights_init(m):
    def linear_init(weight):
        fan_out, fan_in = weight.size()
        weight.data.normal_(0.0, np.sqrt(2.0 / (fan_in + fan_out)))
    if isinstance(m, nn.Linear):
        linear_init(m.weight)
        print('Initialized nn.Linear')
    elif isinstance(m, nn.GRUCell):
        linear_init(m.weight_ih)
        linear_init(m.weight_hh)
        print('Initialized nn.GRUCell')
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.size()) == 2:
                linear_init(param)
                linear_init(param)
        print('Initialized nn.GRU')
    elif isinstance(m, nn.Embedding):
        m.weight.data.uniform_()
        print('Initialized nn.Embedding')


def get_toy_data(batch_size, seq_len, vocab_size):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seq_len', type=int, default=3, help='toy sequence length')
    parser.add_argument('--vocab_size', type=int, default=2,
                        help='character vocab size for toy data')
    parser.add_argument('--emb_size', type=int, default=4, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=16, help='RNN hidden size')
    parser.add_argument('--eps_start', type=float, default=0.9, help='initial eps for eps-greedy')
    parser.add_argument('--eps_end', type=float, default=0.05, help='final eps for eps-greedy')
    parser.add_argument('--eps_decay_steps', type=int, default=800,
                        help='number of steps to exp decay over (4 for e^(-x))')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--critic_iters', type=int, default=5,
                        help='number of critic iters per each actor iter')
    opt = parser.parse_args()
    print(opt)
    cudnn.benchmark = True

    actor = Actor(opt).apply(weights_init)
    critic = Critic(opt).apply(weights_init)
    actor.cuda()
    critic.cuda()

    one = torch.cuda.FloatTensor([1])
    mone = one * -1

    actor_optimizer = optim.RMSprop(actor.parameters(), lr=opt.learning_rate)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=opt.learning_rate)

    print('\nReal examples:')
    print(get_toy_data(opt.batch_size, opt.seq_len, opt.vocab_size), '\n')
    for epoch in xrange(opt.niter):
        # train critic
        for param in critic.parameters():  # reset requires_grad
            param.requires_grad = True  # they are set to False below in actor update
        if epoch < 25 or epoch % 500 == 0:
            critic_iters = 100
        else:
            critic_iters = opt.critic_iters
        Wdists = []
        for critic_i in xrange(critic_iters):
            for param in critic.parameters():
                param.data.clamp_(-1, 1)
            critic.zero_grad()

            generated, corrections, _ = actor.forward()
            E_generated = (critic(generated) * corrections).sum() / opt.batch_size
            E_generated.backward(mone)

            real = torch.from_numpy(get_toy_data(opt.batch_size, opt.seq_len,
                                                 opt.vocab_size)).cuda()
            E_real = critic(real).sum() / opt.batch_size
            E_real.backward(one)

            critic_optimizer.step()
            Wdist = (E_generated - E_real).data[0]
            Wdists.append(Wdist)

        # train actor
        for param in critic.parameters():
            param.requires_grad = False  # to avoid computation
        actor.zero_grad()
        generated, corrections, logprobs = actor.forward()
        actor.step += 1  # do eps decay
        loss = (critic(generated) * corrections * logprobs).sum() / opt.batch_size
        loss.backward(one)
        actor_optimizer.step()

        if epoch % 10 == 0:
            print(epoch, ': Wdist:', np.array(Wdists).mean())
        if epoch % 50 == 0:
            print('Generated:')
            print(generated.cpu().numpy(), '\n')
