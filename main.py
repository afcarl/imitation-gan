from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from six.moves import xrange

import matplotlib
matplotlib.use('Agg') # allows for saving images without display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Actor(nn.Module):
    '''The imitation GAN policy network.'''

    def __init__(self, opt):
        super(Actor, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.cell = nn.GRUCell(opt.emb_size, opt.hidden_size)
        self.dist = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.zero_input = torch.LongTensor(opt.batch_size).zero_().cuda()
        self.zero_state = torch.zeros([opt.batch_size, opt.hidden_size]).cuda()
        self.eps_sample = True  # Do eps sampling

    def forward(self):
        outputs = []
        corrections = []
        logprobs = []
        probs = []  # for debugging
        hidden = Variable(self.zero_state)
        inputs = self.embedding(Variable(self.zero_input))
        for out_i in xrange(self.opt.seq_len):
            hidden = self.cell(inputs, hidden)
            dist = F.log_softmax(self.dist(hidden))
            prob = torch.exp(dist).detach()
            probs.append(prob.mean(0).cpu().squeeze(0).data.numpy())  # for debugging
            if self.eps_sample:
                # this has to be a clone of prob, since we modify this but also use the original
                # prob
                prob_new = prob.data.cpu().numpy()
                draw_randomly = self.opt.eps >= np.random.random_sample([self.opt.batch_size])
                # set uniform distribution with opt.eps probability
                prob_new[draw_randomly, :] = 1. / self.opt.vocab_size
                prob_new = Variable(torch.from_numpy(prob_new).cuda(), requires_grad=False)
            else:
                prob_new = prob
            # eps sampling
            sampled = Variable(torch.multinomial(prob_new.data, 1), requires_grad=False)
            logprob = dist.gather(1, sampled)
            onpolicy_prob = prob.gather(1, sampled)
            offpolicy_prob = prob_new.gather(1, sampled)
            # avoid 0/0
            onpolicy_prob.data.clamp_(1e-8, 1.0)
            offpolicy_prob.data.clamp_(1e-8, 1.0)
            outputs.append(sampled)
            # use importance sampling to correct for eps sampling
            corrections.append(onpolicy_prob / offpolicy_prob)
            logprobs.append(logprob)
            if out_i < self.opt.seq_len - 1:
                inputs = self.embedding(sampled.squeeze(1))
        return (torch.cat(outputs, 1), torch.cat(corrections, 1), torch.cat(logprobs, 1),
                np.array(probs))


class Critic(nn.Module):
    '''The imitation GAN discriminator/critic.'''

    def __init__(self, opt):
        super(Critic, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.rnn = nn.GRU(input_size=opt.emb_size, hidden_size=opt.hidden_size, num_layers=1,
                          batch_first=True)
        self.cost = nn.Linear(opt.hidden_size, 1)
        self.zero_input = torch.LongTensor(opt.batch_size, 1).zero_().cuda()
        self.zero_state = torch.zeros([1, opt.batch_size, opt.hidden_size]).cuda()

    def forward(self, actions):
        actions = torch.cat([self.zero_input, actions], 1)
        actions = Variable(actions)
        inputs = self.embedding(actions)
        outputs, _ = self.rnn(inputs, Variable(self.zero_state))
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


def get_toy_data_grammar(batch_size, seq_len, vocab_size):
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


if __name__ == '__main__':
    # TODO introduce discounted costs. can help the model focus on getting the first action
    #      right before trying to deal with a long sequence of unrewarding actions based on an
    #      incorrect early action. gamma=1 should be fine for simple tasks.
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seq_len', type=int, default=15, help='toy sequence length')
    parser.add_argument('--vocab_size', type=int, default=6,
                        help='character vocab size for toy data')
    parser.add_argument('--emb_size', type=int, default=32, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=128, help='RNN hidden size')
    parser.add_argument('--eps', type=float, default=0.15, help='epsilon for eps sampling')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--clamp_limit', type=float, default=1.0)
    parser.add_argument('--critic_iters', type=int, default=5,
                        help='number of critic iters per each actor iter')
    parser.add_argument('--name', type=str, default='default')
    opt = parser.parse_args()
    print(opt)

    # some logging stuff
    opt.save = 'logs/' + opt.name
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    train_log = open(opt.save + '/train.log', 'w')
    colors = cm.rainbow(np.linspace(0, 1, 3))
    plot_r = []
    plot_f = []
    plot_w = []

    cudnn.benchmark = True
    np.set_printoptions(precision=5, threshold=10000, linewidth=160, suppress=True)

    get_data = get_toy_data_longterm  # TODO make configurable

    actor = Actor(opt).apply(weights_init)
    critic = Critic(opt).apply(weights_init)
    actor.cuda()
    critic.cuda()

    one = torch.cuda.FloatTensor([1])
    mone = one * -1

    actor_optimizer = optim.RMSprop(actor.parameters(), lr=opt.learning_rate)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=opt.learning_rate)
    #critic_optimizer = optim.SGD(critic.parameters(), lr=0.002) #opt.learning_rate)

    print('\nReal examples:')
    print(get_data(opt.batch_size, opt.seq_len, opt.vocab_size), '\n')
    for epoch in xrange(opt.niter):
        actor.eps_sample = True

        # train critic
        for param in critic.parameters():  # reset requires_grad
            param.requires_grad = True  # they are set to False below in actor update
        if epoch < 25 or epoch % 500 == 0:
            critic_iters = 100
        else:
            critic_iters = opt.critic_iters
        Wdists = []
        err_r = []
        err_f = []
        for critic_i in xrange(critic_iters):
            for param in critic.parameters():
                param.data.clamp_(-opt.clamp_limit, opt.clamp_limit)
            critic.zero_grad()

            # eps sampling here can help the critic get signal from less likely actions as well.
            # corrections will ensure that the critic doesn't have to worry about such actions
            # too much though.
            generated, corrections, _, _ = actor.forward()
            E_generated = (critic(generated.data) * corrections).sum() / opt.batch_size
            E_generated.backward(mone)

            real = torch.from_numpy(get_data(opt.batch_size, opt.seq_len,
                                             opt.vocab_size)).cuda()
            E_real = critic(real).sum() / opt.batch_size
            E_real.backward(one)

            critic_optimizer.step()
            Wdist = (E_generated - E_real).data[0]
            Wdists.append(Wdist)
            err_r.append(E_real.data[0])
            err_f.append(E_generated.data[0])

        # train actor
        for param in critic.parameters():
            param.requires_grad = False  # to avoid computation
        if epoch % 25 == 0:
            print_generated = True
            actor.eps_sample = False
        else:
            print_generated = False

        actor.zero_grad()
        generated, corrections, logprobs, probs = actor.forward()
        costs = critic(generated.data)
        loss = (costs * corrections * logprobs).sum() / opt.batch_size
        loss.backward(one)
        actor_optimizer.step()
        plot_r.append(-np.array(err_r).mean())
        plot_f.append(-np.array(err_f).mean())
        plot_w.append(np.array(Wdists).mean())
        if True or epoch % 25 == 0:
            print(epoch, ':\tWdist:', np.array(Wdists).mean(), '\terr R: ', np.array(err_r).mean(), '\terr F: ', np.array(err_f).mean())
            train_log.write('%.4f\t%.4f\t%.4f\n' % (np.array(Wdists).mean() , np.array(err_r).mean(), np.array(err_f).mean()))
            train_log.flush()
            fig = plt.figure()
            x_array = np.array(range(len(plot_w)))
            plt.plot(x_array, np.array(plot_w), c=colors[0])
            plt.plot(x_array, np.array(plot_r), c=colors[1])
            plt.plot(x_array, np.array(plot_f), c=colors[2])
            plt.legend(['W dist', 'D(real)', 'D(fake)'])
            fig.savefig(opt.save + '/train.png')
            plt.close()
        if epoch % 30 == 0: #print_generated:
            print('Generated:')
            print(generated.data.cpu().numpy(), '\n')
            print('Critic costs:')
            print(costs.data.cpu().numpy(), '\n')
            print('Batch-averaged step-wise probs:')
            print(probs, '\n')
