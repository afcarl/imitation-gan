from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from six.moves import xrange
import sys

import matplotlib
matplotlib.use('Agg')  # allows for saving images without display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nnutils  # remove after updating pytorch
import torch.optim as optim

import util


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
        self.eps_sample = True  # do eps sampling

    def forward(self):
        outputs = []
        corrections = []
        logprobs = []
        dists = []
        probs = []  # for debugging
        hidden = Variable(self.zero_state)
        inputs = self.embedding(Variable(self.zero_input))
        for out_i in xrange(self.opt.seq_len):
            hidden = self.cell(inputs, hidden)
            dist = F.log_softmax(self.dist(hidden))
            dists.append(dist.unsqueeze(1))
            dist_new = dist.detach()
            prob = torch.exp(dist_new)
            probs.append(prob.data.mean(0).squeeze(0).cpu().numpy())  # for debugging
            if self.eps_sample:
                dist_new = dist_new.clone()
                draw_randomly = self.opt.eps >= torch.rand([self.opt.batch_size])
                draw_randomly = draw_randomly.byte().unsqueeze(1).cuda().expand_as(dist_new)
                # set uniform distribution with opt.eps probability
                dist_new[draw_randomly] = -np.log(self.opt.vocab_size)
            # eps sampling
            # torch.multinomial is broken, so this is the workaround
            _, sampled = torch.max(dist_new.data -
                                   torch.log(-torch.log(torch.rand(*dist_new.size()).cuda())), 1)
            sampled = Variable(sampled, requires_grad=False)
            logprob = dist.gather(1, sampled)
            onpolicy_prob = prob.gather(1, sampled)
            if self.eps_sample:
                offpolicy_prob = torch.exp(dist_new.gather(1, sampled))
            else:
                offpolicy_prob = onpolicy_prob
            # avoid 0/0
            onpolicy_prob = onpolicy_prob.clamp(1e-8, 1.0)
            offpolicy_prob = offpolicy_prob.clamp(1e-8, 1.0)
            outputs.append(sampled)
            # use importance sampling to correct for eps sampling
            corrections.append(onpolicy_prob / offpolicy_prob)
            logprobs.append(logprob)
            if out_i < self.opt.seq_len - 1:
                inputs = self.embedding(sampled.squeeze(1))
        return (torch.cat(outputs, 1), torch.cat(corrections, 1), torch.cat(logprobs, 1),
                torch.cat(dists, 1), np.array(probs))


class Critic(nn.Module):
    '''The imitation GAN discriminator/critic.'''

    def __init__(self, opt):
        super(Critic, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.rnn = nn.GRU(input_size=opt.emb_size, hidden_size=opt.hidden_size, num_layers=1,
                          batch_first=True)
        self.cost = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.zero_input = torch.LongTensor(opt.batch_size, 1).zero_().cuda()
        self.zero_state = torch.zeros([1, opt.batch_size, opt.hidden_size]).cuda()
        self.gamma = opt.gamma
        self.normalize_rewards = False

    def forward(self, actions):
        actions = Variable(actions, requires_grad=False)
        padded_actions = torch.cat([Variable(self.zero_input, requires_grad=False), actions], 1)
        inputs = self.embedding(padded_actions)
        outputs, _ = self.rnn(inputs, Variable(self.zero_state))
        outputs = outputs.contiguous()
        flattened = outputs.view(-1, self.opt.hidden_size)
        flat_costs = self.cost(flattened)
        if self.normalize_rewards:
            # subtract mean to reduce variance. consider the mean to be a constant, and don't
            # backprop through it
            flat_costs -= flat_costs.mean(1).expand_as(flat_costs).detach()
        costs = flat_costs.view(self.opt.batch_size, self.opt.seq_len + 1, self.opt.vocab_size)
        costs = costs[:, :-1]  # account for the padding
        costs = costs.gather(2, actions.unsqueeze(2)).squeeze(2)
        if self.gamma < 1.0 - 1e-8:
            discount = torch.cuda.FloatTensor([self.gamma ** i for i in xrange(self.opt.seq_len)])
            discount = discount.unsqueeze(0).expand(self.opt.batch_size, self.opt.seq_len)
            discount = Variable(discount, requires_grad=False)
            costs = costs * discount
        return costs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seq_len', type=int, default=15, help='toy sequence length')
    parser.add_argument('--vocab_size', type=int, default=6,
                        help='character vocab size for toy data')
    parser.add_argument('--emb_size', type=int, default=32, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=128, help='RNN hidden size')
    parser.add_argument('--eps', type=float, default=0.0, help='epsilon for eps sampling')
    parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
    parser.add_argument('--gamma_inc', type=float, default=0.0,
                        help='the amount by which to increase gamma at each turn')
    parser.add_argument('--entropy_reg', type=float, default=1.0,
                        help='policy entropy regularization')
    parser.add_argument('--reward_reg', type=float, default=1.0,
                        help='critic reward regularization')
    parser.add_argument('--reward_reg_norm', type=int, default=2)
    parser.add_argument('--normalize_rewards', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='norm for gradient clipping')
    parser.add_argument('--clamp_limit', type=float, default=-1,
                        help='critic param clamping. -1 to disable')
    parser.add_argument('--critic_iters', type=int, default=5,
                        help='number of critic iters per turn')
    parser.add_argument('--actor_iters', type=int, default=1, help='number of actor iters per turn')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--task', type=str, default='longterm', help='longterm or words')
    parser.add_argument('--print_every', type=int, default=25,
                        help='print losses every these many steps')
    parser.add_argument('--gen_every', type=int, default=25,
                        help='generate sample every these many steps')
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
    np.set_printoptions(precision=4, threshold=10000, linewidth=200, suppress=True)

    if opt.task == 'words':
        get_data = util.get_toy_data_words
    elif opt.task == 'longterm':
        get_data = util.get_toy_data_longterm
    else:
        print('error: invalid task name:', opt.task)
        sys.exit(1)

    actor = Actor(opt)  #.apply(util.weights_init)
    critic = Critic(opt)  #.apply(util.weights_init)
    actor.cuda()
    critic.cuda()

    actor_optimizer = optim.RMSprop(actor.parameters(), lr=opt.learning_rate)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=opt.learning_rate)

    print('\nReal examples:')
    print(get_data(opt.batch_size, opt.seq_len, opt.vocab_size), '\n')
    plot_x = []
    for epoch in xrange(opt.niter):
        actor.eps_sample = opt.eps > 1e-8

        # train critic
        for param in critic.parameters():  # reset requires_grad
            param.requires_grad = True  # they are set to False below in actor update
        critic.normalize_rewards = False
        if epoch < 25 or epoch % 500 == 0:
            critic_iters = 100
        else:
            critic_iters = opt.critic_iters
        Wdists = []
        err_r = []
        err_f = []
        for critic_i in xrange(critic_iters):
            if opt.clamp_limit > 0:
                for param in critic.parameters():
                    param.data.clamp_(-opt.clamp_limit, opt.clamp_limit)
            critic.zero_grad()

            # eps sampling here can help the critic get signal from less likely actions as well.
            # corrections would ensure that the critic doesn't have to worry about such actions
            # too much though.
            generated, corrections, _, _, _ = actor()
            costs = critic(generated.data) * corrections
            E_generated = costs.sum() / opt.batch_size
            gen_loss = -E_generated + \
                       opt.reward_reg * costs.norm(opt.reward_reg_norm) / opt.batch_size
            gen_loss.backward()

            real = torch.from_numpy(get_data(opt.batch_size, opt.seq_len,
                                             opt.vocab_size)).cuda()
            costs = critic(real)
            E_real = costs.sum() / opt.batch_size
            real_loss = E_real + opt.reward_reg * costs.norm(opt.reward_reg_norm) / opt.batch_size
            real_loss.backward()

            nnutils.clip_grad_norm(critic.parameters(), opt.max_grad_norm)
            critic_optimizer.step()
            Wdist = (E_generated - E_real).data[0]
            Wdists.append(Wdist)
            err_r.append(E_real.data[0])
            err_f.append(E_generated.data[0])

        # train actor
        for param in critic.parameters():
            param.requires_grad = False  # to avoid computation
        critic.normalize_rewards = bool(opt.normalize_rewards)
        if epoch < 25:
            actor_iters = 1
        else:
            actor_iters = opt.actor_iters
        if epoch % opt.gen_every == 0:
            # disable eps_sample since we intend to visualize a (noiseless) generation.
            print_generated = True
            actor.eps_sample = False
        else:
            print_generated = False

        for actor_i in xrange(actor_iters):
            actor.zero_grad()
            generated, corrections, logprobs, all_logprobs, avgprobs = actor()
            if print_generated:
                generated.data[-1].copy_(torch.from_numpy(get_data(1, opt.seq_len,
                                                                   opt.vocab_size)).cuda())
                corrections[-1].data.zero_()
                all_logprobs = all_logprobs[:-1]
            costs = critic(generated.data)
            loss = (costs * corrections * logprobs).sum() / opt.batch_size
            all_probs = torch.exp(all_logprobs)
            entropy = -(all_probs * all_logprobs).sum() / opt.batch_size
            loss -= opt.entropy_reg * entropy
            loss.backward()
            nnutils.clip_grad_norm(actor.parameters(), opt.max_grad_norm)
            actor_optimizer.step()
            if print_generated:
                # print generated only in the first actor iteration
                print('Generated:')
                print(generated.data.cpu().numpy(), '\n')
                if opt.normalize_rewards:
                    print('Critic disadvantages:')
                else:
                    print('Critic costs:')
                print(costs.data.cpu().numpy(), '\n')
                print('Critic cost sums:')
                print(costs.data.cpu().numpy().sum(1), '\n')
                if opt.task == 'longterm':
                    print('Batch-averaged step-wise probs:')
                    print(avgprobs, '\n')
                print_generated = False
                actor.eps_sample = opt.eps > 1e-8
        critic.gamma = min(critic.gamma + opt.gamma_inc, 1.0)

        plot_x.append(epoch)
        plot_r.append(-np.array(err_r).mean())
        plot_f.append(-np.array(err_f).mean())
        plot_w.append(np.array(Wdists).mean())
        if epoch % opt.print_every == 0:
            print(epoch, ':\tWdist:', np.array(Wdists).mean(), '\terr R: ', np.array(err_r).mean(),
                  '\terr F: ', np.array(err_f).mean(), '\t gamma: ', critic.gamma)
            train_log.write('%.4f\t%.4f\t%.4f\n' % (np.array(Wdists).mean(), np.array(err_r).mean(),
                            np.array(err_f).mean()))
            train_log.flush()
            fig = plt.figure()
            x_array = np.array(plot_x)
            plt.plot(x_array, np.array(plot_w), c=colors[0])
            plt.plot(x_array, np.array(plot_r), c=colors[1])
            plt.plot(x_array, np.array(plot_f), c=colors[2])
            plt.legend(['W dist', 'D(real)', 'D(fake)'])
            fig.savefig(opt.save + '/train.png')
            plt.close()
