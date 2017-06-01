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
from torch import autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util


class Actor(nn.Module):
    '''The imitation GAN policy network (generator).'''

    def __init__(self, opt):
        super(Actor, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.cell = nn.GRUCell(opt.emb_size, opt.actor_hidden_size)
        self.dist = nn.Linear(opt.actor_hidden_size, opt.vocab_size)
        #self.dist1 = nn.Linear(opt.actor_hidden_size, opt.emb_size)
        #self.dist2 = nn.Linear(opt.emb_size, opt.vocab_size)
        #self.embedding.weight = self.dist2.weight  # tie weights
        self.zero_input = torch.LongTensor(opt.batch_size).zero_().cuda()
        self.zero_state = torch.zeros([opt.batch_size, opt.actor_hidden_size]).cuda()

    def forward(self):
        outputs = []
        all_logprobs = []
        all_probs = []
        probs = []  # for debugging
        hidden = Variable(self.zero_state)
        inputs = self.embedding(Variable(self.zero_input))
        for out_i in xrange(self.opt.seq_len):
            hidden = self.cell(inputs, hidden)
            #dist = F.log_softmax(self.dist2(self.dist1(hidden)))
            dist = F.log_softmax(self.dist(hidden))
            all_logprobs.append(dist.unsqueeze(1))
            prob = torch.exp(dist)
            all_probs.append(prob.unsqueeze(1))
            prob_new = prob.detach()
            probs.append(prob.data.mean(0).squeeze(0).cpu().numpy())  # for debugging
            sampled = torch.multinomial(prob_new, 1)
            outputs.append(sampled)
            if out_i < self.opt.seq_len - 1:
                inputs = self.embedding(sampled.squeeze(1))
        return (torch.cat(outputs, 1), torch.cat(all_logprobs, 1), torch.cat(all_probs, 1),
                np.array(probs))


class CostsNet(nn.Module):
    '''The imitation GAN costs network (discriminator).'''

    def __init__(self, opt):
        super(CostsNet, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.rnn = nn.GRU(input_size=opt.emb_size, hidden_size=opt.costsnet_hidden_size,
                          num_layers=opt.costsnet_layers, dropout=opt.costsnet_dropout,
                          batch_first=True)
        self.cost = nn.Linear(opt.costsnet_hidden_size, opt.vocab_size)
        self.zero_input = torch.LongTensor(opt.batch_size, 1).zero_().cuda()
        self.zero_state = torch.zeros([opt.costsnet_layers, opt.batch_size,
                                       opt.costsnet_hidden_size]).cuda()
        self.gradient_penalize = False

    def forward(self, actions):
        if self.gradient_penalize:
            # actions is tuple of (real_batch, fake_batch)
            real, fake = actions
            padded_real = torch.cat([self.zero_input, real], 1)
            padded_fake = torch.cat([self.zero_input, fake], 1)
            onehot_real = torch.zeros(padded_real.size() + (self.opt.vocab_size,)).cuda()
            onehot_fake = torch.zeros(padded_fake.size() + (self.opt.vocab_size,)).cuda()
            padded_real.unsqueeze_(2)
            padded_fake.unsqueeze_(2)
            onehot_real.scatter_(2, padded_real, 1)
            onehot_fake.scatter_(2, padded_fake, 1)
            alpha = torch.rand(real.size(0)).unsqueeze(1).unsqueeze(2).expand_as(onehot_real).cuda()
            onehot_actions = (alpha * onehot_real) + ((1 - alpha) * onehot_fake)
            onehot_actions = Variable(onehot_actions, requires_grad=True)
            inputs = torch.mm(onehot_actions.view(-1, self.opt.vocab_size), self.embedding.weight)
            inputs = inputs.view(onehot_actions.size(0), -1, self.opt.emb_size)
        else:
            padded_actions = torch.cat([self.zero_input, actions], 1)
            inputs = self.embedding(Variable(padded_actions))
            onehot_actions = None
        outputs, _ = self.rnn(inputs, Variable(self.zero_state))
        outputs = outputs.contiguous()
        flattened = outputs.view(-1, self.opt.costsnet_hidden_size)
        flat_costs = self.cost(flattened)
        costs = flat_costs.view(self.opt.batch_size, self.opt.seq_len + 1, self.opt.vocab_size)
        costs = costs[:, :-1]  # account for the padding
        costs_abs = torch.abs(costs)
        if self.opt.smooth_zero > 1e-4:
            select = (costs_abs >= self.opt.smooth_zero).float()
            costs_abs = costs_abs - (self.opt.smooth_zero / 2)
            costs_sq = (costs ** 2) / (self.opt.smooth_zero * 2)
            return (select * costs_abs) + ((1.0 - select) * costs_sq), onehot_actions
        else:
            return costs_abs, onehot_actions


class Critic(nn.Module):
    '''The imitation GAN critic used for stable training of the actor.'''

    def __init__(self, opt):
        super(Critic, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_size)
        self.rnn = nn.GRU(input_size=opt.emb_size, hidden_size=opt.critic_hidden_size,
                          num_layers=opt.critic_layers, dropout=opt.critic_dropout,
                          batch_first=True)
        self.Q = nn.Linear(opt.critic_hidden_size, opt.vocab_size)
        self.V = nn.Linear(opt.critic_hidden_size, 1)
        self.zero_input = torch.LongTensor(opt.batch_size, 1).zero_().cuda()
        self.zero_state = torch.zeros([opt.critic_layers, opt.batch_size,
                                       opt.critic_hidden_size]).cuda()

    def forward(self, actions):
        padded_actions = torch.cat([self.zero_input, actions], 1)
        inputs = self.embedding(Variable(padded_actions))
        outputs, _ = self.rnn(inputs, Variable(self.zero_state))
        outputs = outputs.contiguous()
        flattened = outputs.view(-1, self.opt.critic_hidden_size)
        flat_Q = self.Q(flattened)
        flat_V = self.V(flattened)
        Q = flat_Q.view(self.opt.batch_size, self.opt.seq_len + 1, self.opt.vocab_size)
        V = flat_V.view(self.opt.batch_size, self.opt.seq_len + 1)
        # account for the padding
        return Q[:, :-1], V[:, :-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_actor', type=str, default='', help='actor load file')
    parser.add_argument('--load_costsnet', type=str, default='', help='costsnet load file')
    parser.add_argument('--load_critic', type=str, default='', help='critic load file')
    parser.add_argument('--save_actor', type=str, default='',
                        help='actor save file. saves as actor.model in logs by default')
    parser.add_argument('--save_costsnet', type=str, default='',
                        help='costsnet save file. saves as costsnet.model in logs by default')
    parser.add_argument('--save_critic', type=str, default='',
                        help='critic save file. saves as critic.model in logs by default')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save every these many iters. -1 to disable')
    parser.add_argument('--save_overwrite', type=int, default=1, help='overwrite same save files')
    parser.add_argument('--niter', type=int, default=1000000, help='number of iters to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seq_len', type=int, default=8, help='sequence length')
    parser.add_argument('--vocab_size', type=int, default=60, help='vocab size for data')
    parser.add_argument('--emb_size', type=int, default=32, help='embedding size')
    parser.add_argument('--actor_hidden_size', type=int, default=256, help='Actor RNN hidden size')
    parser.add_argument('--costsnet_hidden_size', type=int, default=256,
                        help='CostsNet RNN hidden size')
    parser.add_argument('--critic_hidden_size', type=int, default=256,
                        help='Critic RNN hidden size')
    parser.add_argument('--costsnet_layers', type=int, default=1)
    parser.add_argument('--costsnet_dropout', type=float, default=0.0)
    parser.add_argument('--critic_layers', type=int, default=1)
    parser.add_argument('--critic_dropout', type=float, default=0.0)
    parser.add_argument('--freeze_actor', type=int, default=-1,
                        help='freeze actor after these many steps')
    parser.add_argument('--freeze_costsnet', type=int, default=-1,
                        help='freeze costsnet after these many steps')
    parser.add_argument('--freeze_critic', type=int, default=-1,
                        help='freeze critic after these many steps')
    # 1e-3 without decay for text, >1e-3 for toys:
    parser.add_argument('--entropy_reg', type=float, default=1e-3,  # crucial.
                        help='policy entropy regularization')
    parser.add_argument('--entropy_decay', type=float, default=1.0,
                        help='policy entropy regularization weight decay per turn')
    parser.add_argument('--entropy_reg_min', type=float, default=5e-4,
                        help='minimum policy entropy regularization')
    parser.add_argument('--costsnet_entropy_reg', type=float, default=0.0,  # <= 1e-3
                        help='costsnet entropy regularization')
    parser.add_argument('--smooth_zero', type=float, default=2e-3,  # 1e-2 for larger tasks
                        help='s, use c^2/2s instead of c-(s/2) when abs costsnet score c<s')
    parser.add_argument('--use_advantage', type=int, default=1)
    parser.add_argument('--exp_replay_buffer', type=int, default=0,
                        help='use a replay buffer with an exponential distribution')
    parser.add_argument('--real_multiplier', type=float, default=7.0,  # crucial
                        help='weight for real samples as compared to fake for costsnet learning')
    parser.add_argument('--replay_actors', type=int, default=10,  # higher with exp buffer
                        help='number of actors for experience replay')
    parser.add_argument('--replay_actors_half', type=int, default=3,
                        help='number of recent actors making up half of the exponential replay')
    parser.add_argument('--solved_threshold', type=int, default=10,  # 200 for complex tasks
                        help='conseq steps the task (if appl) has been solved for before exit')
    parser.add_argument('--solved_max_fail', type=int, default=3,  # 10 for complex tasks
                        help='maximum number of failures before solved streak is reset')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--gradient_penalty', type=float, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='norm for gradient clipping')
    parser.add_argument('--actor_iters', type=int, default=20,  # 15 or 20 for larger tasks
                        help='number of actor iters per turn')  # crucial
    parser.add_argument('--costsnet_iters', type=int, default=25,  # 20 or 25 for larger tasks
                        help='number of costsnet iters per turn')  # crucial
    parser.add_argument('--critic_iters', type=int, default=25,  # TODO investigate
                        help='number of critic iters per turn')  # crucial
    parser.add_argument('--burnin', type=int, default=25, help='number of burnin iterations')
    parser.add_argument('--burnin_actor_iters', type=int, default=1)
    parser.add_argument('--burnin_costsnet_iters', type=int, default=100)
    parser.add_argument('--burnin_critic_iters', type=int, default=100)  # TODO
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--task', type=str, default='lm', help='one of lm/longterm/words')
    parser.add_argument('--lm_data_dir', type=str, default='data/penn')
    parser.add_argument('--lm_char', type=int, default=1, help='1 for character level model')
    parser.add_argument('--lm_word_vocab', type=int, default=1000,
                        help='word vocab size for char LM')
    parser.add_argument('--lm_single_word', type=int, default=1, help='single word GAN')
    parser.add_argument('--print_every', type=int, default=25,
                        help='print losses every these many steps')
    parser.add_argument('--plot_every', type=int, default=1,
                        help='plot losses every these many steps')
    parser.add_argument('--gen_every', type=int, default=50,
                        help='generate sample every these many steps')
    opt = parser.parse_args()
    print(opt)

    # some logging stuff
    opt.save = 'logs/' + opt.name
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    if not opt.save_actor:
        opt.save_actor = opt.save + '/actor.model'
    if not opt.save_costsnet:
        opt.save_costsnet = opt.save + '/costsnet.model'
    if not opt.save_critic:
        opt.save_critic = opt.save + '/critic.model'
    train_log = open(opt.save + '/train.log', 'w')
    colors = cm.rainbow(np.linspace(0, 1, 3))
    plot_r = []
    plot_f = []
    plot_w = []
    plot_cgnorm = []
    plot_ctgnorm = []
    plot_agnorm = []

    # TODO replay for critic?
    opt.replay_size = opt.replay_actors * opt.batch_size * opt.costsnet_iters
    opt.replay_size_half = opt.replay_actors_half * opt.batch_size * opt.costsnet_iters

    cudnn.enabled = False
    np.set_printoptions(precision=4, threshold=10000, linewidth=200, suppress=True)

    if opt.task == 'words':
        task = util.WordsTask(opt.seq_len, opt.vocab_size)
    elif opt.task == 'longterm':
        task = util.LongtermTask(opt.seq_len, opt.vocab_size)
    elif opt.task == 'lm':
        task = util.LMTask(opt.seq_len, opt.vocab_size, opt.lm_data_dir, opt.lm_char,
                           opt.lm_word_vocab, opt.lm_single_word)
        if task.vocab_size != opt.vocab_size:
            opt.vocab_size = task.vocab_size
            print('Updated vocab_size:', opt.vocab_size)
    else:
        print('error: invalid task name:', opt.task)
        sys.exit(1)

    actor = Actor(opt)  #.apply(util.weights_init)
    costsnet = CostsNet(opt)  #.apply(util.weights_init)
    critic = Critic(opt)  #.apply(util.weights_init)
    actor.cuda()
    costsnet.cuda()
    critic.cuda()

    kwargs = {'lr': opt.learning_rate}
    if opt.optimizer == 'Adam':
        kwargs['betas'] = (opt.beta1, opt.beta2)
    actor_optimizer = getattr(optim, opt.optimizer)(actor.parameters(), **kwargs)
    costsnet_optimizer = getattr(optim, opt.optimizer)(costsnet.parameters(), **kwargs)
    critic_optimizer = getattr(optim, opt.optimizer)(critic.parameters(), **kwargs)

    if opt.load_actor:
        state_dict, optimizer_dict, actor_cur_iter = torch.load(opt.load_actor)
        actor.load_state_dict(state_dict)
        actor_optimizer.load_state_dict(optimizer_dict)
        print('Loaded actor from', opt.load_actor)
    else:
        actor_cur_iter = -1
    if opt.load_costsnet:
        state_dict, optimizer_dict, costsnet_cur_iter, buffer = torch.load(opt.load_costsnet)
        costsnet.load_state_dict(state_dict)
        costsnet_optimizer.load_state_dict(optimizer_dict)
        print('Loaded costsnet from', opt.load_costsnet)
    else:
        costsnet_cur_iter = -1
        assert opt.replay_size >= opt.batch_size
        if opt.exp_replay_buffer:
            buffer = util.ExponentialReplayMemory(opt.replay_size, opt.replay_size_half)
        else:
            buffer = util.ReplayMemory(opt.replay_size)
    if opt.load_actor:
        state_dict, optimizer_dict, critic_cur_iter = torch.load(opt.load_critic)
        critic.load_state_dict(state_dict)
        critic_optimizer.load_state_dict(optimizer_dict)
        print('Loaded critic from', opt.load_critic)
    else:
        critic_cur_iter = -1
    start_iter = min(actor_cur_iter, costsnet_cur_iter, critic_cur_iter) + 1

    solved = 0
    solved_fail = 0
    print('\nReal examples:')
    task.display(task.get_data(opt.batch_size))
    print()
    plot_x = []
    for cur_iter in xrange(start_iter, start_iter + opt.niter):
        if solved >= opt.solved_threshold:
            print('%d: Task solved, exiting.' % cur_iter)
            break

        # train costsnet
        train_costsnet = opt.freeze_costsnet < 0 or cur_iter < opt.freeze_costsnet
        if train_costsnet:
            # TODO is this to be moved to critic?
            for param in costsnet.parameters():  # reset requires_grad
                param.requires_grad = True  # they are set to False below in actor update
        if cur_iter < opt.burnin:
            costsnet_iters = opt.burnin_costsnet_iters
        else:
            costsnet_iters = opt.costsnet_iters
        Wdists = []
        err_r = []
        err_f = []
        costsnet_gnorms = []
        for costsnet_i in xrange(costsnet_iters):
            if train_costsnet:
                costsnet.zero_grad()

            generated, _, _, _ = actor()
            buffer.push(generated.data.cpu().numpy())
            generated = buffer.sample(opt.batch_size)
            generated = torch.from_numpy(generated).cuda()
            costs, _ = costsnet(generated)
            norm_costs = costs / costs.sum(2).expand_as(costs)
            if train_costsnet and opt.costsnet_entropy_reg > 0:
                entropy = -((1e-6 + norm_costs) * torch.log(1e-6 + norm_costs)).sum() / \
                          opt.batch_size
            else:
                entropy = 0.0
            costs = costs.gather(2, Variable(generated.unsqueeze(2))).squeeze(2)
            E_generated = costs.sum() / opt.batch_size
            if train_costsnet:
                loss = -E_generated - (opt.costsnet_entropy_reg * entropy)
                loss.backward()

            real = torch.from_numpy(task.get_data(opt.batch_size)).cuda()
            costs, _ = costsnet(real)
            norm_costs = costs / costs.sum(2).expand_as(costs)
            if train_costsnet and opt.costsnet_entropy_reg > 0:
                entropy = -((1e-6 + norm_costs) * torch.log(1e-6 + norm_costs)).sum() / \
                          opt.batch_size
            else:
                entropy = 0.0
            costs = costs.gather(2, Variable(real.unsqueeze(2))).squeeze(2)
            E_real = costs.sum() / opt.batch_size
            if train_costsnet:
                loss = (opt.real_multiplier * E_real) - (opt.costsnet_entropy_reg * entropy)
                loss.backward()

            if train_costsnet and opt.gradient_penalty > 0:
                costsnet.gradient_penalize = True
                costs, inputs = costsnet((real, generated))
                costs = costs * inputs[:, 1:]
                loss = ((opt.real_multiplier + 1) / 2) * costs.sum()
                inputs_grad, = autograd.grad([loss], [inputs], create_graph=True)
                inputs_grad = inputs_grad.view(opt.batch_size, -1)
                norm_sq = (inputs_grad ** 2).sum(1)
                norm_errors = norm_sq - 2 * torch.sqrt(norm_sq) + 1
                loss = opt.gradient_penalty * norm_errors.sum() / opt.batch_size
                loss.backward()
                costsnet.gradient_penalize = False

            costsnet_gnorms.append(util.gradient_norm(costsnet.parameters()))
            if train_costsnet:
                if opt.max_grad_norm > 0:
                    nn.utils.clip_grad_norm(costsnet.parameters(), opt.max_grad_norm)
                costsnet_optimizer.step()
            Wdist = (E_generated - E_real).data[0]
            Wdists.append(Wdist)
            err_r.append(E_real.data[0])
            err_f.append(E_generated.data[0])

        # TODO train critic
        critic_gnorms = []
        train_critic = opt.freeze_costsnet < 0 or cur_iter < opt.freeze_costsnet

        # train actor
        train_actor = opt.freeze_actor < 0 or cur_iter < opt.freeze_actor
        for param in costsnet.parameters():
            param.requires_grad = False  # to avoid computation
        if not train_actor or cur_iter < opt.burnin:
            actor_iters = opt.burnin_actor_iters
        else:
            actor_iters = opt.actor_iters
        if cur_iter % opt.gen_every == 0:
            print_generated = True
        else:
            print_generated = False
        entropy_reg = max(opt.entropy_reg * (opt.entropy_decay ** cur_iter), opt.entropy_reg_min)

        actor_gnorms = []
        for actor_i in xrange(actor_iters):
            if train_actor:
                actor.zero_grad()
            all_generated, all_logprobs, all_probs, avgprobs = actor()
            if print_generated:  # last sample is real, for debugging. do not train on it!
                all_generated = torch.cat([all_generated[:-1],
                                           torch.from_numpy(task.get_data(1)).cuda()], 0)
                all_logprobs = all_logprobs[:-1]
                all_probs = all_probs[:-1]
                generated = all_generated[:-1]
            else:
                generated = all_generated
            logprobs = all_logprobs.gather(2, generated.unsqueeze(2)).squeeze(2)
            all_costs, _ = costsnet(all_generated.data)
            if print_generated:
                costs = all_costs[:-1]
            else:
                costs = all_costs
            costs = costs.gather(2, generated.unsqueeze(2)).squeeze(2)
            # TODO advantage can only be computed with critic implemented
#            if opt.use_advantage:
#                baseline = (costs * all_probs).detach().sum(2).expand_as(costs)
#                disadv = costs - baseline
#            else:
#                disadv = costs
#            disadv = disadv.gather(2, generated.unsqueeze(2)).squeeze(2)
            # TODO optimize_all can be done to train actor using critic
            # FIXME following is Monte Carlo
            disadv = costs - costs.cumsum(1) + costs.sum(1).expand_as(costs)
            if train_actor:
                loss = (disadv * logprobs).sum() / (opt.batch_size - int(print_generated))
            if train_actor:
                entropy = -(all_probs * all_logprobs).sum() / \
                          (opt.batch_size - int(print_generated))
                loss -= entropy_reg * entropy
                loss.backward()
            actor_gnorms.append(util.gradient_norm(actor.parameters()))
            if train_actor:
                if opt.max_grad_norm > 0:
                    nn.utils.clip_grad_norm(actor.parameters(), opt.max_grad_norm)
                actor_optimizer.step()
            if print_generated:
                # print generated only in the first actor iteration
                print('Generated (last row is real):')
                task.display(all_generated.data.cpu().numpy())
                print()
                print('CostsNet costs (last row is real):')
                print(costs.data.cpu().numpy(), '\n')
                print('CostsNet cost sums (last element is real):')
                print(costs.data.cpu().numpy().sum(1), '\n')
                if opt.use_advantage:
                    # FIXME incorrect.
                    print('Critic advantages (real not included):')
                    print(-disadv.data.cpu().numpy(), '\n')
                if opt.task == 'longterm':
                    print('Batch-averaged step-wise probs:')
                    print(avgprobs, '\n')
                print_generated = False

        if cur_iter % opt.print_every == 0:
            extra = []
            if not train_actor:
                extra.append('actor frozen')
            if not train_costsnet:
                extra.append('costsnet frozen')
            if not train_critic:
                extra.append('critic frozen')
            extra = ', '.join(extra)
            print(cur_iter, ':\tWdist:', np.array(Wdists).mean(), '\terr R:',
                  np.array(err_r).mean(), '\terr F:', np.array(err_f).mean(), '\tentropy_reg:',
                  entropy_reg, '\tsolved:', solved, '\tsolved_fail:', solved_fail,
                  '\t[' + extra + ']')
            train_log.write('%.4f\t%.4f\t%.4f\n' % (np.array(Wdists).mean(), np.array(err_r).mean(),
                            np.array(err_f).mean()))
            train_log.flush()
        if cur_iter and cur_iter % opt.plot_every == 0:
            plot_x.append(cur_iter)
            plot_r.append(np.array(err_r).mean())
            plot_f.append(np.array(err_f).mean())
            plot_w.append(np.array(Wdists).mean())
            fig = plt.figure()
            x_array = np.array(plot_x)
            plt.plot(x_array, np.array(plot_w), c=colors[0])
            plt.plot(x_array, np.array(plot_r), c=colors[1])
            plt.plot(x_array, np.array(plot_f), c=colors[2])
            plt.legend(['W dist', 'D(real)', 'D(fake)'], loc=2)
            fig.savefig(opt.save + '/train.png')
            plt.close()

            plot_agnorm.append(np.array(actor_gnorms).mean())
            plot_cgnorm.append(np.array(costsnet_gnorms).mean())
            plot_ctgnorm.append(np.array(critic_gnorms).mean())
            fig = plt.figure()
            plt.plot(x_array, np.array(plot_agnorm), c=colors[0])
            plt.plot(x_array, np.array(plot_cgnorm), c=colors[1])
            plt.plot(x_array, np.array(plot_ctgnorm), c=colors[2])
            plt.legend(['Actor grad norm', 'CostsNet grad norm', 'Critic grad norm'], loc=2)
            fig.savefig(opt.save + '/grads.png')
            plt.close()

        params = [None]
        if opt.task == 'longterm':
            params = [avgprobs]
        elif opt.task == 'words' or opt.task == 'lm':
            generated = generated.data.cpu().numpy()
            params = [generated]
        if task.solved(*params):
            solved += 1
        else:
            reset = True
            if solved > 0:
                reset = False
                solved_fail += 1
                if solved_fail >= opt.solved_max_fail:
                    reset = True
            if reset:
                solved = 0
                solved_fail = 0
        if opt.save_every > 0 and cur_iter and cur_iter % opt.save_every == 0:
            print('Saving model...')
            save_actor = opt.save_actor
            save_costsnet = opt.save_costsnet
            save_critic = opt.save_critic
            if not opt.save_overwrite:
                save_actor += ('.%d' % cur_iter)
                save_costsnet += ('.%d' % cur_iter)
                save_critic += ('.%d' % cur_iter)
            with open(save_actor, 'wb') as f:
                states = [actor.state_dict(), actor_optimizer.state_dict(), cur_iter]
                torch.save(states, f)
                print('Saved actor to', save_actor)
            with open(save_costsnet, 'wb') as f:
                states = [costsnet.state_dict(), costsnet_optimizer.state_dict(), cur_iter, buffer]
                torch.save(states, f)
                print('Saved costsnet to', save_costsnet)
            with open(save_critic, 'wb') as f:
                states = [critic.state_dict(), critic_optimizer.state_dict(), cur_iter]
                torch.save(states, f)
                print('Saved critic to', save_critic)
