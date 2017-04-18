from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
from six.moves import xrange
import sys

import matplotlib
matplotlib.use('Agg')  # allows for saving images without display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import tensorflow as tf

import util


def _linear(args, output_size, bias=True, bias_start=0.0, scope=None, initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    Based on the code from TensorFlow."""
    if not isinstance(args, collections.Sequence) or isinstance(args, six.string_types):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype,
                                 initializer=initializer)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                    initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term


class Actor(object):
    '''The imitation GAN policy network.'''

    def __init__(self, opt):
        super(Actor, self).__init__()
        self.opt = opt
        actor_cell = tf.contrib.rnn.GRUCell(opt.actor_hidden_size)
        projection_cell = tf.contrib.rnn.OutputProjectionWrapper(actor_cell, opt.vocab_size,
                                                                 activation=tf.nn.log_softmax)
        # TODO tie weights with output linear
        self.cell = tf.contrib.rnn.EmbeddingWrapper(projection_cell, opt.vocab_size, opt.emb_size)
        self.eps_sample = False  # TODO do eps sampling

    def forward(self):
        outputs = []
        all_logprobs = []
        all_probs = []
        probs = []  # for debugging
        hidden = self.cell.zero_state(self.opt.batch_size, tf.float32)
        inputs = tf.zeros([self.opt.batch_size], dtype=tf.int32)
        with tf.variable_scope('recurrence') as scope:
            for out_i in xrange(self.opt.seq_len):
                dist, hidden = self.cell(inputs, hidden)
                print(dist.get_shape())  # FIXME
                all_logprobs.append(dist.unsqueeze(1))
                prob = torch.exp(dist)
                all_probs.append(prob.unsqueeze(1))
                prob_new = prob.detach()
                probs.append(prob.data.mean(0).squeeze(0).cpu().numpy())  # for debugging
                sampled = torch.multinomial(prob_new, 1)
                outputs.append(sampled)
                if out_i < self.opt.seq_len - 1:
                    inputs = self.embedding(sampled.squeeze(1))
                scope.reuse_variables()
        return (torch.cat(outputs, 1), torch.cat(all_logprobs, 1), torch.cat(all_probs, 1),
                np.array(probs))


class Critic(object):
    '''The imitation GAN discriminator/critic.'''

    def __init__(self, opt, gradient_penalize):
        super(Critic, self).__init__()
        self.opt = opt
        self.gradient_penalize = gradient_penalize
        self.gamma = opt.gamma
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
        self.embedding = tf.get_variable("embedding", [opt.vocab_size, opt.emb_size],
            initializer=initializer)
        self.cell = tf.contrib.rnn.GRUCell(opt.critic_hidden_size)
        if gradient_penalize:
            self.real = tf.placeholder(tf.int32, [opt.batch_size, opt.seq_len], name='real_actions')
            self.fake = tf.placeholder(tf.int32, [opt.batch_size, opt.seq_len], name='fake_actions')
        else:
            self.actions = tf.placeholder(tf.int32, [opt.batch_size, opt.seq_len], name='actions')

    def forward(self):
        if self.gradient_penalize:
            padded_real = torch.cat([self.zero_input, self.real], 1)
            padded_fake = torch.cat([self.zero_input, self.fake], 1)
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
            padded_actions = torch.cat([self.zero_input, self.actions], 1)
            inputs = self.embedding(Variable(padded_actions))
            onehot_actions = None
        outputs, _ = self.rnn(inputs, Variable(self.zero_state))
        outputs = outputs.contiguous()
        flattened = outputs.view(-1, self.opt.critic_hidden_size)
        flat_costs = self.cost(flattened)
        costs = flat_costs.view(self.opt.batch_size, self.opt.seq_len + 1, self.opt.vocab_size)
        costs = costs[:, :-1]  # account for the padding
        if self.gamma < 1.0 - 1e-8:
            discount = torch.cuda.FloatTensor([self.gamma ** i for i in xrange(self.opt.seq_len)])
            discount = discount.unsqueeze(0).expand(self.opt.batch_size, self.opt.seq_len)
            discount = Variable(discount)
            costs = costs * discount
        costs_abs = torch.abs(costs)
        if self.opt.smooth_zero > 1e-4:
            select = (costs_abs >= self.opt.smooth_zero).float()
            costs_abs = costs_abs - (self.opt.smooth_zero / 2)
            costs_sq = (costs ** 2) / (self.opt.smooth_zero * 2)
            return (select * costs_abs) + ((1.0 - select) * costs_sq), onehot_actions
        else:
            return costs_abs, onehot_actions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=1000000, help='number of iters to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seq_len', type=int, default=20, help='toy sequence length')
    parser.add_argument('--vocab_size', type=int, default=60, help='vocab size for data')
    parser.add_argument('--emb_size', type=int, default=32, help='embedding size')
    parser.add_argument('--actor_hidden_size', type=int, default=512, help='Actor RNN hidden size')
    parser.add_argument('--critic_hidden_size', type=int, default=512,
                        help='Critic RNN hidden size')
    parser.add_argument('--critic_layers', type=int, default=1)  # TODO + add actor_layers
    parser.add_argument('--critic_dropout', type=float, default=0.0)  # TODO + add actor_dropout
    parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon for eps sampling. results in biased policy gradient')
    parser.add_argument('--eps_for_critic', type=int, default=0,
                        help='enable eps sampling of actor during critic training')
    parser.add_argument('--actor_optimize_all', type=int, default=0,
                        help='optimize all actions per timestep (not only the selected ones)')
    parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
    parser.add_argument('--gamma_inc', type=float, default=0.0,
                        help='the amount by which to increase gamma at each turn')
    parser.add_argument('--entropy_reg', type=float, default=1.0,
                        help='policy entropy regularization')
    parser.add_argument('--critic_entropy_reg', type=float, default=0.0,  # <= 1e-3
                        help='critic entropy regularization')
    parser.add_argument('--smooth_zero', type=float, default=1.0,
                        help='s, use c^2/2s instead of c-(s/2) when abs critic score c<s')
    parser.add_argument('--use_advantage', type=int, default=1)
    parser.add_argument('--exp_replay_buffer', type=int, default=0,
                        help='use a replay buffer with an exponential distribution')
    parser.add_argument('--real_multiplier', type=float, default=1.0,
                        help='weight for real samples as compared to fake for critic learning')
    parser.add_argument('--replay_actors', type=int, default=10,  # higher with exp buffer
                        help='number of actors for experience replay')
    parser.add_argument('--replay_actors_half', type=int, default=3,
                        help='number of recent actors making up half of the exponential replay')
    parser.add_argument('--solved_threshold', type=int, default=200,
                        help='conseq steps the task (if appl) has been solved for before exit')
    parser.add_argument('--solved_max_fail', type=int, default=10,
                        help='maximum number of failures before solved streak is reset')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--gradient_penalty', type=float, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='norm for gradient clipping')
    parser.add_argument('--critic_iters', type=int, default=5,  # 20 or 25 for larger tasks
                        help='number of critic iters per turn')
    parser.add_argument('--actor_iters', type=int, default=1,  # 15 or 20 for larger tasks
                        help='number of actor iters per turn')
    parser.add_argument('--burnin', type=int, default=25, help='number of burnin iterations')
    parser.add_argument('--burnin_actor_iters', type=int, default=1)
    parser.add_argument('--burnin_critic_iters', type=int, default=100)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--task', type=str, default='lm', help='one of lm/longterm/words')
    parser.add_argument('--lm_data_dir', type=str, default='data/penn')
    parser.add_argument('--lm_char', type=int, default=1, help='1 for character level model')
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
    train_log = open(opt.save + '/train.log', 'w')
    colors = cm.rainbow(np.linspace(0, 1, 3))
    plot_r = []
    plot_f = []
    plot_w = []
    plot_cgnorm = []
    plot_agnorm = []

    opt.replay_size = opt.replay_actors * opt.batch_size * opt.critic_iters
    opt.replay_size_half = opt.replay_actors_half * opt.batch_size * opt.critic_iters

    np.set_printoptions(precision=4, threshold=10000, linewidth=200, suppress=True)

    if opt.task == 'words':
        task = util.WordsTask(opt.seq_len, opt.vocab_size)
    elif opt.task == 'longterm':
        task = util.LongtermTask(opt.seq_len, opt.vocab_size)
    elif opt.task == 'lm':
        task = util.LMTask(opt.seq_len, opt.vocab_size, opt.lm_data_dir, opt.lm_char)
        if task.vocab_size != opt.vocab_size:
            opt.vocab_size = task.vocab_size
            print('Updated vocab_size:', opt.vocab_size)
    else:
        print('error: invalid task name:', opt.task)
        sys.exit(1)

    with tf.variable_scope('Actor'):
        actor = Actor(opt)
    with tf.variable_scope('Critic') as scope:
        critic = Critic(opt, False)
        scope.reuse_variables()
        gp_critic = Critic(opt, True)

    assert opt.replay_size >= opt.batch_size
    if opt.exp_replay_buffer:
        buffer = util.ExponentialReplayMemory(opt.replay_size, opt.replay_size_half)
    else:
        buffer = util.ReplayMemory(opt.replay_size)

    if opt.optimizer == 'Adam':
        actor_optimizer = tf.train.AdamOptimizer(opt.learning_rate, beta1=opt.beta1,
                                                 beta2=opt.beta2)
        critic_optimizer = tf.train.AdamOptimizer(opt.learning_rate, beta1=opt.beta1,
                                                  beta2=opt.beta2)
    elif opt.optimizer == 'RMSprop':
        actor_optimizer = tf.train.AdamOptimizer(opt.learning_rate)
        critic_optimizer = tf.train.AdamOptimizer(opt.learning_rate)
    solved = 0
    solved_fail = 0

    print('\nReal examples:')
    task.display(task.get_data(opt.batch_size))
    print()
    plot_x = []
    for cur_iter in xrange(opt.niter):
        if solved >= opt.solved_threshold:
            print('%d: Task solved, exiting.' % cur_iter)
            break
        actor.eps_sample = bool(opt.eps_for_critic) and opt.eps > 1e-8

        # train critic
        for param in critic.parameters():  # reset requires_grad
            param.requires_grad = True  # they are set to False below in actor update
        if cur_iter < opt.burnin:
            critic_iters = opt.burnin_critic_iters
        else:
            critic_iters = opt.critic_iters
        Wdists = []
        err_r = []
        err_f = []
        critic_gnorms = []
        for critic_i in xrange(critic_iters):
            critic.zero_grad()

            generated, _, _, _ = actor()
            buffer.push(generated.data.cpu().numpy())
            generated = buffer.sample(opt.batch_size)
            generated = torch.from_numpy(generated).cuda()
            costs, _ = critic(generated)
            entropy = -((1e-6 + costs) * torch.log(1e-6 + costs)).sum() / opt.batch_size
            costs = costs.gather(2, Variable(generated.unsqueeze(2))).squeeze(2)
            E_generated = costs.sum() / opt.batch_size
            loss = -E_generated - (opt.critic_entropy_reg * entropy)
            loss.backward()

            real = torch.from_numpy(task.get_data(opt.batch_size)).cuda()
            costs, _ = critic(real)
            entropy = -((1e-6 + costs) * torch.log(1e-6 + costs)).sum() / opt.batch_size
            costs = costs.gather(2, Variable(real.unsqueeze(2))).squeeze(2)
            E_real = costs.sum() / opt.batch_size
            loss = (opt.real_multiplier * E_real) - (opt.critic_entropy_reg * entropy)
            loss.backward()

            costs, inputs = gp_critic((real, generated))
            loss = costs.sum() / opt.batch_size
            loss.backward(Variable(torch.ones(1).cuda(), requires_grad=True),
                          retain_variables=True)
            # TODO consider each pair individually instead of the sum. this one is incorrect.
            loss = opt.gradient_penalty * (torch.norm(inputs.grad) - 1) ** 2
            loss.backward()  # FIXME this doesn't work on pytorch yet

            critic_gnorms.append(util.gradient_norm(critic.parameters()))
            nn.utils.clip_grad_norm(critic.parameters(), opt.max_grad_norm)
            critic_optimizer.step()
            Wdist = (E_generated - E_real).data[0]
            Wdists.append(Wdist)
            err_r.append(E_real.data[0])
            err_f.append(E_generated.data[0])

        # train actor
        for param in critic.parameters():
            param.requires_grad = False  # to avoid computation
        if cur_iter < opt.burnin:
            actor_iters = opt.burnin_actor_iters
        else:
            actor_iters = opt.actor_iters
        if cur_iter % opt.gen_every == 0:
            # disable eps_sample since we intend to visualize a (noiseless) generation.
            print_generated = True
            actor.eps_sample = False
        else:
            print_generated = False
            actor.eps_sample = opt.eps > 1e-8

        actor_gnorms = []
        for actor_i in xrange(actor_iters):
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
            all_costs, _ = critic(all_generated.data)
            if print_generated:
                costs = all_costs[:-1]
            else:
                costs = all_costs
            if opt.use_advantage:
                baseline = (costs * all_probs).detach().sum(2).expand_as(costs)
                disadv = costs - baseline
            else:
                disadv = costs
            if opt.actor_optimize_all:
                # consider all possible actions at each timestep. this provides much more training
                # signal, possibly helping train faster. however, critic errors on less likely
                # actions can have a worse effect on actor training as compared to considering only
                # the selected action.
                loss = (all_probs.detach() * disadv * all_logprobs).sum() / \
                       (opt.batch_size - int(print_generated))
                disadv = disadv.gather(2, generated.unsqueeze(2)).squeeze(2)
            else:
                disadv = disadv.gather(2, generated.unsqueeze(2)).squeeze(2)
                loss = (disadv * logprobs).sum() / (opt.batch_size - int(print_generated))
            entropy = -(all_probs * all_logprobs).sum() / (opt.batch_size - int(print_generated))
            loss -= opt.entropy_reg * entropy
            loss.backward()
            actor_gnorms.append(util.gradient_norm(actor.parameters()))
            nn.utils.clip_grad_norm(actor.parameters(), opt.max_grad_norm)
            actor_optimizer.step()
            if print_generated:
                costs = all_costs.gather(2, all_generated.unsqueeze(2)).squeeze(2)
                # print generated only in the first actor iteration
                print('Generated (last row is real):')
                task.display(all_generated.data.cpu().numpy())
                print()
                print('Critic costs (last row is real):')
                print(costs.data.cpu().numpy(), '\n')
                print('Critic cost sums (last element is real):')
                print(costs.data.cpu().numpy().sum(1), '\n')
                if opt.use_advantage:
                    print('Critic advantages (real not included):')
                    print(-disadv.data.cpu().numpy(), '\n')
                if opt.task == 'longterm':
                    print('Batch-averaged step-wise probs:')
                    print(avgprobs, '\n')
                print_generated = False
                actor.eps_sample = opt.eps > 1e-8
        critic.gamma = min(critic.gamma + opt.gamma_inc, 1.0)  # FIXME

        if cur_iter % opt.print_every == 0:
            print(cur_iter, ':\tWdist:', np.array(Wdists).mean(), '\terr R:',
                  np.array(err_r).mean(), '\terr F:', np.array(err_f).mean(), '\tgamma:',
                  critic.gamma, '\tsolved:', solved, '\tsolved_fail:', solved_fail)
            train_log.write('%.4f\t%.4f\t%.4f\n' % (np.array(Wdists).mean(), np.array(err_r).mean(),
                            np.array(err_f).mean()))
            train_log.flush()
        if cur_iter % opt.plot_every == 0:
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
            plot_cgnorm.append(np.array(critic_gnorms).mean())
            fig = plt.figure()
            plt.plot(x_array, np.array(plot_agnorm), c=colors[0])
            plt.plot(x_array, np.array(plot_cgnorm), c=colors[1])
            plt.legend(['Actor grad norm', 'Critic grad norm'], loc=2)
            fig.savefig(opt.save + '/grads.png')
            plt.close()

        if opt.task == 'longterm':
            params = [avgprobs]
        elif opt.task == 'words':
            generated = generated.data.cpu().numpy()
            if print_generated and actor_iters == 1:
                generated = generated[:-1]
            params = [generated]
        else:
            params = [None]
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
