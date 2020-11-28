import math
import itertools
import collections
import logging
import numpy as np
import pandas as pd
import gym
import tensorflow as tf
from tensorflow import keras
import env
from keras import backend as K

ATTACKER = 0
DEFENDER = 1
CUT = 0
CONNECT = 1
env = gym.make('network_game-v0')
ACTION_SPACE = 4

def strfboard(state, render_characters='xo', end='\n'):
    """
    Format a board as a string

    """
    board, Graph, player, budget, pass_count = state
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            index = int(board[x,y])
            c = render_characters[index]
            s += c
        s += end
    play_strings = ['A','D']
    s += play_strings[player]
    s += str(budget[0])
    s += ','
    s += str(budget[1])
    s += ','
    s += str(pass_count)
    s += end
    return s[:-len(end)]

def residual(x, filters, kernel_sizes=3, strides=1, activations='relu',
            regularizer=keras.regularizers.l2(1e-4)):
    shortcut = x
    for i, filte in enumerate(filters):
        kernel_size = kernel_sizes if isinstance(kernel_sizes, int) \
                else kernel_sizes[i]
        stride = strides if isinstance(strides, int) else strides[i]
        activation = activations if isinstance(activations, str) \
                else activations[i]
        z = keras.layers.Conv2D(filte, kernel_size, strides=stride,
                padding='same', kernel_regularizer=regularizer,
                bias_regularizer=regularizer)(x)
        y = keras.layers.BatchNormalization()(z)
        if i == len(filters) - 1:
            y = keras.layers.Add()([shortcut, y])
        x = keras.layers.Activation(activation)(y)
    return x


class AlphaZeroAgent:
    def __init__(self, env, batches=1, batch_size=4096,
                 kwargs={}, load=None, sim_count=800,
                 c_init=1.25, c_base=19652., prior_exploration_fraction=0.25):
        self.env = env
        self.board = env.board
        self.batches = batches
        self.batch_size = batch_size

        self.net = self.build_network(**kwargs)
        self.reset_mcts()
        self.sim_count = sim_count  # MCTS times
        self.c_init = c_init  # PUCT coefficient
        self.c_base = c_base  # PUCT coefficient
        self.prior_exploration_fraction = prior_exploration_fraction

    def build_network(self, conv_filters, residual_filters, policy_filters,
                      learning_rate=0.001, regularizer=keras.regularizers.l2(1e-4)):
        # public part
        m,n = self.board.shape
        inputs = keras.Input(shape=m*n)
        x = keras.layers.Reshape(self.board.shape + (1,))(inputs)
        for conv_filter in conv_filters:
            z = keras.layers.Conv2D(conv_filter, 3, padding='same',
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(x)
            y = keras.layers.BatchNormalization()(z)
            x = keras.layers.ReLU()(y)
        for residual_filter in residual_filters:
            x = residual(x, filters=residual_filter, regularizer=regularizer)
        intermediates = x

        # probability part
        for policy_filter in policy_filters:
            z = keras.layers.Conv2D(policy_filter, 3, padding='same',
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(x)
            y = keras.layers.BatchNormalization()(z)
            x = keras.layers.ReLU()(y)
        logits = keras.layers.Conv2D(1, 3, padding='same',
                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        flattens = keras.layers.Flatten()(logits)
        softmaxs = keras.layers.Softmax()(flattens)
        # probs = keras.layers.Reshape(self.board.shape)(softmaxs)
        probs = keras.layers.Dense(ACTION_SPACE)(softmaxs)

        # value part
        z = keras.layers.Conv2D(1, 3, padding='same',
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer)(intermediates)
        y = keras.layers.BatchNormalization()(z)
        x = keras.layers.ReLU()(y)
        flattens = keras.layers.Flatten()(x)
        vs = keras.layers.Dense(1, activation=keras.activations.tanh,
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer)(flattens)
        model = keras.Model(inputs=inputs, outputs=[probs, vs])

        def categorical_crossentropy_2d(y_true, y_pred):
            labels = tf.reshape(y_true, [-1, ACTION_SPACE])
            preds = tf.reshape(y_pred, [-1, ACTION_SPACE])
            return keras.losses.categorical_crossentropy(labels, preds)

        loss = [categorical_crossentropy_2d, keras.losses.MSE]
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def reset_mcts(self):
        def zero_board_factory():  # construct default_dict
            return np.zeros_like([0,1,2,3], dtype=float)

        self.q = collections.defaultdict(zero_board_factory)
        # q value estimation: board -> actionspace
        self.count = collections.defaultdict(zero_board_factory)
        # q value count: board -> board
        self.policy = {}
        self.valid = {}
        self.winner = {}

    def decide(self, observation, greedy=False, return_prob=False):
        # compute policy
        board, _, player,_ ,_ = observation
        s = strfboard(observation)
        while self.count[s].sum() < self.sim_count:
            self.search(observation, prior_noise=True)
        prob = self.count[s] / self.count[s].sum()

        # sample
        location_index = np.random.choice(prob.size, p=prob.reshape(-1))
        location = np.unravel_index(location_index, prob.shape)
        if return_prob:
            return location, prob
        return location

    def learn(self, dfs):
        df = pd.concat(dfs).reset_index(drop=True)
        for batch in range(self.batches):
            indices = np.random.choice(len(df), size=self.batch_size)
            d = df.loc[indices, 'player']
            players = np.stack(d)
            d = df.loc[indices, 'prob']
            probs = np.stack(d)
            d = df.loc[indices, 'winner']
            winners = np.stack(d)
            d = df.loc[indices, 'board']
            boards = d.values
            canonical_boards = boards
            #canonical_boards = np.array(boards, dtype=np.float)

            vs = (players * winners)[:, np.newaxis]
            probs = tf.convert_to_tensor(probs)
            vs = tf.convert_to_tensor(vs)
            y = [probs, vs]
            m,n = self.board.shape
            temp_board = []
            for item in canonical_boards:
                temp_array = np.array(item)
                temp_array = temp_array.reshape(1,m*n)
                temp_board.append(temp_array[0].tolist())

            temp_board_tensor = tf.convert_to_tensor(temp_board)

            #canonical_boards = tf.convert_to_tensor(canonical_boards)
            self.net.fit(temp_board_tensor, y, verbose=0)  # training

        self.reset_mcts()

    def search(self, observation, prior_noise=False):  # MCTS search
        board, Graph, player, budget, pass_count = observation
        s = strfboard(observation)
        if s not in self.winner:
            self.winner[s] = self.env.get_winner(observation)  # get winner
        if self.winner[s] is not None:  # confirm winner
            return self.winner[s]

        if s not in self.policy:  # never visited leaf node
            pis, vs = self.net.predict(board[np.newaxis])
            pi, v = pis[0], vs[0]
            valid = self.env.get_valid(observation)
            masked_pi = pi * valid
            total_masked_pi = np.sum(masked_pi)
            if total_masked_pi <= 0:  # all actions dont have a prob, sometimes happen
                masked_pi = valid  # workaround
                total_masked_pi = np.sum(masked_pi)
            self.policy[s] = masked_pi / total_masked_pi
            self.valid[s] = valid
            return v

        # PUCT upper bound
        count_sum = self.count[s].sum()
        coef = (self.c_init + np.log1p((1 + count_sum) / self.c_base)) * math.sqrt(count_sum) / (1. + self.count[s])
        if prior_noise:  # prior noise
            alpha = 1. / self.valid[s].sum()
            noise = np.random.gamma(alpha, 1., ACTION_SPACE)
            noise *= self.valid[s]
            noise /= noise.sum()
            prior = (1. - self.prior_exploration_fraction) * self.policy[s] + self.prior_exploration_fraction * noise
        else:
            prior = self.policy[s]
        ub = np.where(self.valid[s], self.q[s] + coef * prior, np.nan)
        location_index = np.nanargmax(ub)
        location = np.unravel_index(location_index, ACTION_SPACE)
        canonical_board = 1* board
        canonical_Graph = Graph.copy()
        canonical_player = 1*player
        canonical_budget = [budget[0], budget[1]]
        canonical_pass_count = 1*pass_count
        canonical_observation = (canonical_board,canonical_Graph,canonical_player,canonical_budget,canonical_pass_count)
        next_observation, _, _, _= self.env.next_step(canonical_observation, np.array(location))
        next_board,_, next_player, next_budget, next_pass_count = next_observation
        next_v = self.search(next_observation)
        v = next_v
        self.count[s][location] += 1
        self.q[s][location] += (v - self.q[s][location]) / self.count[s][location]
        return v


def self_play(env, agent, return_trajectory=False, verbose=False):
    if return_trajectory:
        trajectory = []
    observation = env.reset()
    for step in itertools.count():
        board,_,player,_,_ = observation
        action, prob = agent.decide(observation, return_prob=True)
        if verbose:
            print(strfboard(observation))
            logging.info('The {} step：palyer {}, action {}'.format(step, player,
                    action))
        observation, winner, done, _ = env.step(action[0])
        if return_trajectory:
            m,n = board.shape
            board = np.reshape(board, m*n)
            trajectory.append((player, board, prob))
        if done:
            if verbose:
                print(strfboard(observation))
                logging.info('Winner {}'.format(winner))
            break
    if return_trajectory:
        df_trajectory = pd.DataFrame(trajectory,
                columns=['player', 'board', 'prob'])
        df_trajectory['winner'] = winner
        return df_trajectory
    else:
        return winner

train_iterations = 200
train_episodes_per_iteration = 20
batches = 2
batch_size = 32
sim_count = 200
net_kwargs = {}
net_kwargs['conv_filters'] = [256,]
net_kwargs['residual_filters'] = [[256, 256],]
net_kwargs['policy_filters'] = [256,]

agent = AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=sim_count,
        batches=batches, batch_size=batch_size)



for iteration in range(train_iterations):
    # self-playing
    dfs_trajectory = []
    for episode in range(train_episodes_per_iteration):
        print('episode: ', episode+1)
        df_trajectory = self_play(env, agent,
                                  return_trajectory=True, verbose=False)
        logging.info('train {} turn {}: collect {} experience'.format(
            iteration, episode, len(df_trajectory)))
        dfs_trajectory.append(df_trajectory)
    # learn from experience
    agent.learn(dfs_trajectory)
    logging.info('train {}: learning finished'.format(iteration))

    # show result
    self_play(env, agent, verbose=True)