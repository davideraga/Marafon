import numpy as np
import tensorflow as tf
from tensorflow import keras
from Models.Q_Network import QNet
from collections import deque

def get_negative_action_mask(action_mask):
    return (1 - action_mask) * -1e8



class DQN_Agent:
    """Class that implements a Double (Deep) Q-Network agent"""
    def __init__(self, seed=0, replay_buffer=deque(maxlen=100000), batch_size=252, Q_Net=QNet(), target_Q_net=QNet(), training=True,
                 max_n_actions=16, steps_for_update=8, steps_for_target_ud=1000, updating=True, discount=1, epsilon=0.9, epsilon_decay=0.9999,
                 optimizer=keras.optimizers.Adam(learning_rate=1e-5, clipvalue=1)):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.Q_net = Q_Net
        self.target_Q_net = target_Q_net
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.optimizer = optimizer
        self.last_obs = None
        self.last_action = None
        self.training = training
        self.discount = discount
        self.max_n_actions = max_n_actions
        self.update_iterations = 0
        self.steps_for_update = steps_for_update
        self.rnd = np.random.RandomState(seed)
        self.steps_for_target_ud = steps_for_target_ud
        self.updating = updating
        self.steps = 0

    def choose_action(self, obs, action_mask, n_actions, reward=0):
        """this function  chooses the action based on epsilon greedy policy, stores exp and calls update,
        the reward is from the last transition"""
        action = 0
        illegal_action_mask = get_negative_action_mask(action_mask)
        if self.training and self.last_action != None:
            exp = (self.last_obs, self.last_action, reward, obs, illegal_action_mask, 0)#the last is done
            self.replay_buffer.append(exp)
            if self.updating and len(self.replay_buffer) > self.batch_size:
                if (self.steps % self.steps_for_update) == 0:
                    self.update()
                self.steps += 1
            self.epsilon = self.epsilon * self.epsilon_decay

        if n_actions == 1 or (self.training and self.rnd.rand() < self.epsilon):#random action
            if n_actions == 1:
                 r = 1
            else:
                r = self.rnd.randint(1, n_actions+1)
            c = 0
            for i in range(len(action_mask)):
                c += action_mask[i]
                if c == r:
                    action = i
                    break
        else:
            obs_t = tf.convert_to_tensor(obs)
            values = tf.squeeze(self.Q_net(tf.expand_dims(obs_t, 0)))+illegal_action_mask
            action = tf.argmax(values).numpy()
        if self.training:
            self.last_obs = obs
            self.last_action = action
        return action




    def update(self):
        """does one update step with stochastic gradient descent"""
        rnd_indx = self.rnd.randint(0, len(self.replay_buffer), self.batch_size)
        obs = []
        actions = []
        rewards = []
        next_obs = []
        illegal_action_masks = []
        dones = []
        for index in rnd_indx:
            exp = self.replay_buffer[index]
            obs.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_obs.append(exp[3])
            illegal_action_masks.append(exp[4])
            dones.append(exp[5])
        obs = tf.convert_to_tensor(obs, dtype="float32")
        next_obs = tf.convert_to_tensor(next_obs, dtype="float32")
        illegal_action_masks = tf.convert_to_tensor(illegal_action_masks, dtype="float32")
        actions = tf.convert_to_tensor(actions, dtype="int32")
        dones = tf.convert_to_tensor(dones, dtype="float32")
        rewards = tf.convert_to_tensor(rewards, dtype="float32")
        max_actions = tf.argmax(self.Q_net(next_obs) + illegal_action_masks, axis=1) #indexing
        next_Qs = tf.gather(self.target_Q_net(next_obs), max_actions, batch_dims=1, axis=1)
        targets = rewards + self.discount * (1 - dones) * next_Qs#td target: reward + discounted max of next obs
        with tf.GradientTape() as tape:
            Qs = self.Q_net(obs)
            Q_a = tf.gather(Qs, actions, batch_dims=1, axis=1)#indexing
            loss = tf.reduce_mean((targets - Q_a)**2)
        self.optimizer.minimize(loss, self.Q_net.trainable_variables, tape=tape)
        self.update_iterations += 1
        if (self.update_iterations % self.steps_for_target_ud) == 0:
            self.update_target()


    def update_target(self):
        self.target_Q_net.set_weights(self.Q_net.get_weights())

    def done(self, reward):
        """called at the end of the episode, stores the last exp"""
        if self.training:
            exp = (self.last_obs, self.last_action, reward, self.last_obs, np.zeros(self.max_n_actions), 1)
            self.replay_buffer.append(exp)
            if self.updating and len(self.replay_buffer) > self.batch_size:
                if (self.steps % self.steps_for_update) == 0:
                    self.update()
                self.steps += 1
            self.epsilon = self.epsilon * self.epsilon_decay
        self.last_obs = None
        self.last_action = None


