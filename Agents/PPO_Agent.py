import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from Models.MarafonPolicyMasked import PolicyNet
from Models.MarafonV import VNet


def get_negative_action_mask(action_mask):
    mask = (1 - action_mask.astype("float32")) * -1e8
    return mask


class PPO_Agent:
    """Class that implements a Proximal Policy Optimization Clip Agent"""
    def __init__(self, seed=0, V_net=VNet(), policy_net=PolicyNet(), training=True, max_actions=16, buffer=[], updating=True, gae=True, V_loss_clipping=False, adv_norm=False, discount=1,
                 lambda_coef=0.95, n_minibatches=4, minibatch_size=512, n_epochs=4,
                 clip_ratio=0.1, entropy_weight=0.001, V_clip_range=0.5,
                 V_optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipvalue=1),
                 P_optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1)):
        self.V_net = V_net
        self.policy_net = policy_net
        self.rnd = np.random.RandomState(seed)
        self.training = training
        self.rewards = []
        self.obs = []
        self.actions = []
        self.masks = []
        self.probs = []
        self.discount = discount
        self.lambda_coef = lambda_coef
        self.V_optimizer = V_optimizer
        self.P_optimizer = P_optimizer
        self.max_actions = max_actions
        self.buffer = buffer
        self.n_minibatches = n_minibatches
        self.minibatch_size = minibatch_size
        self.buffer_size = self.minibatch_size*self.n_minibatches
        self.n_epochs = n_epochs
        self.clip_ratio = clip_ratio
        self.entropy_weight = entropy_weight
        self.gae = gae
        self.V_loss_clipping = V_loss_clipping
        self.V_clip_range = V_clip_range
        self.updating = updating
        self.adv_norm = adv_norm



    def choose_action(self, obs, action_mask, n_actions, reward=0):
        """this function chooses an action based on the policy and stores exp
         reward is from the last transition
        """
        action = 0
        mask = tf.convert_to_tensor(get_negative_action_mask(action_mask), dtype="float32")
        if n_actions == 1:
            for i in range(len(action_mask)):
                if action_mask[i] == 1:
                    action = i
                    break
        else:
            obs_t = tf.convert_to_tensor(obs)
            prob = tf.squeeze(self.policy_net(tf.expand_dims(obs_t, 0), mask))
            distr = tfp.distributions.Categorical(probs=prob)
            action = distr.sample().numpy()
        if self.training:
            if len(self.actions) > 0:
                self.rewards.append(reward)
            self.actions.append(action)
            self.obs.append(obs)
            self.masks.append(mask)
            if n_actions == 1:
                self.probs.append(1)
            else:
                self.probs.append(prob[action])
        return action


    def done(self, reward):
        """this function ends the episode, calculates the advantages and returns
        if the buffer has reached the size it calls the update,
        since the episodes are short and of about the same length,
        I decided not to truncate the episodes
        """
        if self.training:
            self.rewards.append(reward)
            n_steps = len(self.rewards)
            returns = np.zeros(n_steps)
            advs = np.zeros(n_steps)
            V = tf.squeeze(self.V_net(tf.convert_to_tensor(self.obs, dtype="float32")))
            adv = self.rewards[n_steps - 1] - V[n_steps - 1]
            tot_rew = self.rewards[n_steps - 1]
            advs[n_steps - 1] = adv
            returns[n_steps - 1] = tot_rew
            for i in range(n_steps - 2, -1, -1):
                tot_rew = tot_rew * self.discount + self.rewards[i]
                if self.gae:
                    adv = adv * self.discount * self.lambda_coef + self.rewards[i] + self.discount * V[i + 1] - V[i]
                else:
                    adv = tot_rew - V[i]
                advs[i] = adv
                returns[i] = tot_rew
            for i in range(len(returns)):
                if self.V_loss_clipping:
                    exp = (self.obs[i], self.actions[i], self.masks[i], self.probs[i], returns[i], advs[i], V[i])
                else:
                    exp = (self.obs[i], self.actions[i], self.masks[i], self.probs[i], returns[i], advs[i])
                self.buffer.append(exp)
            self.rewards.clear()
            self.obs.clear()
            self.actions.clear()
            self.masks.clear()
            self.probs.clear()
            if self.updating and len(self.buffer) >= self.buffer_size:
                self.update()
                self.buffer.clear()


    def update(self):
        """
        ppo update
        for each epoch:
            shuffle the buffer and divide it into minibatches
            for each minibatch: do the clipped update
        """
        for epoch in range(self.n_epochs):
            mbatch_start = 0
            self.rnd.shuffle(self.buffer)
            for minibatch in range(self.n_minibatches):
                if self.V_loss_clipping:
                    obs, actions, illegal_action_masks, old_probs, returns, advs , old_values= zip(*self.buffer[mbatch_start: mbatch_start + self.minibatch_size])
                else:
                    obs, actions, illegal_action_masks, old_probs, returns , advs = zip(*self.buffer[mbatch_start: mbatch_start+self.minibatch_size])
                returns = tf.convert_to_tensor(returns, dtype="float32")
                obs = tf.convert_to_tensor(obs, dtype="float32")
                actions = tf.convert_to_tensor(actions, dtype="int32")
                illegal_action_masks = tf.convert_to_tensor(illegal_action_masks, dtype="float32")
                legal_action_mask = tf.cast(tf.equal(illegal_action_masks, 0), dtype="float32")
                advs = np.array(advs, dtype="float32")
                old_probs = np.array(old_probs)
                if self.adv_norm:
                    advs = (advs - tf.math.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-10)
                if self.V_loss_clipping:
                    with tf.GradientTape() as V_tape:
                        V = tf.squeeze(self.V_net(obs))
                        V_clipped = old_values + tf.clip_by_value(V-old_values, clip_V_min=-self.V_clip_range, clip_V_max=self.V_clip_range)
                        V_loss_noclip = (returns - V)**2
                        V_loss_clip = (returns - V_clipped)**2
                        V_loss = tf.reduce_mean(tf.math.maximum(V_loss_clip, V_loss_noclip))
                    self.V_optimizer.minimize(V_loss,  self.V_net.trainable_variables, tape=V_tape)
                else:
                    with tf.GradientTape() as V_tape:
                        V = tf.squeeze(self.V_net(obs))
                        V_loss = tf.reduce_mean((returns - V)**2)
                    self.V_optimizer.minimize(V_loss,  self.V_net.trainable_variables, tape=V_tape)

                with tf.GradientTape() as P_tape:
                    new_probs = self.policy_net(obs, illegal_action_masks)
                    neg_entropy = tf.reduce_mean(tf.reduce_sum(legal_action_mask * new_probs * tf.math.log(new_probs+1e-10), axis=1))#minus entropy, the small constant is to avoid log(0)
                    new_probs_a = tf.gather(new_probs, actions, batch_dims=1, axis=1)
                    ratio = new_probs_a / old_probs
                    ratio_clipped = tf.clip_by_value(ratio, clip_value_min=1 - self.clip_ratio, clip_value_max=1+self.clip_ratio)
                    p_loss = -tf.reduce_mean(tf.math.minimum(advs*ratio, advs*ratio_clipped)) + neg_entropy*self.entropy_weight
                self.P_optimizer.minimize(p_loss, self.policy_net.trainable_variables, tape=P_tape)
                mbatch_start += self.minibatch_size



