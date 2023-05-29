from Agents.PPO_Agent import PPO_Agent
from Environment.Marafon_Env import *
from Models.MarafonPolicyMasked import PolicyNet
from Models.MarafonV import VNet
from Models.Q_Network import QNet
from Agents.DQN_Agent import DQN_Agent
from Agents.RandomAgent import RandomAgent
from collections import deque
from utils import AverageQueue
import pickle
import argparse
from tensorflow import keras

parser = argparse.ArgumentParser(description='')
parser.add_argument('--agent_type', '-a', default="dqn", choices=["ppo", "dqn"])
parser.add_argument('--self_play', '-sp', default="no", choices=["yes", "no"])
parser.add_argument('--n_episodes', '-n', default=400000, type=int)
args = parser.parse_args()

max_episodes = args.n_episodes
n_episodes_save = 20000
m_env = Marafon_Env(seed=40, verbose=False, auto_last_round=True)#29
m_env.reset(0)
state = m_env.get_obs(0)
p = [0, 0, 0, 0]

if args.agent_type == "dqn":
    name = "dqn_prova"
    Q_net = QNet(input_shape=state.shape, num_actions=m_env.n_actions)
    target_net = QNet(num_actions=m_env.n_actions, input_shape=state.shape)
    target_net.set_weights(Q_net.get_weights())
    exp_replay_buffer = deque(maxlen=100000)
    batch_size = 256
    eps_decay = 0.9999
    learning_rate = 8e-5
    if args.self_play == "yes":
        steps_for_update = 4
    else:
        steps_for_update = 8
    steps_for_target_ud = 1000
    clip_value = 1
    discount = 1
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=clip_value)
    max_n_actions = 16 #environment specific constant
    p[0] = DQN_Agent(seed=0, replay_buffer=exp_replay_buffer, batch_size=batch_size, Q_Net=Q_net, target_Q_net=target_net, training=True, steps_for_update=steps_for_update,
                     steps_for_target_ud=steps_for_target_ud, updating=True, epsilon_decay=eps_decay, optimizer=optimizer, discount=discount, max_n_actions=max_n_actions)
    p[2] = DQN_Agent(seed=0, replay_buffer=exp_replay_buffer,  Q_Net=Q_net,  training=True, updating=False, max_n_actions=max_n_actions, epsilon_decay=eps_decay)
    if args.self_play == "yes":
        p[1] = DQN_Agent(seed=0, replay_buffer=exp_replay_buffer,  Q_Net=Q_net,  training=True, updating=False, max_n_actions=max_n_actions, epsilon_decay=eps_decay)
        p[3] = DQN_Agent(seed=0, replay_buffer=exp_replay_buffer,  Q_Net=Q_net,  training=True, updating=False, max_n_actions=max_n_actions, epsilon_decay=eps_decay)


if args.agent_type == "ppo":
    name = "ppo_prova"
    policy_net = PolicyNet(num_actions=m_env.n_actions, input_shape=state.shape)
    V_net = VNet(state.shape)
    buffer = []
    mbatch_size = 512
    n_batches = 4
    n_epochs = 4
    entropy_weight = 0.001
    clip_ratio = 0.2
    V_learning_rate = 1e-5
    P_learning_rate = 1e-4
    gae = True
    discount = 1
    lambda_coef = 0.92
    clip_value = 1
    V_optimizer = keras.optimizers.Adam(learning_rate=V_learning_rate, clipvalue=clip_value)
    P_optimizer = keras.optimizers.Adam(learning_rate=P_learning_rate, clipvalue=clip_value)
    if args.self_play == "no":
        p[0] = PPO_Agent(seed=0, V_net=V_net, policy_net=policy_net, training=True, buffer=buffer, updating=False, gae=gae, discount=discount,
                         lambda_coef=lambda_coef, minibatch_size=mbatch_size, n_minibatches=n_batches, n_epochs=n_epochs, clip_ratio=clip_ratio)
        p[2] = PPO_Agent(seed=0, V_net=V_net, policy_net=policy_net, training=True, buffer=buffer, updating=True, P_optimizer=P_optimizer, V_optimizer=V_optimizer, gae=gae, discount=discount,
                         lambda_coef=lambda_coef, minibatch_size=mbatch_size, n_minibatches=n_batches, n_epochs=n_epochs, clip_ratio=clip_ratio, entropy_weight=entropy_weight)
    else:
        p[0] = PPO_Agent(seed=0, V_net=V_net, policy_net=policy_net, training=True, buffer=buffer, updating=False, gae=gae, discount=discount,
                         lambda_coef=lambda_coef, minibatch_size=mbatch_size, n_minibatches=n_batches, n_epochs=n_epochs, clip_ratio=clip_ratio)
        p[1] = PPO_Agent(seed=0, V_net=V_net, policy_net=policy_net, training=True, buffer=buffer, updating=False,  gae=gae, discount=discount,
                         lambda_coef=lambda_coef, minibatch_size=mbatch_size, n_minibatches=n_batches, n_epochs=n_epochs, clip_ratio=clip_ratio)
        p[2] = PPO_Agent(seed=0, V_net=V_net, policy_net=policy_net, training=True, buffer=buffer, updating=False,  gae=gae, discount=discount,
                         lambda_coef=lambda_coef, minibatch_size=mbatch_size, n_minibatches=n_batches, n_epochs=n_epochs, clip_ratio=clip_ratio)
        p[3] = PPO_Agent(seed=0, V_net=V_net, policy_net=policy_net, training=True, buffer=buffer, updating=True, P_optimizer=P_optimizer, V_optimizer=V_optimizer, gae=gae, discount=discount,
                         lambda_coef=lambda_coef, minibatch_size=mbatch_size, n_minibatches=n_batches, n_epochs=n_epochs, clip_ratio=clip_ratio, entropy_weight=entropy_weight)


if args.self_play == "no":
    p[1] = RandomAgent(100)
    p[3] = RandomAgent(100)



pt0=0
pt1=0
running_avg = 0
run_avgs = []
avg_q = AverageQueue(1000)
for episode in range(max_episodes):
    starting_p = episode % 4
    m_env.reset(starting_p)
    #print(m_env.cards)
    #print (m_env.p_cards)
    while not m_env.is_done():
        next_p = m_env.get_next_player()
        actions, n = m_env.get_legal_actions()
        if isinstance(p[next_p], RandomAgent):
            action = p[next_p].choose_action(actions, n)
        else:
            obs = m_env.get_obs(next_p)
            action = p[next_p].choose_action(obs, actions, n, m_env.get_turn_reward(next_p))
        #m_env.check_legal_action(action)
        m_env.do_action(action)
    for i in range(m_env.n_players):
        if not isinstance(p[i], RandomAgent):
            p[i].done(m_env.get_turn_reward(i))

    print("--------")
    #if m_env.get_payoff(0)+m_env.get_payoff(1) >11:
     #   print("maraffa found")
    pt0+=m_env.get_payoff(0)
    pt1+=m_env.get_payoff(1)
    # print(m_env.get_payoff(0))
    # print(m_env.get_payoff(1))
    print(pt0)
    print(pt1)
    #running_avg = running_avg * 0.999 + (m_env.get_payoff(0) - m_env.get_payoff(1)) * 0.001
    running_avg = avg_q.append(m_env.get_payoff(0) - m_env.get_payoff(1))
    print(running_avg)
    run_avgs.append(running_avg)
    if (episode+1) % n_episodes_save == 0:
        f_name = "./checkpoints/" + name + "_" + str(max_episodes // 10000)
        if args.agent_type == "dqn":
            p[0].Q_net.save_weights(f_name)
        if args.agent_type == "ppo":
            p[0].policy_net.save_weights(f_name + "_p")
            p[0].V_net.save_weights(f_name + "_v")
        with open("./training_data/" + name, "wb") as f:
            pickle.dump(run_avgs, f)

f_name = "./checkpoints/"+name + "_" + str(max_episodes // 10000)
if args.agent_type == "dqn":
    p[0].Q_net.save_weights(f_name)
if args.agent_type == "ppo":
    p[0].policy_net.save_weights(f_name+"_p")
    p[0].V_net.save_weights(f_name+"_v")
with open("./training_data/" + name, "wb") as f:
    pickle.dump(run_avgs, f)
print("----------------------")
print(pt0)
print(pt1)
