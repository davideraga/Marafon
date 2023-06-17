from Agents.DQN_Agent import DQN_Agent
from Environment.Marafon_Env import *
import numpy as np
from Models.MarafonPolicyMasked import PolicyNet
from Agents.PPO_Agent import PPO_Agent
from Agents.RandomAgent import RandomAgent
from Models.Q_Network import QNet
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--agent_type', '-a', default="ppo", choices=["ppo", "dqn"] )
parser.add_argument('--vs_agent_type', '-vs', default="rnd", choices=["ppo", "dqn", "rnd"])
parser.add_argument('--n_games', '-n', default=1000, type=int)
args = parser.parse_args()
max_games = args.n_games
m_env = Marafon_Env(seed=103, verbose=False)#29 103
m_env.reset(0)
state = m_env.get_obs(0)

p = [0, 0, 0, 0]

if args.agent_type == "dqn":
    Q_net = QNet(input_shape=state.shape, num_actions=m_env.n_actions)
    Q_net(m_env.get_obs(0)[np.newaxis])
    #Q_net.load_weights('final_checkpoints/dqn_exploit_40')
    # Q_net.load_weights('final_checkpoints/dqn_sp_exploit_40')
    #Q_net.load_weights('final_checkpoints/dqn_4_40')
    Q_net.load_weights('final_checkpoints/dqn_sp_4_20')
    agent = DQN_Agent(training=False, Q_Net=Q_net)

if args.agent_type == "ppo":
    policy_net = PolicyNet(num_actions=m_env.n_actions, input_shape=state.shape)
    #policy_net.load_weights("final_checkpoints/ppo_sp_4_20_p").expect_partial()
    policy_net.load_weights("final_checkpoints/ppo_4_40_p").expect_partial()
    agent = PPO_Agent(policy_net=policy_net, training=False)


if args.vs_agent_type == "rnd":
    agent_1 = RandomAgent(seed=100)

if args.vs_agent_type == "ddqn":
    Q_net_1 = QNet(input_shape=state.shape, num_actions=m_env.n_actions)
    Q_net_1(m_env.get_obs(0)[np.newaxis])
    # Q_net.load_weights('final_checkpoints/dqn_4_40')
    Q_net.load_weights('final_checkpoints/dqn_sp_4_20')
    agent_1 = DQN_Agent(training=False, Q_Net=Q_net_1)

if args.vs_agent_type == "ppo":
    policy_net_1 = PolicyNet(num_actions=m_env.n_actions, input_shape=state.shape)
    # policy_net_1.load_weights("final_checkpoints/ppo_sp_4_20_p").expect_partial()
    policy_net_1.load_weights("final_checkpoints/ppo_4_40_p").expect_partial()
    agent_1 = PPO_Agent(policy_net=policy_net_1, training=False)

p[0] = agent
p[1] = agent_1
p[2] = agent
p[3] = agent_1
pt0 = 0
pt1 = 0
running_avg = 0
games = 0
episodes = 0
win_0 = 0
win_1 = 0
episodes = 0
s_pt0 = 0
f_pt0 = 0
f_pt1 = 0
s_pt1 = 0
sums = [0]*9
eps_game = 0
signs_done = 0
no_signs = 0
while games < max_games:
    starting_p = (eps_game + games)%4
    #starting_p = episodes%4
    m_env.reset(starting_p)
    #print(m_env.cards)
    #print (m_env.p_cards)
    actions, n = m_env.get_legal_actions()
    next_p = m_env.get_next_player()
    if isinstance(p[next_p], RandomAgent):
        action = p[next_p].choose_action(actions, n)
    else:
        obs = m_env.get_obs(next_p)
        action = p[next_p].choose_action(obs, actions, n)
        m_env.check_legal_action(action)
    #print (action)
    m_env.set_briscola(action)
    #print (m_env.briscola)
    for round in range(m_env.n_rounds):
        #print("new round")
        #print(round)
        for player in range(m_env.n_players):
            #state = m_env.get_obs_encoding(player)
            next_p = m_env.get_next_player()
            actions, n = m_env.get_legal_actions()
            if isinstance(p[next_p], RandomAgent):
                action = p[next_p].choose_action(actions, n)
            else:
                obs = m_env.get_obs(next_p)
                action = p[next_p].choose_action(obs, actions, n)
                m_env.check_legal_action(action)
            #print ("p"+ str(next_p)+ ": " + str(action)+"  "+str(m_env.p_cards[m_env.next_player][action]))
            #print(action)
            m_env.play_card(action)
            if player == 0:
                actions, n = m_env.get_legal_actions()
                if isinstance(p[next_p], RandomAgent):
                    action = p[next_p].choose_action(actions, n)
                else:
                    obs = m_env.get_obs(next_p)
                    action = p[next_p].choose_action(obs, actions, n)
                    m_env.check_legal_action(action)
                    if action == 15:
                        signs_done += 1
                    if action == 14:
                        no_signs += 1

                m_env.do_sign(action)
                #print("p" + str(next_p) + ": " + signs[m_env.sign])
        #print("reward "+str(m_env.get_turn_reward(0)))
        #print(m_env.get_payoff(0))
        #print(m_env.get_payoff(1))
        reward = m_env.get_payoff(0) - m_env.get_payoff(1)
        sums[round] += reward
    #print (m_env.past_rounds)
    print("--------")
    #if m_env.get_payoff(0)+m_env.get_payoff(1) >11:
     #   print("maraffa found")
    pt0+=m_env.get_payoff(0)
    pt1+=m_env.get_payoff(1)
    #print (m_env.cards_mask)
    print(pt0)
    print(pt1)
    eps_game+=1
    if starting_p %2 == 0:
        f_pt0+=m_env.get_payoff(0)
        s_pt1+=m_env.get_payoff(1)
    else:
        s_pt0+=m_env.get_payoff(0)
        f_pt1+=m_env.get_payoff(1)
    if pt0 >= 41 and pt0 > pt1:
        games += 1
        win_0 += 1
        pt0 = 0
        pt1 = 0
        eps_game = 0
        print("games " + str(games))
    if pt1 >= 41 and pt1 > pt0:
        games += 1
        win_1 += 1
        pt1 = 0
        pt0 = 0
        eps_game = 0
        print("games "+str(games))
    episodes+=1
    running_avg = running_avg * 0.99 + (m_env.get_payoff(0) - m_env.get_payoff(1)) * 0.01
    print(running_avg)
print("----------------------")
print(win_0)
print(win_1)
print("sign ratio : "+str(float(signs_done)/(signs_done+no_signs)))
print("team 0 "+str(2*f_pt0/episodes)+"  "+str(2*s_pt0/episodes))
print("team 0 "+str((f_pt0+s_pt0)/episodes))
print("team 1 "+str(2*f_pt1/episodes)+"  "+str(2*s_pt1/episodes))
print("team 1 "+str((f_pt1+s_pt1)/episodes))
for i in range(9):
    sums[i] /= episodes
print(sums)
