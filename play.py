from Agents.DQN_Agent import DQN_Agent
from Environment.Marafon_Env import *
import numpy as np
from Models.MarafonPolicyMasked import PolicyNet
from Agents.PPO_Agent import PPO_Agent
import time
import argparse
from Models.Q_Network import QNet


def convert_human_action(action_mask, n_action): #converts the action from human to how the env wants it
    if action < 0:
        quit()
    c = 0
    for i in range(len(action_mask)):
        c += action_mask[i]
        if c == n_action + 1:
            return i, True
    return 0, False


parser = argparse.ArgumentParser(description='')
parser.add_argument('--agent_type', '-a', default="ppo", choices=["ppo", "dqn"])
parser.add_argument('--n_games', '-n', default=1, type=int)

args = parser.parse_args()

max_games = args.n_games

delay = True
m_env = Marafon_Env(seed=int(time.time()), verbose=True)
m_env.reset(0)
state = m_env.get_obs(0)
p = [0, 0, 0, 0]


if args.agent_type == "dqn":
    Q_net = QNet(input_shape=state.shape, num_actions=m_env.n_actions)
    Q_net(m_env.get_obs(0)[np.newaxis])
    Q_net.load_weights("./final_checkpoints/dqn_sp_4_20")
    agent = DQN_Agent(training=False, Q_Net=Q_net)

if args.agent_type == "ppo":
    policy_net = PolicyNet(num_actions=m_env.n_actions, input_shape=state.shape)
    policy_net.load_weights("./final_checkpoints/ppo_sp_4_20_p").expect_partial()
    agent = PPO_Agent(policy_net=policy_net, training=False)


p[0] = "human"
p[1] = agent
p[2] = agent
p[3] = agent
pt0=0
pt1=0

print("you are p0, your teammate is p2")
starting_rnd = np.random.randint(0, 4)
episode = 0
games = 0
while games < max_games:
    starting_p=(episode+starting_rnd)%4
    m_env.reset(starting_p)
    #print(m.cards)
    #print (m.p_cards)
    actions, n = m_env.get_legal_actions()
    next_p = m_env.get_next_player()
    if p[next_p] == "human":
        m_env.show_cards_human(next_p)
        briscola_msg = "choose the briscola: 0: bastoni, 1: coppe, 2: denara, 3: spade "
        action = int(input(briscola_msg))
        good_action = False
        while not good_action:
            action, good_action = convert_human_action(actions, action)
            if not good_action:
                action = int(input(briscola_msg))
    else:
        obs = m_env.get_obs(next_p)
        action = p[next_p].choose_action(obs, actions, n)
    #print (action)
    m_env.set_briscola(action)
    #print (m.briscola)
    for round in range(m_env.n_rounds):
        #print("new round")
        #print(round)
        for player in range(m_env.n_players):
            #state = m.get_obs_encoding(player)
            next_p = m_env.get_next_player()
            actions, n = m_env.get_legal_actions()
            if p[next_p] == "human":
                m_env.show_cards_human(next_p)
                card_msg = "play card (insert the number): "
                action = int(input(card_msg))
                good_action = False
                while not good_action:
                    action, good_action = convert_human_action(actions, action)
                    if not good_action:
                        action = int(input(card_msg))
            else:
                obs = m_env.get_obs(next_p)
                action = p[next_p].choose_action(obs, actions, n)
            #print ("p"+ str(next_p)+ ": " + str(action)+"  "+str(m.p_cards[m.next_player][action]))
            #print(action)
            m_env.play_card(action)
            if player == 0:
                actions, n = m_env.get_legal_actions()
                if p[next_p] == "human":
                    sign_msg = "do you want do do a sign? 0: no 1: yes"
                    action = int(input(sign_msg))
                    good_action = False
                    while not good_action:
                        action, good_action = convert_human_action(actions, action)
                        if not good_action:
                            action = int(input(sign_msg))
                else:
                    obs = m_env.get_obs(next_p)
                    action = p[next_p].choose_action(obs, actions, n)
                m_env.do_sign(action)
            if delay:
                time.sleep(1)
                #print("p" + str(next_p) + ": " + signs[m.sign])
        #print("reward "+str(m.get_turn_reward(0)))
        #print(m.get_payoff(0))
        #print(m.get_payoff(1))
        if delay:
            time.sleep(2)
    #print (m.past_rounds)
    #if m.get_payoff(0)+m.get_payoff(1) >11:
     #   print("maraffa found")
    #print(m_env.get_payoff(0))
    #print(m_env.get_payoff(1))
    pt0+=m_env.get_payoff(0)
    pt1+=m_env.get_payoff(1)
    #print (m.cards_mask)
    print(pt0)
    print(pt1)
    print("--------")
    episode += 1
    if pt0 >= 41 and pt0 > pt1:
        games += 1
        pt0 = 0
        pt1 = 0
        print("You win!")
        print("----------------------")
        print("games " + str(games))
    if pt1 >= 41 and pt1 > pt0:
        games += 1
        pt1 = 0
        pt0 = 0
        print("You lose")
        print("----------------------")
        print("games "+str(games))
    if delay:
        time.sleep(2)

print("----------------------")
