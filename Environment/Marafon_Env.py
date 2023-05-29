
import numpy as np
from dataclasses import dataclass

#suit: 0 bastoni, 1 coppe, 2 denara, 3 spade
ranks = ['4', '5', '6', '7', '8', '9', '10', '1', '2', '3']
suits = ["bastoni", "coppe", "denara", "spade"]
signs = ["nothing", " busso", "striscio", "volo"]

def card_to_suit(card):
    return card // 10

def card_to_suit_b(card, briscola): #returns suit modified to make the briscsola always the 0 suit
    return ((card // 10) - briscola + 4) % 4

def card_to_rank(card):
    return card % 10

def card_mask_to_n(card):
        for i in range (40):
            if card[i] == 1:
                return i
        return -1

def suit_b_trans(suit, briscola):#trasnforms the suit based on the briscola
    if briscola < 0:
        briscola = 0
    if suit == -1:
        return -1
    return (suit - briscola +4) % 4

def rank_to_value(rank):#calculates the value in points of the cards
    if rank == 7:
        return 3
    if rank >= 4:
        return 1
    return 0

def card_to_str(card):
    return ranks[card_to_rank(card)] + " of " + suits[card_to_suit(card)]

@dataclass
class round_m:
    first_player: int
    sign: int
    round_suit: int
    round_cards: np.ndarray


class Marafon_Env:
    """
    implements the logic of the game Maraffone a.k.a. Beccacino
    """
    def __init__(self, seed=0, verbose=False, auto_last_round=True):
        self.verbose = verbose
        self.auto_last_round = auto_last_round
        if self.auto_last_round:
            self.n_rounds = 9
        else:
            self.n_rounds = 10
        self.n_players = 4
        self.n_actions = 16
        self.rnd = np.random.RandomState(seed)
        self.last_reward = None
        self.team_points = None
        self.team_figures = None
        self.win_player = None
        self.past_rounds = None
        self.cards_mask = None
        self.briscola_played = None
        self.max_rank = None
        self.sign_to_do = None
        self.maraffa_player = None
        self.sign = None
        self.starting_player = None
        self.maraffa = None
        self.briscola = None
        self.cards = None
        self.p_cards = None
        self.first_player = None
        self.round_cards = None
        self.round_suit = None
        self.round = 0
        self.cards_played_t = 0
        self.next_player = 0
        self.done = 0

    def reset(self, starting_player):
        """this function resets things for a new hand/episode, it must be called before playing"""
        if self.verbose:
            print("new hand")
        self.starting_player = starting_player#the player who starts the hand
        self.first_player = starting_player#the player who plays first in the round
        self.team_points = np.zeros(2, dtype=int)
        self.team_figures = np.zeros(2, dtype=int)
        self.round_cards = np.full(4, -1, dtype=int)
        self.cards_played_t = 0
        self.next_player = starting_player
        self.round = 0
        self.maraffa = False
        self.maraffa_player=-1
        self.briscola= - 1
        self.round_suit = -1
        self.sign=0
        self.max_rank=-1
        self.win_player = starting_player#player who is winning the current round
        self.sign_to_do = False
        self.briscola_played = False
        self.cards_mask = np.zeros(40, dtype=int) #debug
        self.past_rounds = []
        self.last_reward = np.zeros(4, dtype=int)
        self.done = 0
        self.cards = np.arange(0, 40)
        self.rnd.shuffle(self.cards)
        self.p_cards = np.reshape(self.cards, [4, 10])#cards of each player
        self.p_cards.sort()


    def do_action(self, action):
        """this function does the right action based on input"""
        if action < 10:
            return self.play_card(action)
        if action < 14:
            return self.set_briscola(action)
        return self.do_sign(action)

    def set_briscola(self, briscola):
        self.briscola= briscola - 10
        if self.verbose:
            print("the briscola is "+str(suits[self.briscola]))
        for i in range(4):
            c = 0
            for card in self.p_cards[i]: #checking for maraffa
                if card_to_rank(card) >= 7 and card_to_suit(card) == self.briscola:
                    c += 1
            if c == 3:
                team = i%2
                opp_team = (i+1)%2
                self.maraffa_player = i
                self.team_figures[team] = 9
                self.maraffa = True
                if self.verbose:
                    print("p"+str(self.maraffa_player)+" reveals maraffa: +3 points")



    def play_card(self, position):
        card = self.p_cards[self.next_player][position]
        if self.verbose:
            print("p"+str(self.next_player)+" plays: "+card_to_str(card))
        self.cards_mask[card] =1
        suit = card_to_suit(card)
        rank = card_to_rank(card)
        if self.cards_played_t == 0:
            self.round_suit = suit
            if self.round != 9:
                self.sign_to_do = True
            self.max_rank = rank
        if suit == self.briscola: #calcling winning player
            if not self.briscola_played:
                self.win_player = self.next_player
                self.briscola_played = True
                self.max_rank = rank
            else:
                if rank > self.max_rank:
                    self.max_rank = rank
                    self.win_player = self.next_player
        if suit == self.round_suit and (not self.briscola_played):
            if rank > self.max_rank:
                self.max_rank = rank
                self.win_player = self.next_player

        self.round_cards[self.next_player] = card
        for i in range(10):
            if self.p_cards[self.next_player][i] == card:
                self.p_cards[self.next_player][i] = -1  #remove card from hand

        if not self.sign_to_do:
            self.next_player = (self.next_player+1)%4
        self.cards_played_t += 1
        if self.cards_played_t == 4:
            self.end_round()
        return 0

    def do_sign(self, action):
        c_suit=0
        c_big=0
        if action == 15:
            for card in self.p_cards[self.next_player]:
                if card_to_suit(card) == self.round_suit:
                    c_suit += 1
                    if card_to_rank(card) >= 7:
                        c_big += 1
            if c_big > 0:
                self.sign = 1 # busso - I have more cards of the same suit and a big card
            else:
                if c_suit > 0:
                    self.sign = 2# striscio - I have more cards of the same suit
                else:
                    self.sign = 3# volo - I fly - I have no more cards of the same suit
        else:
            self.sign = 0
        if self.verbose:
            print("p"+str(self.next_player)+" does sign: "+signs[self.sign])
        self.next_player = (self.next_player + 1) % 4
        self.sign_to_do = False

    def end_round(self):
        """this is called automatically by play card at the end of each round"""
        if self.verbose:
            print("round "+str(self.round)+" has ended")
        round_value = 0
        for card in self.round_cards: #giving winning team value
            rank = card_to_rank(card)
            round_value += rank_to_value(rank)
        if self.round == 9: #last round bonus
            round_value += 3

        win_team = self.win_player%2
        if self.verbose:
            print("p"+str(self.win_player)+" wins this round")
            print("")
        self.team_figures[win_team] += round_value
        self.past_rounds.append(round_m(self.first_player, self.sign, self.round_suit, np.copy(self.round_cards)))
        self.round += 1
        self.next_player = self.win_player #setting things for next round
        self.first_player = self.win_player
        self.cards_played_t = 0
        self.round_suit = -1
        self.sign = 0
        self.briscola_played = False
        self.max_rank = -1
        for i in range(4):
            self.round_cards[i] = -1

        if self.round == 9 and self.auto_last_round : #solving last round (playing the last card is forced)
            for i in range(4):
                for j in range(10):
                    card = self.p_cards[self.next_player][j]
                    if card >= 0:
                        #print("playing "+str(card))
                        self.play_card(j)
                        break

        if self.round == 10 and self.done == 0:
            self.done = 1
            if self.verbose:
                print("hand has ended")
                for i in range(2):
                    print("team "+str(i)+" points: "+str(self.get_payoff(i)))
                print("-------------------\n")

    def get_legal_actions(self):
        """gives a mask of legal actions, and the number of the legal actions"""
        actions = np.zeros(16)
        if self.briscola == -1:
            actions[10] = 1
            actions[11] = 1
            actions[12] = 1
            actions[13] = 1
            n_actions = 4
        else:
            if self.sign_to_do:
                n_actions = 2
                actions[14] = 1
                actions[15] = 1
            else:
                n_actions = 0
                if self.round_suit >= 0:
                    for i in range(10):
                        card = self.p_cards[self.next_player][i]
                        if card != -1 and card_to_suit(card) == self.round_suit:# have to answer with the same suit
                            actions[i] = 1
                            n_actions += 1
                if n_actions == 0: # if no card of the same suit you can play another or first player
                    for i in range(10):
                        card = self.p_cards[self.next_player][i]
                        if card != -1:
                            actions[i] = 1
                            n_actions += 1
        return actions, n_actions

    def get_obs(self, player):
        """gives an observation of the game state based on the player"""
        i_t = np.empty(4, dtype=int)  # indexes transformed based on player, 0 gives the player
        i_r = np.empty(4, dtype=int)  # indexes transformed based on player reversed, the player has value 0
        for i in range(4):
            i_t[i] = (player + i) % 4
            i_r[i] = (i - player + 4) % 4
        len_counter = 0
        if self.auto_last_round:
            state = np.zeros(780)# 780
        else:
            state = np.zeros(849)
        for card in self.p_cards[player]:
            if card >= 0:
                state[len_counter + card_to_suit_b(card, self.briscola)] = 1
                len_counter += 4
                state[len_counter + card_to_rank(card)] = 1
                len_counter += 10
            else:
                len_counter += 14
        state[len_counter + self.briscola + 1] = 1
        len_counter += 5
        if self.maraffa:
            state[len_counter] = 1
            state[len_counter + 1 + i_r[self.maraffa_player]] = 1
        len_counter += 5
        state[len_counter + self.round] = 1
        if self .auto_last_round:
            len_counter += 9
        else:
            len_counter += 10
        state[len_counter + i_r[self.starting_player]] = 1
        len_counter += 4
        state[len_counter + i_r[self.win_player]] = 1
        len_counter += 4
        state[len_counter + i_r[self.first_player]] = 1
        len_counter += 4
        state[len_counter + self.sign] = 1
        len_counter += 4
        state[len_counter + suit_b_trans(self.round_suit, self.briscola) + 1] = 1
        len_counter += 5
        for i in range(4):  # cards are positioned based on the player who played the card, relative to the player who is observing
            card = self.round_cards[i_t[i]]
            if card >= 0:
                state[len_counter + card_to_suit_b(card, self.briscola)] = 1
                len_counter += 4
                state[len_counter + card_to_rank(card)] = 1
                len_counter += 10
            else:
                len_counter += 14
        for past_round in reversed(self.past_rounds):
            state[len_counter + i_r[past_round.first_player]] = 1
            len_counter += 4
            state[len_counter + past_round.sign] = 1
            len_counter += 4
            state[len_counter + suit_b_trans(past_round.round_suit, self.briscola)] = 1
            len_counter += 4
            for i in range(4):
                card = past_round.round_cards[i_t[i]]
                state[len_counter + card_to_suit_b(card, self.briscola)] = 1
                len_counter += 4
                state[len_counter + card_to_rank(card)] = 1
                len_counter += 10
        #print(len_counter)
        return state


    def get_true_obs(self, player):
        "experimental full state observation"
        i_t = np.empty(4, dtype=int)  # indexes transformed based on player, 0 gives the player
        i_r = np.empty(4, dtype=int)  # indexes transformed based on player reversed, the player has value 0
        for i in range(4):
            i_t[i] = (player + i) % 4
            i_r[i] = (i - player + 4) % 4
        team = player % 2
        len_counter = 0
        state = np.zeros(780)  # 780
        for p in i_t:
            for card in self.p_cards[p]:
                if card >= 0:
                    state[len_counter + card_to_suit_b(card, self.briscola)] = 1
                    len_counter += 4
                    state[len_counter + card_to_rank(card)] = 1
                    len_counter += 10
                else:
                    len_counter += 14
        state[len_counter + self.briscola + 1] = 1
        len_counter += 5
        if self.maraffa:
            state[len_counter] = 1
            state[len_counter + 1 + i_r[self.maraffa_player]] = 1
        len_counter += 5
        state[len_counter + self.round] = 1
        len_counter += 9
        state[len_counter + i_r[self.starting_player]] = 1
        len_counter += 4
        state[len_counter + i_r[self.win_player]] = 1
        len_counter += 4
        state[len_counter + i_r[self.first_player]] = 1
        len_counter += 4
        state[len_counter + self.sign] = 1
        len_counter += 4
        state[len_counter + suit_b_trans(self.round_suit, self.briscola) + 1] = 1
        len_counter += 5
        for i in range(
                4):  # cards are positioned based on the player who played the card, relative to the player who is observing
            card = self.round_cards[i_t[i]]
            if card >= 0:
                state[len_counter + card_to_suit_b(card, self.briscola)] = 1
                len_counter += 4
                state[len_counter + card_to_rank(card)] = 1
                len_counter += 10
            else:
                len_counter += 14
        # print(len_counter)
        return state

    def get_first_player(self):
        """first player in the round"""
        return self.first_player

    def get_next_player(self):
        return self.next_player

    def get_payoff(self, player):
        """this functions gives the cumulative reward of the episode"""
        return self.team_figures[player%2] // 3

    def get_turn_reward(self, player, figures=False):
        """this function gives a reward based on the difference of points earned since it was last called from the same player
        , figures=True gives a floating point reward"""
        team = player%2
        opp_team = (player+1)%2
        if figures:
            reward = (self.team_figures[team] / 3) - (self.team_figures[opp_team] / 3)
        else:
            reward = (self.team_figures[team]//3) - (self.team_figures[opp_team]//3)
        res = reward-self.last_reward[player]
        self.last_reward[player] = reward
        return res

    def is_done(self):
        return self.done

    def check_legal_action(self, action):
        mask, n = self.get_legal_actions()
        assert mask[action] == 1, "illegal action "+str(action)

    def show_cards_human(self, player):
        c = 0
        i = 0
        legal_actions, num = self.get_legal_actions()
        for card in self.p_cards[player]:
            if card >= 0:
                if self.briscola < 0 or legal_actions[i] == 1:
                    print(str(c)+": "+card_to_str(card))#cards that you can play
                    c += 1
                else:
                    print("-: " + card_to_str(card)) #cards that you can't play right now
            i += 1

"""
#debug test
m = Marafon_Env(30, auto_last_round=True, verbose=True)
from Agents.RandomAgent import RandomAgent
signs = ["nothing", "knock", "swipe", "fly"]
p = [0, 0, 0, 0]
p[0] = RandomAgent(100)
p[1] = RandomAgent(100)
p[2] = RandomAgent(100)
p[3] = RandomAgent(100)
pt0=0
pt1=0
s_pt0=0
f_pt0=0
f_pt1=0
s_pt1=0

for episode in range(10):
    starting_p=episode%4
    m.reset(starting_p)
    #print(m.cards)
    print (m.p_cards)
    actions, n = m.get_legal_actions()
    action = p[starting_p].choose_action(actions, n)
    #print (action)
    m.set_briscola(action)
    print (m.briscola)
    for round in range(m.n_rounds):
        #print("new round")
        #print(round)
        for player in range(4):
            next_p = m.get_next_player()
            state = m.get_obs(next_p)
            actions, n = m.get_legal_actions()
            #print(next_p)
            action = p[next_p].choose_action(actions, n)
            #print ("p"+ str(next_p)+ ": " + str(action)+"  "+str(m.p_cards[m.next_player][action]))
            m.play_card(action)
            if player == 0 and round != 9:
                #state = m.get_obs(next_p)
                actions, n = m.get_legal_actions()
                action = p[next_p].choose_action(actions, n)
                m.do_sign(action)
                #print("p" + str(next_p) + ": " + signs[m.sign])

        print(m.get_payoff(0))
        print(m.get_payoff(1))
    print (m.past_rounds)
    print("--------")
    pt0+=m.get_payoff(0)
    pt1+=m.get_payoff(1)
    if starting_p %2 == 0:
        f_pt0+=m.get_payoff(0)
        s_pt1+=m.get_payoff(1)
    else:
        s_pt0+=m.get_payoff(0)
        f_pt1+=m.get_payoff(1)
    print (m.cards_mask)


for episode in range(10):
    starting_p = episode % 4
    m.reset(starting_p)
    # print(m.cards)
    while not m.is_done():
        next_p = m.get_next_player()
        actions, n = m.get_legal_actions()
        action = p[next_p].choose_action(actions, n)
        m.check_legal_action(action)
        m.do_action(action)
    print(m.past_rounds)
    print(m.cards_mask)

print(pt0)
print(pt1)
print("team 0 "+str(f_pt0)+"  "+str(s_pt0))
print("team 1 "+str(f_pt1)+"  "+str(s_pt1))"""


