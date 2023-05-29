# Marafon
Playing Marrafone a.k.a Beccacino with Reinforcement Learning
Project for Autonomous and Adaptive Systems
Implements a simple enviroment to simulate the game, and a few agents.

train.py is the scrip to train the agents, the parameters are:
-a : agent type ["ppo", "dqn"]
-sp : self play "yes" or versus rnd "no"
-n : number of episodes
Hyperparameters relative to the algorithm can be edited in the code

test_games.py is the script to test the agents, it makes the agents play some games and it outputs some statistics, the parameters are:
-a : agent type ["ppo", "dqn"]
-vs : agent type of the second team ["ppo", "dqn", rnd]
-n : number of games (a game is composed by more episodes who scores > 41 and > than the other wins)

play.py is the script to play with the agents, 1 agent in your team and vs 2,  there is a simple terminal interface, the parameters are:
-a : agent type ["ppo", "dqn"]
-n : number of games (a game is composed by more episodes who scores > 41 and > than the other wins)


Requirements:
python==3.9
tensorflow==2.10.0
numpy==1.23.5
