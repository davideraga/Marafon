
import numpy as np

class RandomAgent:
    """Agent that does random actions"""
    def __init__(self, seed):
        self.rnd = np.random.RandomState(seed)

    def choose_action(self, action_mask, n_actions):
        """chooses a random action"""
        if n_actions == 1:
            r = 1
        else:
            r = self.rnd.randint(1, n_actions+1)
        c = 0
        for i in range(len(action_mask)):
            c += action_mask[i]
            if c == r:
                return i
        return 0


