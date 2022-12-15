import math
from mdp import MDP

STATES = ['.', ';', '+', "x", "O", "@"]
ACTIONS = ['^', '<', '>', 'v']
REWARDS = {'.': -0.1, ';': -0.3, '+': -1.0, 'x': -10.0, 'O': 10.0, '@': -math.inf}
POSITIVE_REWARDS = {'.': 3.0, ';': 1.5, '+': 1.0, 'x': 0.0, 'O': 10.0, '@': -math.inf}


if __name__ == "__main__":

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    iter = 300000
    map_path = "maps/choices.map"

    mdp = MDP(STATES, ACTIONS, REWARDS, alpha, gamma,
                   map_path, iter, epsilon, qsumf=None, mode="standard")
    mdp.qlearning()


