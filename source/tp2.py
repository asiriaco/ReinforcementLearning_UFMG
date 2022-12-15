import math
from mdp import QLearning
import json

STATES = ['.', ';', '+', "x", "O", "@"]
ACTIONS = ['^', '<', '>', 'v']
REWARDS = {'.': -0.1, ';': -0.3, '+': -1.0, 'x': -10.0, 'O': 10.0, '@': -math.inf}
POSITIVE_REWARDS = {'.': 3.0, ';': 1.5, '+': 1.0, 'x': 0.0, 'O': 10.0, '@': -math.inf}


if __name__ == "__main__":

    #loading config file
    with open("appconfig.json", encoding='utf-8') as config_file:
        config = json.load(config_file)

    #loading hyperparameters
    alpha = config['HyperParameters']['alpha']
    gamma = config['HyperParameters']['gamma']
    epsilon = config['HyperParameters']['epsilon']

    #loading run mode settings
    iterations = config['RunMode']['iterations']
    starting_coordinates = (config['RunMode']['initial_y'], config['RunMode']['initial_x'])
    run_mode = config['RunMode']['run_mode']
    run_mode = "standard" if run_mode not in ["positive_rewards", "stochastic"] else run_mode

    #loading paths
    map_path = "maps/" + config['FileNames']['map_name']

    #setting rewards dictionary according to the run mode
    rewards = POSITIVE_REWARDS if run_mode == "positive_rewards" else REWARDS

    #creating model based on parameters above
    model = QLearning(STATES, ACTIONS, rewards,
                      alpha, gamma, map_path, iterations,
                      epsilon, starting_coordinates, mode=run_mode)

    #training model
    model.fit()



