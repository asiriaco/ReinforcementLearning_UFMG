import random
import numpy as np
import operator as op
from map_handler import Map


class QLearning:


####global variables area
    GRASS = '.'
    TALL_GRASS = ';'
    WATER = '+'
    FIRE = 'x'
    GOAL = 'O'
    WALL = '@'

    UP = '^'
    LEFT = '<'
    RIGHT = '>'
    DOWN = 'v'

 ###for stochastic use only
    PERPENDICULAR_POSITIONS = {
        'UP_OR_DOWN': ['<', '>'],
        'LEFT_OR_RIGHT': ['^', 'v']
    }
    #weights specified like: 0.8 chance of performing selected action, 0.2 of performing perpendicular actions (0.1 each)
    STOCHASTIC_WEIGHTS = (0.8, 0.1, 0.1)

    FREE = [GRASS, TALL_GRASS, WATER]
    TERMINALS = [GOAL, FIRE]

    def __init__(self, states, actions, rewards, alpha, gamma, map_path, iterations,
                 epsilon, starting_coordinates,  mode="standard"):

        self.gamma = gamma
        self.map = Map(map_path)
        self.iterations = iterations
        self.epsilon = epsilon
        self.mode = mode
        self.starting_coordinates = starting_coordinates
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.alpha = alpha

    def create_Qmatrix(self):
        height, width = self.map.get_height(), self.map.get_width()

        return [[dict(zip([a for a in self.actions],
                          [0. for x in range(len(self.actions))]))
                 for w in range(width)] for h in range(height)]

    def set_checkpoint(self, height, width):

        x, y = np.random.randint(
            0, height - 1), np.random.randint(0, width - 1)
        while self.map.get_position(x, y) not in self.FREE:
            x, y = np.random.randint(
                0, height - 1), np.random.randint(0, width - 1)

        return x, y

    def get_best_action(self, Q, state):

        x, y = state
        best_action = max(sorted(Q[x][y].items()), key=op.itemgetter(1))[0]
        return best_action

    def select_action(self, Q, state):

        random_action = np.random.choice([action for action in self.actions])

        if self.epsilon is not None:
            best_action = self.get_best_action(Q, state)
            next_action = np.random.choice([random_action, best_action],
                                           p=[self.epsilon,
                                           (1 - self.epsilon)])
        else:
            next_action = random_action

        #dealing with stochastic mode of execution
        if self.mode == "stochastic":
            if next_action == self.UP or next_action == self.DOWN:
                possibilities = list(self.PERPENDICULAR_POSITIONS['UP_OR_DOWN'])
                possibilities.insert(0, next_action)
                next_action = random.choices(possibilities,
                                             weights=self.STOCHASTIC_WEIGHTS,
                                             k=1)
                next_action = str(next_action[0])

            elif next_action == self.LEFT or next_action == self.RIGHT:
                    possibilities = list(self.PERPENDICULAR_POSITIONS['LEFT_OR_RIGHT'])
                    possibilities.insert(0, next_action)
                    next_action = random.choices(possibilities,
                                                 weights=self.STOCHASTIC_WEIGHTS,
                                                 k=1)
                    next_action = str(next_action[0])

        new_state = self.simulate_action(state, next_action)
        return next_action, new_state

    def simulate_action(self, state, action):

        height, width = self.map.get_height(), self.map.get_width()
        x, y = state
        x_bound, y_bound = height - 1, width - 1

        #defining grid new position based on the current action
        if action == self.UP:
            new_state = (x - 1, y)
        elif action == self.LEFT:
            new_state = (x, y - 1)
        elif action == self.RIGHT:
            new_state = (x, y + 1)
        elif action == self.DOWN:
            new_state = (x + 1, y)
        else:
            new_state = (-1, -1)

        new_x, new_y = new_state
        if new_x < 0 or new_x > x_bound or new_y < 0 or new_y > y_bound:
            return state
        new_position = self.map.get_position(new_x, new_y)
        if new_position == self.WALL:
            return state
        elif new_x < 0 or new_x > x_bound or new_y < 0 or new_y > y_bound:
            return state
        else:
            return new_state

    def get_maximum_q(self, Q, state):
        x, y = state

        position = self.map.get_position(x, y)
        if position in self.TERMINALS:
            return self.rewards[position]

        return max(Q[x][y].items(), key=op.itemgetter(1))[1]

    def updateQmatrix(self, Q, action, state, new_maximum_q):

        x, y = state
        current_q = Q[x][y][action]
        state_r = self.rewards[self.map.get_position(*state)]
        Q[x][y][action] = current_q + self.alpha * \
            (state_r + self.gamma * new_maximum_q - current_q)

    def get_qsum(self, Q):

        height, width = self.map.get_height(), self.map.get_width()

        qsum = 0.
        for line in range(height):
            for row in range(width):
                qsum += self.get_maximum_q(Q, (line, row))

        return qsum

    def fit(self):

        height, width = self.map.get_height(), self.map.get_width()
        Q = self.create_Qmatrix()
        episode = 0
        iteration = 0
        while True:
            #selecting initial state passed as parameter
            if iteration == 0:
                state = self.starting_coordinates
            #initializing a random state at the begin of an episode (which is not the first)
            else:
                state = self.set_checkpoint(height, width)

            #Loop through the map until finding a terminal state
            while self.map.get_position(*state) not in self.TERMINALS:
                action, new_state = self.select_action(Q, state)
                new_maximum_q = self.get_maximum_q(Q, new_state)
                self.updateQmatrix(Q, action, state, new_maximum_q)
                state = new_state
                iteration += 1
                if iteration == self.iterations:
                    break
            episode += 1
            if iteration == self.iterations:
                break
        self.show_results(Q)

    def show_results(self, Q):

        height, width = self.map.get_height(), self.map.get_width()

        for x in range(height):
            row = []
            for y in range(width):
                position = self.map.get_position(x, y)
                if position in self.FREE:
                    position = self.get_best_action(Q, (x, y))
                row.append(position)
            print(''.join(row))



