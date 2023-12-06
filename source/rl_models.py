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

    MOVEMENTS = {
        "^": lambda x, y: (x - 1, y),
        "v": lambda x, y: (x + 1, y),
        "<": lambda x, y: (x, y - 1),
        ">": lambda x, y: (x, y + 1)
    }

    #for stochastic use only
    PERPENDICULAR_POSITIONS = {
        'UP_OR_DOWN': ['<', '>'],
        'LEFT_OR_RIGHT': ['^', 'v']
    }
    #weights specified like: 0.8 chance of performing selected action, 0.2 of performing perpendicular actions (0.1 each)
    STOCHASTIC_WEIGHTS = (0.8, 0.1, 0.1)

    FREE_TO_GO_SPACES = [GRASS, TALL_GRASS, WATER]
    TERMINAL_STATES = [GOAL, FIRE]

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
        self.Q = self.map.create_map_Qmatrix(actions)

    def set_checkpoint(self, height, width):

        x, y = np.random.randint(0, height - 1), np.random.randint(0, width - 1)
        while self.map.get_position(x, y) not in self.FREE_TO_GO_SPACES:
            x, y = np.random.randint(0, height - 1), np.random.randint(0, width - 1)

        return x, y

    def get_best_action(self,  state):

        x, y = state
        best_action = max(sorted(self.Q[x][y].items()), key=op.itemgetter(1))[0]
        return best_action

    def choose_action(self, state):

        random_action = np.random.choice([action for action in self.actions])

        if self.epsilon is not None:
            best_action = self.get_best_action(state)
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

        new_state = self.test_action(state, next_action)
        return next_action, new_state

    def test_action(self, state, action):

        height, width = self.map.get_height(), self.map.get_width()
        x, y = state
        x_edge, y_edge = height - 1, width - 1

        #defining grid new position based on the current action
        if action not in self.actions:
            new_state = (-1, -1)
        else:
            new_state = self.MOVEMENTS[action](x, y)

        new_x, new_y = new_state
        if new_x < 0 or new_x > x_edge or new_y < 0 or new_y > y_edge:
            return state

        new_position = self.map.get_position(new_x, new_y)

        if new_position == self.WALL:
            return state
        elif new_x < 0 or new_x > x_edge or new_y < 0 or new_y > y_edge:
            return state
        else:
            return new_state

    def get_maximum_q(self,  state):
        x, y = state

        position = self.map.get_position(x, y)
        if position in self.TERMINAL_STATES:
            return self.rewards[position]

        return max(self.Q[x][y].items(), key=op.itemgetter(1))[1]

    def updateQmatrix(self,  action, state, new_maximum_q):

        x, y = state
        current_q = self.Q[x][y][action]
        state_r = self.rewards[self.map.get_position(*state)]
        self.Q[x][y][action] = current_q + self.alpha * \
            (state_r + self.gamma * new_maximum_q - current_q)

    def get_qsum(self):

        height, width = self.map.get_height(), self.map.get_width()

        qsum = 0.
        for line in range(height):
            for row in range(width):
                qsum += self.get_maximum_q((line, row))

        return qsum

    def fit(self):

        height, width = self.map.get_height(), self.map.get_width()
        episode = 0
        iteration = 0
        while True:
            #selecting initial state passed as parameter
            if iteration == 0:
                state = self.starting_coordinates
            else:
                state = self.set_checkpoint(height, width)

            #Loop through the map until finding a terminal state
            while self.map.get_position(*state) not in self.TERMINAL_STATES:
                action, new_state = self.choose_action(state)
                new_maximum_q = self.get_maximum_q(new_state)
                self.updateQmatrix(action, state, new_maximum_q)
                state = new_state
                iteration += 1
                if iteration == self.iterations:
                    break
            episode += 1
            if iteration == self.iterations:
                break
        self.show_policy()

    def show_policy(self):

        height, width = self.map.get_height(), self.map.get_width()

        for x in range(height):
            row = []
            for y in range(width):
                position = self.map.get_position(x, y)
                if position in self.FREE_TO_GO_SPACES:
                    position = self.get_best_action((x, y))
                row.append(position)
            print(''.join(row))



