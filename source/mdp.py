import random

import numpy as np
import operator as op

from map import Map


class MDP():

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

    PERPENDICULAR_POSITIONS = {
        'UP_OR_DOWN': ['<', '>'],
        'LEFT_OR_RIGHT': ['^', 'v']
    }

    STOCHASTIC_WEIGHTS = (80, 10, 10)

    FREE = [GRASS, TALL_GRASS, WATER]
    TERMINALS = [GOAL, FIRE]

    def __init__(self, S, A, R, alpha, gamma, map_filename, iterations,
                 epsilon, qsumf, mode="standard"):
        '''
            Initialize the Markov Decision Problem instance.

            @S: (string list) Set of states where the agent can be.
            @A: (string list) Set of actions the agent can execute.
            @R: (dict {string: int}) Reward function, mapping states to
                rewards.
            @alpha: (float) Learning rate.
            @gamma: (float) Discount factor for Q-Learning algorithm.
            @map_filename: (string) Name of the file to read mapa data from.
            @iterations: (int) Number of iterations to run the Q-Learning
                algorithm for.
            @epsilon: (float) Epsilon for e-greedy policy.
            @qsumf: (str) Name of file to print QSum data to.
        '''

        self.states = S
        self.actions = A
        self.rewards = R
        self.alpha = alpha
        self.gamma = gamma
        self.map = Map(map_filename)
        self.iterations = iterations
        self.epsilon = epsilon
        self.qsumf = qsumf
        self.mode = mode

    def init_qmatrix(self):
        '''
            Initialize the Q matrix for Q-Learning algorithm.

            @return: ((dict {string: int} list) list) Initial Q matrix for
                current MDP.
        '''

        height, width = self.map.get_height(), self.map.get_width()

        return [[dict(zip([a for a in self.actions],
                          [0. for x in range(len(self.actions))]))
                 for w in range(width)] for h in range(height)]

    def select_initial_state(self, height, width):
        '''
            Select initial state for Q-Learning algorithm randomly.

            @height: (int) Map height.
            @width: (int) Map width:

            @return: (int, int) Initial state coordinates.
        '''

        x, y = np.random.randint(
            0, height - 1), np.random.randint(0, width - 1)
        while self.map.get_position(x, y) not in self.FREE:
            x, y = np.random.randint(
                0, height - 1), np.random.randint(0, width - 1)

        return x, y

    def get_best_action(self, Q, state):
        '''
            Get the best action for current state.

            @Q: ((dict {string: int} list) list) Q matrix.
            @state: (int, int) Current state.

            @return: (string) Optimal action  for current state.
        '''

        x, y = state
        best_action = max(sorted(Q[x][y].items()), key=op.itemgetter(1))[0]
        if self.mode == "stochastic":
            possibilities = []
            if best_action == self.UP or best_action == self.DOWN:
                possibilities = list(self.PERPENDICULAR_POSITIONS['UP_OR_DOWN'])
                possibilities.insert(0, best_action)
                best_action = random.choices(possibilities, weights=self.STOCHASTIC_WEIGHTS, k=1)
                best_action = str(best_action[0])
            elif best_action == self.LEFT or best_action == self.RIGHT:
                possibilities = list(self.PERPENDICULAR_POSITIONS['LEFT_OR_RIGHT'])
                possibilities.insert(0, best_action)
                best_action = random.choices(possibilities, weights=self.STOCHASTIC_WEIGHTS, k=1)
                best_action = str(best_action[0])
        return best_action

    def select_action(self, Q, state):
        '''
            Select a random action to perform.

            @Q: ((dict {string: int} list) list) Q matrix.
            @state: (int, int) Current state.

            @return: (string (int, int)) Action to perform and the state the
                agent will get to, by performing it.
        '''

        random_action = np.random.choice([action for action in self.actions])

        if self.epsilon is not None:
            best_action = self.get_best_action(Q, state)
            next_action = np.random.choice([random_action, best_action],
                                           p=[self.epsilon,
                                           (1 - self.epsilon)])
        else:
            next_action = random_action

        new_state = self.simulate_action(state, next_action)
        return next_action, new_state

    def simulate_action(self, state, action):
        '''
            Simulate performing an action at a particular state.

            @state: (int, int) Current state.
            @action: (string) Action to simulate.

            @return: (int, int) State the agent would get to, by performing the
                given action.
        '''

        height, width = self.map.get_height(), self.map.get_width()

        x, y = state
        x_bound, y_bound = height - 1, width - 1

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

    def get_maxq(self, Q, state):
        '''
            Get the maximum value in the Q matrix entry of a particular state.

            @Q: ((dict {string: int} list) list) Q matrix.
            @state: (int, int) Current state.

            @return: (float) Maximum Q matrix value for the given state.
        '''

        x, y = state

        position = self.map.get_position(x, y)
        if position in self.TERMINALS:
            return self.rewards[position]

        return max(Q[x][y].items(), key=op.itemgetter(1))[1]

    def updateQ(self, Q, action, state, new_state_maxq):
        '''
            Update the Q matrix entry for the current state.

            @Q: ((dict {string: int} list) list) Q matrix.
            @action: (string) Action the agent is about to perform.
            @state: (int, int) Current state.
            @new_state_maxq: (float) Maximum Q matrix value for the given
                state.
        '''

        x, y = state
        cur_q = Q[x][y][action]
        state_r = self.rewards[self.map.get_position(*state)]
        Q[x][y][action] = cur_q + self.alpha * \
            (state_r + self.gamma * new_state_maxq - cur_q)

    def get_qsum(self, Q):
        '''
            Get the sum of the max values for each entry in the Q matrix. This
            is defined as a metric to assess convergende through episodes.

            @Q: ((dict {string: int} list) list) Q matrix.

            @return: (float) Sum of the max values for each entry in the Q
            matrix.
        '''

        height, width = self.map.get_height(), self.map.get_width()

        qsum = 0.
        for line in range(height):
            for row in range(width):
                qsum += self.get_maxq(Q, (line, row))

        return qsum

    def qlearning(self):
        '''
            Run the Q-Learning algorithm for the MDP instance.
        '''

        if self.qsumf is not None:
            qsum_file = open(self.qsumf, 'w')

        height, width = self.map.get_height(), self.map.get_width()

        Q = self.init_qmatrix()
        episode = 0

        iteration = 0
        while True:
            if iteration == 0:
                state = (0, 5)
            else:
                state = self.select_initial_state(height, width)


            # While terminal state hasn't been reached.
            while self.map.get_position(*state) not in self.TERMINALS:
                action, new_state = self.select_action(Q, state)
                new_state_maxq = self.get_maxq(Q, new_state)
                self.updateQ(Q, action, state, new_state_maxq)
                state = new_state

                iteration += 1

                if self.qsumf is not None:
                    qsum_file.write('{}\t{}\n'.format(
                        iteration, self.get_qsum(Q)))

                if iteration == self.iterations:
                    break

            episode += 1

            if iteration == self.iterations:
                break

        if self.qsumf is not None:
            qsum_file.close()

        self.print_results(Q)

    def print_results(self, Q):
        '''
            Print results.

            @Q: ((dict {string: int} list) list) Q matrix.
        '''

        q_file = open('q.txt', 'w')
        pi_file = open('pi.txt', 'w')

        height, width = self.map.get_height(), self.map.get_width()

        for x in range(height):
            for y in range(width):
                # Q File
                q_file.write(('{},{},direita,{}' +
                             '\n').format(x, y, Q[x][y][self.RIGHT]))
                q_file.write(('{},{},esquerda,{}' +
                             '\n').format(x, y, Q[x][y][self.LEFT]))
                q_file.write(('{},{},acima,{}' +
                             '\n').format(x, y, Q[x][y][self.UP]))
                q_file.write(('{},{},abaixo,{}' +
                             '\n').format(x, y, Q[x][y][self.DOWN]))

                # Pi file
                position = self.map.get_position(x, y)

                if position in self.FREE:
                    position = self.get_best_action(Q, (x, y))

                pi_file.write(position)

            pi_file.write('\n')

        q_file.close()
        pi_file.close()