import numpy as np


class Map:

    def __init__(self, input_file):

        map_file = open(input_file, 'r')

        # Get dimensions
        self.width, self.height = map_file.readline().split(' ')
        self.height = int(self.height)
        self.width = int(self.width)

        grid = []
        next_line = map_file.readline().strip()
        while next_line:
            grid = grid + [c for c in next_line]
            next_line = map_file.readline().strip()

        self.grid = np.array(grid)
        self.grid.shape = (self.height, self.width)

        map_file.close()

        return

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_position(self, x, y):
        try:
            return self.grid[x, y]
        except IndexError:
            print('width {}, height {}\n'.format(self.width, self.height) + "x: {}, y: {} are out of the bounds".format(x, y) + "\n")
