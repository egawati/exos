import math as math
import numpy as np


import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


class OnlineStatistic:
    def __init__(self, npoints, linear_sum, squared_sum):
        self.npoints = npoints
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum

    def update_stat(self, new_point):
        self.npoints =
