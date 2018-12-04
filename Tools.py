

import random

#used for generating number randomly
class RandomOperator:

    def __init__(self):
        return

    def getUniformInteger(self, lower, upper):
        return random.randint(lower, upper)

    def getUniformDouble(self, lower, upper):
        return random.uniform(lower, upper)
