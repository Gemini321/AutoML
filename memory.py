import numpy as np

class RelayMemory(object):
    def __init__(self, layer_limit):
        self.__capacity = 500
        self.__pos = 0
        self.__size = 0

        self.__layers = np.zeros((self.__capacity, layer_limit), dtype=np.int)
        self.__reward = np.zeros((self.__capacity, 1), dtype=np.float)

    def push(self, layer, reward):
        '''Push one record into relay memory. Records are stored in a cyclic queue'''
        self.__layers[self.__pos, :] = layer
        self.__reward[self.__pos, :] = reward

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = self.__size + 1 if self.__size < 500 else self.__size

    def sample(self, batch_size):
        '''Randomly sample a batch of records'''
        indices = np.random.randint(0, self.__size, size=(batch_size,))
        layers = self.__layers[indices, :]
        rewards = self.__reward[indices, :]
        return layers, rewards

    def size(self):
        return self.__size

    def capacity(self):
        return self.__capacity
