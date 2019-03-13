import random as rnd
import math


class Network:
    def __init__(self,
                 batch,
                 answ,
                 n_features,
                 layers,  # list of numbers of neurons on i-layer
                 learning_rate):
        self.batch = batch
        self.layers = layers
        self.n_features = n_features
        self.weights = []
        self.neurons = []
        self.answ = answ
        self.learning_rate = learning_rate

    def init_weights(self):
        weights0 = []
        for i in range(self.n_features * self.layers[0]):
            weights0.append(rnd.uniform(
                -1 / (2 * self.n_features),
                1 / (2 * self.n_features))
            )
        self.weights.append(weights0)
        for i in range(1, len(self.layers)):
            weights0 = []
            for w in range(self.layers[i - 1] * self.layers[i]):
                weights0.append(rnd.uniform(
                    -1 / (2 * self.layers[i - 1]), 1 / (2 * self.layers[i - 1])
                ))
            self.weights.append(weights0)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoidD(self, x):
        return math.exp(-x) / (self.sigmoid(x) ** 2)

    def forward(self, obj):
        neurons0 = []
        count = 1
        sum = 0
        j = 0
        self.init_weights()

        for i in range(len(self.weights[0])):
            sum += obj[j] * self.weights[0][i]
            j += 1
            if i == count * len(self.n_features) - 1:
                neurons0.append([self.sigmoid(sum), self.sigmoidD(sum)])
                sum = 0
                j = 0
                count += 1
        self.neurons.append(neurons0)

        for i in range(1, len(self.weights)):
            sum = 0
            j = 0
            count = 1
            neurons0 = []
            for k in range(len(self.weights[i])):
                sum += self.neurons[i - 1][j] * self.weights[i][k]
                j += 1
                if k == count * self.layers[i - 1] - 1:
                    neurons0.append([self.sigmoid(sum), self.sigmoidD(sum)])
                    sum = 0
                    j = 0
                    count += 1
            self.neurons.append(neurons0)

    def loss(self, pos):
        Q = 0
        for i in range(len(self.neurons[len(self.layers) - 1])):
            Q += (self.neurons[len(self.layers) - 1][i][0] - self.answ[pos][i]) ** 2
        return (1 / 2) * Q

    def lossD(self, pos, answ_index):
        return self.neurons[len(self.layers) - 1][answ_index][0] - self.answ[pos][answ_index]

    def learn(self):
        self.init_weights()
        Q = 100
        while Q != 0:
            i = rnd.randint(0, len(self.batch))
            self.forward(self.batch[i])

            L = self.loss(i)
            err = []
            for i in range(self.neurons[len(self.layers) - 1]):
                dQ = self.lossD(i)
                err.append(dQ)
                self.neurons[len(self.layers) - 1][i].append(dQ)

            for i in range(len(self.neurons) - 2, -1, -1):
                size = len(self.neurons[i])
                for j in range(size):
                    error = 0
                    for k in range(len(self.neurons[i + 1])):
                        error += self.neurons[i + 1][k][2] * \
                                 self.neurons[i + 1][k][1] * \
                                 self.weights[i + 1][k * size + j]
                    self.neurons[i][j].append(error)

            for i in range(len(self.weights)):
                count = 0
                neuron = 0
                for j in range(len(self.weights[i])):

                    if i == 0:
                        if count == self.n_features:
                            count = 0
                            neuron += 1

                        self.weights[i][j] -= self.learning_rate *\
                                              self.batch(i)[count] *\
                                              self.neurons[i][neuron][1] *\
                                              self.neurons[i][neuron][2]
                        count += 1

                    else:
                        if count == len(self.neurons[i-1]):
                            count = 0
                            neuron += 1
                        self.weights[i][j] -= self.learning_rate * \
                                              self.neurons[i-1][count][0] * \
                                              self.neurons[i][neuron][1] * \
                                              self.neurons[i][neuron][2]
                        count +=1

            l = len(self.batch)
            Q = (1 - 1/l)*Q + (1/l)*L