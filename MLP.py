# Carl McCann   12508463
import numpy as np


class MLP:
    def __init__(self, ni, nh, no):
        self.no_of_inputs = ni
        self.no_of_hidden_units = nh
        self.no_of_outputs = no
        self.w1 = np.array
        self.w2 = np.array
        self.dw1 = np.array
        self.dw2 = np.array
        self.z1 = np.array
        self.z2 = np.array
        self.h = np.array
        self.o = np.array

    # sigmoid for range 0..1
    def activation_sigmoid(self, sig_input, der=False):
        if der:
            return np.exp(-sig_input) / (1 + np.exp(-sig_input)) ** 2
        else:
            return 1 / (1 + np.exp(-sig_input))

    # tanh for range -1..1
    def activation_tanh(self, tanh_input, der=False):
        if der:
            return 1 - (np.power(self.activation_tanh(tanh_input), 2))
        else:
            return (2 / (1 + np.exp(tanh_input * -2))) - 1

    # used for testing to see the state of the mlp
    def info(self):
        print('Number of Inputs:      \n' + str(self.no_of_inputs))
        print('Number of Hidden Units:\n' + str(self.no_of_hidden_units))
        print('Number of Outputs:     \n' + str(self.no_of_outputs))
        print('w1:                    \n' + str(self.w1))
        print('w2:                    \n' + str(self.w2))
        print('dw1:                   \n' + str(self.dw1))
        print('dw2:                   \n' + str(self.dw2))
        print('z1:                    \n' + str(self.z1))
        print('z2:                    \n' + str(self.z2))
        print('h:                     \n' + str(self.h))
        print('o:                     \n' + str(self.o) + '\n')


    # initialises weights to small random values, and fills deltas with
    # 0s to the same size matrix as the corresponding weight matrix
    def randomise(self):
        self.w1 = np.array((np.random.uniform(0.0, 1, (self.no_of_inputs,self.no_of_hidden_units))).tolist())
        self.dw1 = np.dot(self.w1, 0)
        self.w2 = np.array((np.random.uniform(0.0, 1, (self.no_of_hidden_units, self.no_of_outputs))).tolist())
        self.dw2 = np.dot(self.w2, 0)

    # forward pass, computes activations in z1 and z2, then fills h and o with the activated
    # values either sigmoid or tanh depending on whether it is the xor or sine problem
    def forward(self, input_vectors, sin=False):
        self.z1 = np.dot(input_vectors, self.w1)
        if sin:
            self.h = self.activation_tanh(self.z1)
        else:
            self.h = self.activation_sigmoid(self.z1)

        self.z2 = np.dot(self.h, self.w2)
        if sin:
            self.o = self.activation_tanh(self.z2)
        else:
            self.o = self.activation_sigmoid(self.z2)


    # backpropagation, computes error, computes activation derivates based on whether
    # xor or sine problem, multiplies the derivatives by the error, then computes
    # the dot product of these values and the values in h and o to get the deltas
    def backwards(self, input_vectors, target, sin=False):
        output_error = np.subtract(target, self.o)
        if sin:
            activation_out_2 = self.activation_tanh(self.z2, True)
            activation_out_1 = self.activation_tanh(self.z1, True)
        else:
            activation_out_2 = self.activation_sigmoid(self.z2, True)
            activation_out_1 = self.activation_sigmoid(self.z1, True)

        dw2_a = np.multiply(output_error, activation_out_2)
        self.dw2 = np.dot(self.h.T, dw2_a)
        dw1_a = np.multiply(np.dot(dw2_a, self.w2.T), activation_out_1)
        self.dw1 = np.dot(input_vectors.T, dw1_a)
        return np.mean(np.abs(output_error))

    # changes weights with regard to learning rate after the deltas have been computed in backwards
    def update_weights(self, learning_rate):
        self.w1 = np.add(self.w1, learning_rate * self.dw1)
        self.w2 = np.add(self.w2, learning_rate * self.dw2)
        self.dw1 = np.array
        self.dw2 = np.array
