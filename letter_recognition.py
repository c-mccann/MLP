import numpy as np
from MLP import MLP

np.random.seed(1)

# should be np arrays

training_data = []
desired_output = []
desired_output_numbers = []
with open('letter_recognition_data.csv', 'r') as f:
    for line in f:
        training_data.append(line.rstrip('\n').split(','))
        desired_output.append(line.rstrip('\n').split(',')[0])

for i in range(len(desired_output)):
    desired_output_numbers.append(ord(str(desired_output[i]).lower()) - 96)



# set up mlp and set initial weights
inputs = 17
hidden = 10
outputs = 26

learning_rate = 0.05
max_epochs = 1000

mlp = MLP(inputs, hidden, outputs)
mlp.randomise()

