# Carl McCann   12508463
import numpy as np
from MLP import MLP

# for testing,
np.random.seed(1)

# Generate the vectors and the desired output
fifty_vectors = []
desired_sin_output = []
for i in range(0, 50):
    vector = list(np.random.uniform(-1.0, 1.0, 4))
    vector = [float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3])]
    fifty_vectors.append(vector)

fifty_vectors = np.array(fifty_vectors)

for i in range(0, 50):
    summed_vector_comps = np.sum(fifty_vectors)
    desired_sin_output.append([np.sin(np.sum(fifty_vectors[i]))])

desired_sin_output = np.array(desired_sin_output)

max_epochs = 1000000
learning_rate = 0.005
# set up mlp and set initial weights
inputs = 4
hidden = 5
outputs = 1
mlp = MLP(inputs, hidden, outputs)
mlp.randomise()

print('\nPre-Training Testing:\n')
for i in range(len(fifty_vectors) - 10, len(fifty_vectors)):
    mlp.forward(fifty_vectors[i],sin=True)
    print('Target:\t' + str(desired_sin_output[i]))
    print('Output:\t' + str(mlp.o) + '\n')

print()

with open('sin_output/sin_output_mlp_size_(' + str(inputs) + ', ' + str(hidden) + ', ' + str(outputs) + ')_LR_' +
                  str(learning_rate) + '_epochs_' + str(max_epochs) + '.txt', 'w') as f:

    f.write('MLP Size\t\t\t(' + str(inputs) + ', ' + str(hidden) + ', ' + str(outputs) + ')\n')
    f.write('Epochs:\t\t\t\t' + str(max_epochs) + '\n')
    f.write('Learning Rate:\t\t' + str(learning_rate) + '\n\n')

    print('Training:\n')
    f.write('Training:\n')
    # loop similar to the suggestion in the specification
    for i in range(0, max_epochs):
        error = 0
        mlp.forward(fifty_vectors[:len(fifty_vectors) - 10], True)

        error = mlp.backwards(fifty_vectors[:(len(fifty_vectors) - 10)], desired_sin_output[:len(fifty_vectors) - 10],
                              True)
        mlp.update_weights(learning_rate)
        # gives error after every 10% progress
        if (i + 1) % (max_epochs / 10) == 0:
            length = len(str(i))
            # formatting for file writing
            whitespace = ''
            while length < 10:
                whitespace += ' '
                length += 1
            print('Epoch:\t' + str(i + 1) + whitespace + '\tError:\t' + str(error))
            f.write('Epoch:\t' + str(i + 1) + whitespace + '\tError:\t' + str(error) + '\n')

    print('\nTesting:\n')
    for i in range(len(fifty_vectors) - 10, len(fifty_vectors)):
        mlp.forward(fifty_vectors[i],sin=True)
        print('Target:\t' + str(desired_sin_output[i]))
        print('Output:\t' + str(mlp.o) + '\n')
        f.write('\nTarget:\t' + str(desired_sin_output[i]) + '\n')
        f.write('Output:\t' + str(mlp.o) + '\n')

