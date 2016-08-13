# Carl McCann   12508463
import numpy as np
from MLP import MLP

# for testing,
np.random.seed(1)


# XOR training data
# desired output:
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
desired_xor_output = np.array([[0], [1], [1], [0]])

max_epochs = 100000
learning_rate = 1
# set up mlp and set initial weights
inputs = 2
hidden = 2
outputs = 1
mlp = MLP(inputs, hidden, outputs)
mlp.randomise()

print('\nPre-Training Testing:\n')
for i in range(len(xor_inputs)):
    mlp.forward(xor_inputs[i])
    print('Target:\t' + str(desired_xor_output[i]))
    print('Output:\t' + str(mlp.o))

print()

with open('xor_output/xor_output_mlp_size_(' + str(inputs) + ', ' + str(hidden) + ', ' + str(outputs) + ')_LR_' +
                  str(learning_rate) + '_epochs_' + str(max_epochs) + '.txt', 'w') as f:

    f.write('MLP Size\t\t\t(' + str(inputs) + ', ' + str(hidden) + ', ' + str(outputs) + ')\n')
    f.write('Epochs:\t\t\t\t' + str(max_epochs) + '\n')
    f.write('Learning Rate:\t\t' + str(learning_rate) + '\n\n')

    print('Training:\n')
    f.write('Training:\n')
    # loop similar to the suggestion in the specification
    for i in range(0, max_epochs):
        error = 0
        mlp.forward(xor_inputs)
        error = mlp.backwards(xor_inputs, desired_xor_output)
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
    f.write('\nTesting:\n')
    for i in range(len(xor_inputs)):
        mlp.forward(xor_inputs[i])
        print('Target:\t' + str(desired_xor_output[i]))
        print('Output:\t' + str(mlp.o))
        f.write('\nTarget:\t' + str(desired_xor_output[i]) + '\n')
        f.write('Output:\t' + str(mlp.o) + '\n')
