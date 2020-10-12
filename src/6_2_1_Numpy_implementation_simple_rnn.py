#
# * Listing 6.21 P.197

#%%
import numpy as np 

# * Number of timesteps in the input sequence
timesteps = 100

# * Dimension of the input feature space
input_features = 32

# * Dimension of output feature space
output_features = 64

# * Input data: random noise for example (100 x 32)
inputs = np.random.random((timesteps, input_features))

# * Inital state: an all-zero vector (64, )
state_t = np.zeros((output_features,))

# * Create random weight matrics
W = np.random.random((output_features, input_features)) # * (64, 32)
U = np.random.random((output_features, output_features)) # * (64, 64)
b = np.random.random((output_features, ))               # * (64, )

# print('W shape', W.shape)
# print('U shape', U.shape)
# print('b shape', b.shape)

successive_outputs = []

# input_t = inputs[0]
# print('input_t shape', input_t.shape)
# print('dot(W, input_t) shape', np.dot(W, input_t).shape)
# print('dot(U, state_t) shape', np.dot(U, state_t).shape)

# * input_t is a vector of shap(input_features)
for input_t in inputs:
    # * Combines inputs with current state (previous output) to obtain current output
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

    # * Stores this output in a list
    successive_outputs.append(output_t)

    # * update state of network for next timestep
    state_t = output_t

# print('successive_outputs')
# print(type(successive_outputs))
# print(len(successive_outputs))
# print(len(successive_outputs[0]))


# # * final output is a 2D tensor of shape(timesteps, output_features)
final_output_sequence = np.concatenate(successive_outputs, axis=0)

print('final_output_sequence')
print(final_output_sequence.shape)
# print(final_output_sequence)
# %%
