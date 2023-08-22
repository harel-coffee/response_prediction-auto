import numpy as np
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU
import AT_Classes as Classes
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard
import networkx as nx
from node2vec import Node2Vec
from scipy import sparse
from time import time


UseUpper = True

PltNAme = 'crossEntropy5_1Layer_UseUpper='+str(UseUpper)+'_encdim=128'
maxNumPoints = 12
# inputSize  = int((maxNumPoints * (maxNumPoints - 1))/2)
# inputSize = int(maxNumPoints * maxNumPoints)

train = []
G = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\scenarios.mat')
MergerAM = G['MergerAM']
numBranches = MergerAM.shape
counter = 0
# numBranches[1]
for i in range(numBranches[1]):
    scenarios = MergerAM[0,i]
    # print(scenarios.shape)
    if scenarios.any():
        # scenarios.shape[2]
        for j in range(scenarios.shape[2]):
            scenario = scenarios[:,:,j]
            S = Classes.cl_scenario(maxNumPoints, scenarios.shape[0],scenario)
            if UseUpper:
                scenario_arr = S.getUpperArr()
            else:
                scenario_arr = S.getWholeArr()

            train.append(scenario_arr)

# Data splicing
train = np.asarray(train, dtype=np.float)
print(train.shape)
x_train, x_test = train_test_split(train, test_size=0.33, random_state=42)
print(x_train.shape)
print(x_test.shape)


## ------------------------------------    Deep Autoencoder

# this is the size of our encoded representations
encoding_dim = 128 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_data = Input(shape=(x_train.shape[1],))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_data)
# encoded = Dense(64, activation='relu')(encoded) #added
# encoded = Dense(32, activation='relu')(encoded) #added
# "decoded" is the lossy reconstruction of the input

# decoded = Dense(64, activation='relu')(encoded) #added
# decoded = Dense(128, activation='relu')(decoded) #added
decoded = Dense(x_train.shape[1], activation='softmax')(encoded)


# this model maps an input to its reconstruction
autoencoder = Model(input_data, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_data, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Multiclass SVM Loss: To be modified and included in AE (check other loss functions as well)
def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1) # idea: can change 1 to a functional one
    margins[y] = 0 # to skipp iterating, make the current class loss inactive in sum
    loss_i = np.sum(margins)
    return loss_i


tensorboard = TensorBoard(log_dir="logs/"+PltNAme+"{}".format(time()))

history = autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard])

# encode and decode some digits
# note that we take them from the *test* set
encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

n = 20  # how many digits we will display
inpSiz = int(np.round(np.sqrt(x_train.shape[1]))*np.round(np.sqrt(x_train.shape[1])))

plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i,:inpSiz].reshape(int(np.sqrt(inpSiz)), int(np.sqrt(inpSiz))))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_data[i,:inpSiz].reshape(int(np.sqrt(inpSiz)), int(np.sqrt(inpSiz))))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # ## ------------------------------------    Convolutional autoencoder
#
# x_train = x_train.astype('float32') #/ 255.
# x_test = x_test.astype('float32') #/ 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
#
# input_data = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
#
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_data)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)
#
# encoder = Model(input_data, encoded)
#
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
# autoencoder = Model(input_data, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#
# autoencoder.fit(x_train, x_train,
#                 epochs=1,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[tensorboard])
#
# decoded_data = autoencoder.predict(x_test)
#
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1,n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_data[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
#
#
# encoded_data = encoder.predict(x_test)
# # decoded_data = decoder.predict(encoded_data)
# n = 10
# plt.figure(figsize=(20, 8))
# for i in range(1,n):
#     ax = plt.subplot(1, n, i)
#     plt.imshow(encoded_data[i].reshape(4, 4 * 8).T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

# # ## ------------------------------------    Deep autoencoder
# input_data = Input(shape=(784,))
# encoded = Dense(128, activation='relu')(input_data)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(32, activation='relu')(encoded)
#
# decoded = Dense(64, activation='relu')(encoded)
# decoded = Dense(128, activation='relu')(decoded)
# decoded = Dense(784, activation='softmax')(decoded)
#
#
# autoencoder = Model(input_data, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
# autoencoder.fit(x_train, x_train,
#                 epochs=3,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))




############################################ Node2Vec
# if Use_Node2vec:
#     scenario1 = sparse.csr_matrix(scenario)
#     scenario_G = nx.from_scipy_sparse_matrix(scenario1, parallel_edges=True,
#                                              create_using=nx.hoffman_singleton_graph())  # create_using=nx.MultiGraph())
#     # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
#     scenario_node2vec = Node2Vec(scenario_G, dimensions=64, walk_length=30, num_walks=200,
#                                  workers=4)  # Use temp_folder for big graphs
#     scenario_arr = np.asarray(scenario_node2vec.walks, dtype=np.int8)
#     scenario_arr = scenario_arr.flatten()