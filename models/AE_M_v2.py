# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:46:57 2022

@author: SeyedM.MousaviKahaki
"""
######################################## Working
# from google.colab import files
# files.upload()

# !cp kaggle.json ~/.kaggle/
# !kaggle datasets download -d jessicali9530/celeba-dataset
### Adopted from: https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8

import os
from glob import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
# from keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from keras.models import load_model
from skimage.metrics import structural_similarity as compare_ssim
import statistics
import cv2


WEIGHTS_FOLDER = 'C:/DATA/Code/weights/'
DATA_FOLDER = 'C:/DATA/extracted_cutted_Augmented/data/AE_Data/'
LOG_DIR = 'C:/DATA/Code/weights/logDir/'

# WEIGHTS_FOLDER = '/home/seyedm.mousavikahaki/Documents/wxc4/FinalPatches/AE_Data/weights/'
# DATA_FOLDER = '/home/seyedm.mousavikahaki/Documents/wxc4/FinalPatches/AE_Data/'
# LOG_DIR = '/home/seyedm.mousavikahaki/Documents/wxc4/FinalPatches/AE_Data/weights/logdir'

# tensorboard --logdir /home/seyedm.mousavikahaki/Documents/wxc4/FinalPatches/AE_Data/weights/logdir

# tensorboard --logdir C:\DATA\Code\weights\logDir

filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.png')))
NUM_IMAGES = len(filenames)
print("Total number of images : " + str(NUM_IMAGES))
# prints : Total number of images : 202599


INPUT_DIM = (256,256,3) # Image dimension
BATCH_SIZE = 128
Z_DIM = 200 # Dimension of the latent vector (z)

data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(DATA_FOLDER, 
                                                                   target_size = INPUT_DIM[:2],
                                                                   batch_size = BATCH_SIZE,
                                                                   shuffle = True,
                                                                   class_mode = 'input',
                                                                   subset = 'training'
                                                                   )

# for my_batch in data_flow:
#     labels = my_batch[1]
#     images = my_batch[0]
#     for _ in range(len(labels)):
#         plt.imshow(images[_])
#         plt.colorbar()
#         plt.show()
#         print(labels[_])
#     break

# ENCODER
def build_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, 
                  conv_strides):
  
  # Clear tensorflow session to reset layer index numbers to 0 for LeakyRelu, 
  # BatchNormalization and Dropout.
  # Otherwise, the names of above mentioned layers in the model 
  # would be inconsistent
  global K
  K.clear_session()
  
  # Number of Conv layers
  n_layers = len(conv_filters)

  # Define model input
  encoder_input = Input(shape = input_dim, name = 'encoder_input')
  x = encoder_input

  # Add convolutional layers
  for i in range(n_layers):
      x = Conv2D(filters = conv_filters[i], 
                  kernel_size = conv_kernel_size[i],
                  strides = conv_strides[i], 
                  padding = 'same',
                  name = 'encoder_conv_' + str(i)
                  )(x)

      x = LeakyReLU()(x)
    
  # Required for reshaping latent vector while building Decoder
  shape_before_flattening = K.int_shape(x)[1:] 
  
  x = Flatten()(x)

  # Define model output
  encoder_output = Dense(output_dim, name = 'encoder_output')(x)

  return encoder_input, encoder_output, shape_before_flattening, Model(encoder_input, encoder_output)

encoder_input, encoder_output,  shape_before_flattening, encoder  = build_encoder(input_dim = INPUT_DIM,
                                    output_dim = Z_DIM, 
                                    conv_filters = [32, 64, 64, 64],
                                    conv_kernel_size = [3,3,3,3],
                                    conv_strides = [2,2,2,2])






# Decoder
def build_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, 
                  conv_strides):

  # Number of Conv layers
  n_layers = len(conv_filters)

  # Define model input
  decoder_input = Input(shape = (input_dim,) , name = 'decoder_input')

  # To get an exact mirror image of the encoder
  x = Dense(np.prod(shape_before_flattening))(decoder_input)
  x = Reshape(shape_before_flattening)(x)

  # Add convolutional layers
  for i in range(n_layers):
      x = Conv2DTranspose(filters = conv_filters[i], 
                  kernel_size = conv_kernel_size[i],
                  strides = conv_strides[i], 
                  padding = 'same',
                  name = 'decoder_conv_' + str(i)
                  )(x)
      
      # Adding a sigmoid layer at the end to restrict the outputs 
      # between 0 and 1
      if i < n_layers - 1:
        x = LeakyReLU()(x)
      else:
        x = Activation('sigmoid')(x)

  # Define model output
  decoder_output = x

  return decoder_input, decoder_output, Model(decoder_input, decoder_output)

decoder_input, decoder_output, decoder = build_decoder(input_dim = Z_DIM,
                                        shape_before_flattening = shape_before_flattening,
                                        conv_filters = [64,64,32,3],
                                        conv_kernel_size = [3,3,3,3],
                                        conv_strides = [2,2,2,2]
                                        )

# The input to the model will be the image fed to the encoder.
simple_autoencoder_input = encoder_input

# The output will be the output of the decoder. The term - decoder(encoder_output) 
# combines the model by passing the encoder output to the input of the decoder.
simple_autoencoder_output = decoder(encoder_output)

# Input to the combined model will be the input to the encoder.
# Output of the combined model will be the output of the decoder.
simple_autoencoder = Model(simple_autoencoder_input, simple_autoencoder_output)

simple_autoencoder.summary()

# settting learning rate
LEARNING_RATE = 0.0005
N_EPOCHS = 20

optimizer = Adam(lr = LEARNING_RATE)

def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

simple_autoencoder.compile(optimizer=optimizer, loss = r_loss)

run = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = LOG_DIR + run
# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(
  log_dir=logdir,
  histogram_freq=1,
  write_images=True
)


checkpoint_ae = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'AE/epc200_im256_batch256_'+run+'_weights.h5'), save_weights_only = False, verbose=1)

history1 = simple_autoencoder.fit_generator(data_flow, 
                                 shuffle=True, 
                                 epochs = N_EPOCHS, 
                                 initial_epoch = 0, 
                                 steps_per_epoch=NUM_IMAGES / BATCH_SIZE,
                                 callbacks=[checkpoint_ae,tensorboard])

###### Saving Model
np.save(os.path.join(WEIGHTS_FOLDER, 'AE/epc200_im256_batch256_'+run+'_history.npy'),history1.history)
# history1=np.load('C:/DATA/Code/weights_old/AE/epc200_im256_batch256_20220211-093252_history.npy',allow_pickle='TRUE').item()


# saving whole model
simple_autoencoder.save(os.path.join(WEIGHTS_FOLDER, 'AE/epc200_im256_batch256_'+run+'_simple_autoencoderModel.h5'))
encoder.save(os.path.join(WEIGHTS_FOLDER, 'AE/epc200_im256_batch256_'+run+'_EncoderModel.h5'))
decoder.save(os.path.join(WEIGHTS_FOLDER, 'AE/epc200_im256_batch256_'+run+'_DecoderModel.h5'))

# simple_autoencoder_loaded = load_model(os.path.join(WEIGHTS_FOLDER, 'C:/DATA/Code/weights/AE/epc200_im256_batch256_20220220-144059_weights.h5'),
#                                         compile=False)

# loading whole model
simple_autoencoder = load_model('C:/DATA/Code/weights/AE/epc200_im256_batch256_20220222-104322_simple_autoencoderModel.h5',compile=False)
encoder = load_model('C:/DATA/Code/weights/AE/epc200_im256_batch256_20220222-104322_EncoderModel.h5',compile=False)


# ### Test Loading
# example_batch = next(data_flow)
# example_batch = example_batch[0]
# example_images = example_batch[:2]

# encodings = encoder.predict(example_images)
# encodings_loaded = encoder_loaded.predict(example_images)


train_loss = history1.history['loss']
# val_loss = history1.history['val_loss']

# train_acc = history1.history['accuracy']
# val_acc = history1.history['val_bag_accuracy']

fig = plt.figure()
plt.plot(train_loss)
# plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# save_fig_name = 'Results/' + str(irun) + '_' + str(ifold) + "_loss_batchsize_" + str(batch_size) + "_epoch"  + ".png"
# fig.savefig(save_fig_name)


# fig = plt.figure()
# plt.plot(train_acc)
# # plt.plot(val_acc)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# # save_fig_name = 'Results/' + str(irun) + '_' + str(ifold) + "_val_batchsize_" + str(batch_size) + "_epoch"  + ".png"
# # fig.savefig(save_fig_name)





#### Reconstruct
import matplotlib.pyplot as plt

example_batch = next(data_flow)
example_batch = example_batch[0]
example_images = example_batch[:10]

def plot_compare(images=None, add_noise=False):
  
  if images is None:
    example_batch = next(data_flow)
    example_batch = example_batch[0]
    images = example_batch[:10]

  n_to_show = images.shape[0]

  if add_noise:
    encodings = encoder.predict(images)
    encodings += np.random.normal(0.0, 1.0, size = (n_to_show,200))
    reconst_images = decoder.predict(encodings)

  else:
    reconst_images = simple_autoencoder.predict(images)

  fig = plt.figure(figsize=(3, 3))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
  ssimScore=[]
  psnrs = [] 
  for i in range(n_to_show):
      img0 = images[i].squeeze()
      sub = fig.add_subplot(2, n_to_show, i+1)
      sub.axis('off')        
      sub.imshow(img0)

      img1 = reconst_images[i].squeeze()
      sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
      sub.axis('off')
      sub.imshow(img1) 
      
      (score, diff) = compare_ssim(img0, img1, full=True,multichannel=True)
      ssimScore.append(score)
      psnr = cv2.PSNR(img0, img1)
      psnrs.append(psnr)
      
      # diff = (diff * 255).astype("uint8")
      
  return ssimScore,psnrs

ssimScores,psnrs = plot_compare(example_images)   

print(statistics.mean(psnrs)) 

plot_compare(images = example_images, add_noise = False)


#Attempting to generate images from latent vectors sampled from a standard normal distribution
def generate_images_from_noise(n_to_show = 10): 
  reconst_images = decoder.predict(np.random.normal(0,1,size=(n_to_show,Z_DIM)))

  fig = plt.figure(figsize=(15, 3))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)

  for i in range(n_to_show):
      img = reconst_images[i].squeeze()
      sub = fig.add_subplot(2, n_to_show, i+1)
      sub.axis('off')        
      sub.imshow(img)



generate_images_from_noise()      
















