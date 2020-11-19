import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
from unet import unet_model
# DANIAR SHENANIGANS: #####

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# DANIAR SHENANIGANS ENDDDD ######

#generator = tf.keras.models.load_model('saved_model')
#generator = Generator()
generator = unet_model(3)
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint('./training_checkpoints'))

autoencoder = tf.keras.models.load_model('saved_model')


hdf5_store = h5py.File("./cache.hdf5", "r", libver='latest')
'''input_data = hdf5_store.create_dataset("messy_input_9", (1000, 256, 256, 3), compression="gzip")
output_data = hdf5_store.create_dataset("clean_output_9", (1000, 256, 256, 3), compression="gzip")

#input_data = hdf5_store['input_data_adobe_2'][0:2000]
output_data = hdf5_store['exput_data_adobe_2'][0:1000]
#input_data = generator(hdf5_store['input_data_adobe_2'][0:1000])
for i in range(100):
    print(i)
    input_data[i*10:(i+1)*10] = generator(hdf5_store['input_data_adobe_2'][i*10:(i+1)*10])
'''
input_data = hdf5_store['messy_input_9'][...]
output_data = hdf5_store['exput_data_adobe_2'][0:1000]
watermarked = hdf5_store['input_data_adobe_2'][0:1000]
hdf5_store.close()
#print(x_data.shape)

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.02, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(input_data, watermarked, test_size=0.02, random_state=42)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(y_test[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(X_test[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

plt.show()

#opt = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.5)
#autoencoder.compile(optimizer=opt)
autoencoder.fit(X_train, y_train,
                batch_size=8,
                epochs=50,
                shuffle=False,
                validation_data=(X_test, y_test))
encoded_imgs = autoencoder.encoder(X_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(X_test[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()