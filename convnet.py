from PIL import Image
import h5py
import os
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np

hdf5_store = h5py.File("./cache.hdf5", "a", libver='latest')

def load_data_2():
    folder = "/Users/gollyzoom/Downloads/test_h5py/input_data/"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    onlyfiles = [f for f in onlyfiles if "adobe" in f]
    print(onlyfiles[0:5])
    print(len(onlyfiles))
    train_files = []
    y_train = []
    i = 0
    for _file in onlyfiles:
        # if(i < 10400):
        train_files.append(_file)
        i += 1
        if (i >= 10400):
            break
    i = 0
    print("Files in train_files: %d" % len(train_files))
    n = len(train_files)
    input_data = hdf5_store.create_dataset("input_data_adobe_2", (10000, 256, 256, 3), compression="gzip")
    # imgs = np.empty((n,3,256,256), dtype='uint16')
    num = 0
    image_width = 256
    image_height = 256
    ratio = 1

    image_width = int(image_width / ratio)
    image_height = int(image_height / ratio)

    channels = 3
    nb_classes = 1
    dataset = np.ndarray(shape=(1000, image_height, image_width, channels,),
                         dtype=np.float32)
    for _file in train_files:
        img = load_img(folder + _file)  # this is a PIL image
        img.thumbnail((image_width, image_height))
        # Convert to Numpy Array
        x = img_to_array(img)
        # Normalize
        x = (x - 128.0) / 128.0
        dataset[num % 1000, :, :, :] = x
        # input_data[num,:,:,:] = x
        num += 1
        if (num % 1000 == 0):
            print("hey")
            factor = num / 1000
            # input_data.resize((input_data.shape[0] + dataset.shape[0]), axis=0)
            input_data[int((factor - 1) * 1000):int(factor * 1000)] = dataset
            print("done")
        # if (num % 200 == 0):
        # print(num)
    print(input_data[1:])

    folder = "/Users/gollyzoom/Downloads/test_h5py/output_data/"
    # hdf5_store = h5py.File("test.hdf5", "a")

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    onlyfiles = [f for f in onlyfiles if "adobe" in f]
    print(onlyfiles[0:5])
    print(len(onlyfiles))
    train_files = []
    y_train = []
    i = 0
    for _file in onlyfiles:
        train_files.append(_file)
        i += 1
        if (i >= 10400):
            break
    i = 0
    print("Files in train_files: %d" % len(train_files))
    n = len(train_files)
    output_data = hdf5_store.create_dataset("exput_data_adobe_2", (10000, 256, 256, 3), compression="gzip")
    num = 0
    image_width = 256
    image_height = 256
    ratio = 1

    image_width = int(image_width / ratio)
    image_height = int(image_height / ratio)

    channels = 3
    nb_classes = 1
    dataset = np.ndarray(shape=(1000, image_height, image_width, channels,),
                         dtype=np.float32)
    for _file in train_files:
        img = load_img(folder + _file)  # this is a PIL image
        img.thumbnail((image_width, image_height))
        # Convert to Numpy Array
        x = img_to_array(img)
        # x = x.reshape((256, 256,3))
        # Normalize
        x = (x - 128.0) / 128.0
        dataset[num % 1000, :, :, :] = x
        num += 1
        if (num % 1000 == 0):
            print("hey")
            factor = num / 1000
            output_data[int((factor - 1) * 1000):int(factor * 1000)] = dataset
            print("done")
    print(output_data[1:])
    hdf5_store.close()

### CREATE H5PY FILES ###
# def load_training_data():

# load_training_data()
# load_data_2()

# x_data,x_data_water = load_data()
# print(x_data.shape)
# x_data = x_data[:,0,:]#np.mean(x_data, axis=1)
# im2display = x_data[1].transpose((1,2,0))
#########################




def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

print(tf.__version__)

class SimpleAuto(Model):
    def __init__(self):
        super(SimpleAuto, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 3)),
            layers.Conv2D(16, (4, 4), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (4, 4), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(4, 4), activation='linear', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 3)),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


f = h5py.File('./cache.hdf5', 'r')
x_data = f['input_data_adobe_2'][0:1000]
x_data_water = f['exput_data_adobe_2'][0:1000]
f.close()
print(x_data.shape)
X_train, X_test, y_train, y_test = train_test_split(x_data, x_data_water, test_size=0.002, random_state=42)

autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.encoder.summary()

n = 1
plt.figure(figsize=(1, 1))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(y_test[i])
    plt.gray()
plt.show()

autoencoder.decoder.summary()

autoencoder.fit(X_train, y_train,
                epochs=50,
                shuffle=False,
                validation_data=(X_test, y_test))
encoded_imgs = autoencoder.encoder(X_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
autoencoder.save('saved_model/100_iter')
autoencoder.decoder.summary()
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
