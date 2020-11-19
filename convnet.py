
from PIL import Image
import h5py
import os
import numpy as np
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
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

hdf5_store = h5py.File("./cache.hdf5", "a",libver='latest')
#results = hdf5_store.create_dataset("results", (1000,1000,1000,5), compression="gzip")

# do something...
#results[100, 25, 1, 4] = 42

def load_data_2():
    folder = "/Users/gollyzoom/Downloads/test_h5py/input_data/"
    #hdf5_store = h5py.File("test.hdf5", "a")


    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    onlyfiles = [f for f in onlyfiles if "adobe" in f]
    print(onlyfiles[0:5])
    print(len(onlyfiles))
    train_files = []
    y_train = []
    i=0
    for _file in onlyfiles:
        #if(i < 10400):
        train_files.append(_file)
        i += 1
        if( i >= 10400):
            break
    i = 0
    print("Files in train_files: %d" % len(train_files))
    n = len(train_files)
    input_data = hdf5_store.create_dataset("input_data_adobe_2", (10000, 256,256,3), compression="gzip")
    #imgs = np.empty((n,3,256,256), dtype='uint16')
    num = 0
    image_width = 256
    image_height = 256
    ratio = 1

    image_width = int(image_width / ratio)
    image_height = int(image_height / ratio)

    channels = 3
    nb_classes = 1
    dataset = np.ndarray(shape=(1000,  image_height, image_width,channels,),
                         dtype=np.float32)
    for _file in train_files:
        img = load_img(folder + _file)  # this is a PIL image
        img.thumbnail((image_width, image_height))
        # Convert to Numpy Array
        x = img_to_array(img)
        #x = x.reshape((256, 256,3))
        # Normalize
        x = (x - 128.0) / 128.0
        dataset[num%1000,:,:,:] = x
        #input_data[num,:,:,:] = x
        num +=1
        if(num % 1000 ==0):
            print("hey")
            factor = num/1000
            #input_data.resize((input_data.shape[0] + dataset.shape[0]), axis=0)
            input_data[int((factor-1)*1000):int(factor*1000)] = dataset
            print("done")
        #if (num % 200 == 0):
        #print(num)
    print(input_data[1:])
    
    
    
    folder = "/Users/gollyzoom/Downloads/test_h5py/output_data/"
    #hdf5_store = h5py.File("test.hdf5", "a")


    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    onlyfiles = [f for f in onlyfiles if "adobe" in f]
    print(onlyfiles[0:5])
    print(len(onlyfiles))
    train_files = []
    y_train = []
    i=0
    for _file in onlyfiles:
        train_files.append(_file)
        i += 1
        if (i >= 10400):
            break
        #label_in_file = _file.find("_")
        #y_train.append(int(_file[0:label_in_file]))'''
    i = 0
    print("Files in train_files: %d" % len(train_files))
    n = len(train_files)
    output_data = hdf5_store.create_dataset("exput_data_adobe_2", (10000,256,256,3), compression="gzip")
    #imgs = np.empty((n,3,256,256), dtype='uint16')
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
        # input_data[num,:,:,:] = x
        num += 1
        if (num % 1000 == 0):
            print("hey")
            factor = num / 1000
            # input_data.resize((input_data.shape[0] + dataset.shape[0]), axis=0)
            output_data[int((factor - 1) * 1000):int(factor * 1000)] = dataset
            print("done")
        # if (num % 200 == 0):
        # print(num)
    print(output_data[1:])
    hdf5_store.close()

#def load_training_data():

#load_training_data()
#load_data_2()

#x_data,x_data_water = load_data()
#print(x_data.shape)
#x_data = x_data[:,0,:]#np.mean(x_data, axis=1)
#im2display = x_data[1].transpose((1,2,0))

'''im2display = np.reshape(x_data[8],(x_data.shape[2],x_data.shape[3],x_data.shape[1]))
print(im2display.shape)
fig = plt.figure(figsize=(256, 256))
sub = fig.add_subplot(1, 1, 1)
sub.imshow(im2display, interpolation='nearest')
plt.show()'''
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

'''x_data = np.reshape(x_data,(x_data.shape[0],x_data.shape[2],x_data.shape[3],x_data.shape[1]))
x_data_water = np.reshape(x_data_water,(x_data_water.shape[0],x_data_water.shape[2],x_data_water.shape[3],x_data_water.shape[1]))

x_data = rgb2gray(x_data)
x_data_water = rgb2gray(x_data_water)

print(x_data.shape)
print(x_data_water.shape)

x_train = x_data[0:-100].astype('float32') / 255.
x_test = x_data[-100:].astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

x_train_noisy = x_data_water[0:-100].astype('float32') / 255.
x_test_noisy = x_data_water[-100:].astype('float32') / 255.


x_train_noisy = x_train_noisy[..., tf.newaxis]
x_test_noisy = x_test_noisy[..., tf.newaxis]

print("after last dim")
print(x_train.shape)
print(x_train_noisy.shape)'''


print(tf.__version__)

class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    '''self.encoder = tf.keras.Sequential([
      layers.Input(shape=(256, 256, 3)),
      layers.Conv2D(16, (4,4), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (4,4), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=4, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=4, strides=2, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=(4,4), activation='linear', padding='same')])'''
    '''self.encoder = tf.keras.Sequential([
        layers.Input(shape=(256, 256, 3)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same')
    ])

    self.decoder = tf.keras.Sequential([
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),

        layers.UpSampling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),

        layers.UpSampling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),

        layers.UpSampling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

        layers. Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])'''
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

    '''self.encoder = tf.keras.Sequential([
      layers.Input(shape=(256,256,3)),
      layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.UpSampling2D((2, 2)),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.UpSampling2D((2, 2)),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),

      layers.Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])'''
    '''self.encoder = Sequential()
    self.encoder.add(InputLayer(img.shape))
    self.encoder.add(Flatten())
    self.encoder.add(Dense(code_size))

    # The decoder
    self.decoder = Sequential()
    self.decoder.add(InputLayer((code_size,)))
    self.decoder.add(Dense(np.prod(img.shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    self.decoder.add(Reshape(img_shape))'''
      # Encoder


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
print("pineapple")
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.5), loss=losses.MeanSquaredError())
autoencoder.encoder.summary()
print("---")

n = 1
plt.figure(figsize=(1, 1))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(y_test[i])
    plt.gray()
plt.show()

#autoencoder.decoder.summary()

autoencoder.fit(X_train, y_train,
                epochs=50,
                shuffle=False,
                validation_data=(X_test, y_test))
encoded_imgs = autoencoder.encoder(X_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
autoencoder.save('saved_model/100_iter') 
autoencoder.decoder.summary()
'''n = 10
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
plt.show()'''
