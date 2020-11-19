import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers, losses
import time
import tensorflow as tf
from IPython import display
import h5py
from unet import unet_model
from sklearn.model_selection import train_test_split
# generator = tf.keras.models.load_model('saved_model/color_iter')
from tensorflow.keras.models import Model

# DANIAR SHENANIGANS: #####

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# DANIAR SHENANIGANS ENDDDD ######

OUTPUT_CHANNELS = 3
alpha = .8


'''class make_generator_model(Model):
    def __init__(self):
        super(make_generator_model, self).__init__()
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

'''


# define the standalone discriminator model
def make_discriminator_model(in_shape=(256, 256, 3)):
    model = tf.keras.Sequential()
    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile model
    # opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# def make_discriminator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=[256, 256, 3]))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))
#
#     model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))
#
#     model.add(layers.Flatten())
#     model.add(layers.Dense(1))
#
#     return model

generator = unet_model(OUTPUT_CHANNELS)
#generator = make_generator_model()
discriminator = make_discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(real_input, generated_output, fake_output):
    return (1 - alpha) * cross_entropy(tf.ones_like(fake_output), fake_output) + alpha * tf.reduce_mean(
        tf.math.squared_difference(generated_output, real_input))
    # tf.reduce_mean(tf.math.squared_difference(generated_output,real_input))


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
BUFFER_SIZE = 60000
BATCH_SIZE = 8
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
'''@tf.function
def train_step(images):
    ###XXX replace noise with second param
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator.encoder(images[0])
        generated_images = generator.decoder(generated_images)
        # print(images[0].shape)
        # print(images[1].shape)

        real_output = discriminator(images[2], training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(images[1], generated_images, fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))'''


@tf.function
def train_generator(images):
    with tf.GradientTape() as gen_tape:
        '''generated_images = generator.encoder(images[0])
        generated_images = generator.decoder(generated_images)'''
        generated_images = generator(images[0])
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(images[1], generated_images, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


@tf.function
def train_discriminator(images):
    with tf.GradientTape() as disc_tape:
        '''generated_images = generator.encoder(images[0])
        generated_images = generator.decoder(generated_images)'''
        generated_images = generator(images[0])
        real_output = discriminator(images[2], training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        photo_op = dataset[0]
        for image_batch in dataset:
            train_generator(image_batch)
            train_discriminator(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1, photo_op)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs, photo_op)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    '''predictions = model.encoder(test_input[0])
    predictions = model.decoder(predictions)'''
    predictions = model(test_input[0])
    fig = plt.figure(figsize=(4, 4))
    # print(predictions[i])
    for i in range(BATCH_SIZE):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.squeeze(predictions[i]))
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


f = h5py.File('./cache.hdf5', 'r')
x_data_water = f['input_data_adobe_2'][0:1000]
x_data_unwater = f['exput_data_adobe_2'][0:1000]
x_data_real = f['exput_data_adobe_2'][1000:2000]

generate_and_save_images(generator, 0, (x_data_water[0:BATCH_SIZE], None))
# x_data_real = f['exput_data_adobe-2'][3000:6000]
f.close()
# X_train, X_test, y_train, y_test = train_test_split(x_data, x_data_water, test_size=0.002, random_state=42)
x_data_water = tf.data.Dataset.from_tensor_slices(x_data_water).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
x_data_unwater = tf.data.Dataset.from_tensor_slices(x_data_unwater).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
x_data_real = tf.data.Dataset.from_tensor_slices(x_data_real).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
training_data = list(zip(x_data_water, x_data_unwater, x_data_real))
train(training_data, EPOCHS)
'''n = 8
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(training_data[0][0][i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(training_data[0][1][i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()
training_data = list(zip(x_data_water,x_data_real))
train(training_data, EPOCHS)'''
