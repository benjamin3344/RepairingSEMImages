# cyclegan.py
# Raymond Zeng
# April 18, 2020
#
# CycleGAN for converting noisy SEM images to clean SEM images
# Following tutorial at https://www.tensorflow.org/tutorials/generative/cyclegan

import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import pix2pix_gen

# constants
BUFFER_SIZE         = 1024
BATCH_SIZE          = 16
IMG_ROWS_0          = 200                   # height of original input image
IMG_COLS_0          = 333                   # width of original input image
IMG_WIDTH           = 192                   # width of output image
IMG_HEIGHT          = 192                   # height of output image
CLEAN_DATA_FILE     = "data_small.npy"      # npy file containing clean images
NOISY_DATA_FILE     = "blurry_small.npy"    # npy file containing noisy images
TRAINING_IMAGES     = 1000                  # rest of images go in the test set
LAMBDA              = 10                    # for calculating losses
OUTPUT_CHANNELS     = 1                     # grayscale
EPOCHS              = 200                   # epochs to train for
OUTPUT_IMAGES       = 100                   # number of images to output

# randomly crop images
def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 1])
    return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

# random jittering (random crop + random flip left/right)
def random_jitter(image):
    image = tf.image.resize(image, [IMG_ROWS_0, IMG_COLS_0],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image):
    image = normalize(image)
    return image

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

def generate_images(model, test_input, index, name):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    test_input = tf.reshape(test_input, [1, IMG_WIDTH, IMG_HEIGHT])
    prediction = tf.reshape(prediction, [1, IMG_WIDTH, IMG_HEIGHT])
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig("test3/" + name + ".png")
    plt.close()

# @tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)
        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

AUTOTUNE = tf.data.experimental.AUTOTUNE # optimize input pipeline

# load datasets
clean_dataset = np.load(CLEAN_DATA_FILE)
noisy_dataset = np.load(NOISY_DATA_FILE)

clean_train = tf.data.Dataset.from_tensor_slices(clean_dataset[:TRAINING_IMAGES])
clean_test = tf.data.Dataset.from_tensor_slices(clean_dataset[TRAINING_IMAGES:])
noisy_train = tf.data.Dataset.from_tensor_slices(noisy_dataset[:TRAINING_IMAGES])
noisy_test = tf.data.Dataset.from_tensor_slices(noisy_dataset[TRAINING_IMAGES:])

clean_train = clean_train.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

clean_test = clean_test.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

noisy_train = noisy_train.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

noisy_test = noisy_test.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

# create sample images from training and test sets
sample_clean = next(iter(clean_train))[0]
sample_noisy = next(iter(noisy_train))[0]
sample_clean = random_jitter(sample_clean)
sample_noisy = random_jitter(sample_noisy)

print("Creating generators...")

# use pix2pix generator and discriminator
generator_g = pix2pix_gen.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix_gen.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

print("Creating discriminators...")

discriminator_x = pix2pix_gen.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix_gen.discriminator(norm_type='instancenorm', target=False)

print("Generators and discriminators created")

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# create checkpoints
checkpoint_path = "./test3/"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

sample_noisy = tf.reshape(sample_noisy, [1, IMG_WIDTH, IMG_HEIGHT, 1])
generate_images(generator_f, sample_noisy, 0, "0")

# perform training
for epoch in range(EPOCHS):
    start = time.time()
    print("Epoch " + str(epoch + 1) + ":")

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((clean_train, noisy_train)):
        train_step(image_x, image_y)
        if n % 20 == 0 and n > 0:
            print ("\tProcessed ", n, "images...")
        n += 1

    # Using a consistent image (sample_noisy) so that the progress of the model
    # is clearly visible.
    generate_images(generator_f, sample_noisy, epoch + 1, str(epoch + 1))

    # save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

# print output images
for i in range(OUTPUT_IMAGES):
    sample_test = next(iter(noisy_test))[0]
    sample_test = random_jitter(sample_test)
    sample_test = tf.reshape(sample_test, [1, IMG_WIDTH, IMG_HEIGHT, 1])
    generate_images(generator_f, sample_test, i + 1, str(i + 1))
