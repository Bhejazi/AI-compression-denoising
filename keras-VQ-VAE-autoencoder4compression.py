#%% Load libraries
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:52:42 2025

Vector-Quantized Variational Autoencoders (VQ-VAEs) for image compression

@author: bhejazi
"""
import os
from PIL import Image
from colorama import Fore, Style
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import tensorflow as tf

from sklearn.model_selection import train_test_split

#%% VectorQuantizer layer
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value = w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae")

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True) 
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity)

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
#%% Encoder and decoder

# !!! Make sure to adjust the layers properly with mirror number of covolutions in layers !!!
# edit number of layers for different compression rates
l_dim = 8
n_embeddings = 64

def get_encoder(latent_dim=l_dim):
    encoder_inputs = keras.Input(shape=(512, 512, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x) # low compression
    #x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x) # mid compression
    #x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x) # high compression
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

def get_decoder(latent_dim=l_dim):
    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    #x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(latent_inputs)
    #x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(latent_inputs)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

#%% Standalone VQ-VAE model
# latent_dim equal to the value set in previous step
# num_embeddings equal to number of filters in encoder and decoder models
def get_vqvae(latent_dim=l_dim, num_embeddings=n_embeddings):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(512, 512, 1))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")

get_vqvae().summary()

#%% Loss functions
# Here latent_dim and num_embeddings are x2 the values from the previous steps 
# since both encoder and decoder are in the VQ-VAE model
class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=2*l_dim, num_embeddings=2*n_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
    
#%% Preparing Dataset
# Remembr to change dtype depending on image format
def load_images(path):
    folder_info = os.listdir(path)
    print("Loading images..")
        
    file_idx = [i for i, file_name in enumerate(folder_info) if file_name[-4:] == "tiff"]
    #file_idx = [i for i, file_name in enumerate(folder_info)]
    
    n_images = len(file_idx)
    temp_img = np.array(Image.open(os.path.join(path, folder_info[file_idx[0]])))
    rows, cols = temp_img.shape
    
    images = np.zeros((n_images, rows, cols), dtype="uint8")

    for i, idx in enumerate(file_idx):
        images[i, :, :] = np.array(Image.open(os.path.join(path, folder_info[idx])))
      
    return images
       
while True:
    print("\n" + "Enter image file path:" + Fore.BLUE + Style.BRIGHT)
    path = input ("-> ")
    print(Style.RESET_ALL)

    if os.path.exists(path): #check if path is valid
        break
    else:
        print (Fore.RED + Style.BRIGHT + "Path does not exist, re-enter correct path")

images = load_images(path)

x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

# normalize the image data
if np.max(x_train) > 255:
    x_train = x_train.astype('float32') / 65535
    x_test = x_test.astype('float32') / 65535
elif np.max(x_train) > 1 & np.max(x_train) < 255:
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

img_size = images.shape[1]
# reshape in the input data for the model
x_train = x_train.reshape(len(x_train), img_size, img_size, 1)
x_test = x_test.reshape(len(x_test), img_size, img_size, 1)

x_train_scaled = (x_train) - 0.5
x_test_scaled = (x_test) - 0.5

data_variance = np.var(x_train)

#%% Train the VQ-VAE model
# Here latent_dim is the same as in the 'Encoder and decoder' section and num_embeddings is x2 the values from the 'Encoder and decoder' section
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=l_dim, num_embeddings=2*n_embeddings)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer.fit(x_train_scaled, epochs=100, batch_size=128)

#%% Reconstruction results on the test set
def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5, cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

trained_vqvae_model = vqvae_trainer.vqvae
idx = np.random.choice(len(x_train_scaled), 3)
#test_images = x_test_scaled[idx]
train_images = x_train_scaled
reconstructions_train = trained_vqvae_model.predict(train_images)

train_images_vis = train_images[idx]
reconstructions_train_vis = reconstructions_train[idx]

for train_image_vis, reconstructed_image_vis in zip(train_images_vis, reconstructions_train_vis):
    show_subplot(train_image_vis, reconstructed_image_vis)
    
#%% Encoded images
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

# decoder = vqvae_trainer.vqvae.get_layer("decoder")

encoded_outputs = encoder.predict(train_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

codebook_indices_vis = codebook_indices[idx]

# decoded_outputs = decoder.predict(encoded_outputs)

for i in range(len(train_images_vis)):
    plt.subplot(1, 2, 1)
    plt.imshow(train_images_vis[i].squeeze() + 0.5, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices_vis[i], cmap="gray")
    plt.title("Code")
    plt.axis("off")
    plt.show()

#%% Save images
size_conversion = "512x512-to-64x64x4"
file_extension = "binary-images-VQ-VAE"

encoded_path = f"M:/codes/codes_BAM/python/autoencoder/temp/encoded/encoded_compress_{size_conversion}-{file_extension}.npy"
decoded_path = f"M:/codes/codes_BAM/python/autoencoder/temp/decoded/decoded_compress_{size_conversion}-{file_extension}.npy"
train_images_path = f"M:/codes/codes_BAM/python/autoencoder/temp/train_images_compress-{file_extension}.npy"

np.save(encoded_path, encoded_outputs)
np.save(decoded_path, reconstructions_train)
# np.save(train_images_path, train_images)

#%% Model visualization
import visualkeras
visualkeras.layered_view(get_encoder(), draw_volume=True, spacing=100, scale_xy=1, scale_z=1)
visualkeras.layered_view(get_decoder(), draw_volume=True, spacing=100, scale_xy=1, scale_z=1, draw_reversed=True)