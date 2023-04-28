import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse
from keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define model parameters
batch_size = 100
latent_dim = 2
intermediate_dim = 256
epochs = 20

# Encoder architecture
inputs = Input(shape=(784,), name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Sampling latent space
z = Lambda(sampling, output_shape=(latent_dim,),
           name='z')([z_mean, z_log_var])

# Decoder architecture
decoder_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(intermediate_dim, activation='relu')(decoder_inputs)
outputs = Dense(784, activation='sigmoid')(x)

# Define the models
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_inputs, outputs, name='decoder')
cvae_outputs = decoder(encoder(inputs)[2])
cvae = Model(inputs, cvae_outputs, name='cvae')

# Define the loss function
reconstruction_loss = mse(inputs, cvae_outputs)
reconstruction_loss *= 784
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
cvae.add_loss(vae_loss)
cvae.compile(optimizer='rmsprop')

# Train the model
cvae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None), verbose=2)

# Evaluate the model
z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
print(z_mean)

