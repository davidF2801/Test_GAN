import numpy as np
import tensorflow.keras.datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('seaborn')




def build_generator(latent_dim, height, width, channels):
  inputs = layers.Input(shape=(latent_dim,))
  x = layers.Dense(128 * 16 * 16)(inputs)
  x = layers.LeakyReLU()(x)
  x = layers.Reshape((16, 16, 128))(x)
  x = layers.Conv2D(256, 5, padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(256, 5, padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(256, 5, padding='same')(x)
  x = layers.LeakyReLU()(x)
  output = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
  model = Model(inputs, output)
  print(model.summary())
  return model



def build_discriminator(height, width, channels):
  inputs = layers.Input(shape=(height, width, channels))
  x = layers.Conv2D(128, 3)(inputs)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(128, 4, strides=2)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(128, 4, strides=2)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(128, 4, strides=2)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.4)(x)
  output = layers.Dense(1, activation='sigmoid')(x)
  model = Model(inputs, output)
  optimizer = optimizers.RMSprop(learning_rate=0.0008, clipvalue=1.0)
  model.compile(optimizer=optimizer, loss='binary_crossentropy')
  print(model.summary())
  return model


def load_dataset(class_label):
  (X_train, y_train), (X_test, y_test) = tfds.cifar10.load_data()
  X_train = X_train[y_train.flatten() == class_label]
  X_test = X_test[y_test.flatten() == class_label]
  X_train = X_train.astype('float64') / 255.0
  X_test = X_test.astype('float64') / 255.0
  combined_data = np.concatenate([X_train, X_test])
  return combined_data



def train_gan(batch_size, iter_num, latent_dim, data, gan_model, generator_model, discriminator_model):
  start = 0
  d_loss_lst, gan_loss_lst = [], []
  for step in range(iter_num):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator_model.predict(random_latent_vectors)
    real_images = data[start: start + batch_size]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.zeros((batch_size, 1)),
                             np.ones((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    d_loss = discriminator_model.train_on_batch(combined_images, labels)
    d_loss_lst.append(d_loss)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_labels = np.ones((batch_size, 1))
    gan_loss = gan_model.train_on_batch(random_latent_vectors, misleading_labels)
    gan_loss_lst.append(gan_loss)

    if step % 200 == 0:
      print("Iteration {0}/{1}".format(step, iter_num))
      print("[==============================] d-loss: {0:.3f}, gan-loss: {1:.3f}".format(d_loss_lst[-1], gan_loss_lst[-1]))
    
    start += batch_size
    if start > len(data) - batch_size:
      start = 0

  return gan_model, generator_model, discriminator_model, d_loss_lst, gan_loss_lst

def show_generated_image(generator_model, latent_dim, row_num=4):
  num_image = row_num**2
  random_latent_vectors = np.random.normal(size=(num_image, latent_dim))
  generated_images = generator_model.predict(random_latent_vectors)
  plt.figure(figsize=(10,10))
  for i in range(num_image):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.subplot(row_num,row_num,i+1)
    plt.grid(False)
    plt.xticks([]); plt.yticks([])
    plt.imshow(img)
  plt.show()


def plot_learning_curve(d_loss_lst, gan_loss_lst):
  fig = plt.figure()
  plt.plot(d_loss_lst, color='skyblue')
  plt.plot(gan_loss_lst, color='gold')
  plt.title('Model Learning Curve')
  plt.xlabel('Epochs'); plt.ylabel('Cross Entropy Loss')
  plt.show()






latent_dim = 128
height = 32
width = 32
channels = 3
generator = build_generator(latent_dim, height, width, channels)
plot_model(generator, show_shapes=True, show_layer_names=True)



discriminator = build_discriminator(height, width, channels)
plot_model(discriminator, show_shapes=True, show_layer_names=True)


discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
optimizer = optimizers.RMSprop(learning_rate=0.0004, clipvalue=1.0)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')
print(gan.summary())
plot_model(gan, show_shapes=True, show_layer_names=True)


batch_size = 32
iter_num = 10000
train_image = load_dataset(8)
gan, generator, discriminator, d_history, gan_history = train_gan(batch_size, iter_num, latent_dim, train_image, gan, generator, discriminator)
show_generated_image(generator, latent_dim)
plot_learning_curve(d_history, gan_history)


