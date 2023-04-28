import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Separate loss functions for generator and discriminator


# Binary design variables example
class BinaryDesignExample:
    def __init__(self):
        self.num_design_vars = 12
        self.num_examples = 8192
        self.latent_dim = 256
        self.batch_size = 128
        self.num_epochs = 200
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.lambda_s = 0.2
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)


        # Build the discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.discriminator_loss,
                                   optimizer=self.discriminator_optimizer,
                                   metrics=['accuracy'])

        # Build the generator model
        self.generator = self.build_generator()
        self.generator.compile(loss=self.generator_loss,
                               optimizer=self.generator_optimizer)

        # Build the GAN model
        self.gan = self.build_gan()
        self.gan.compile(loss=self.generator_loss, 
                         optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta1))





    def generator_loss(self, fake_output):

      return tf.reduce_mean(tf.math.log(1 - fake_output))

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_mean(tf.math.log(real_output))
        fake_loss = tf.reduce_mean(tf.math.log(1 - fake_output))
        return -(real_loss + fake_loss)


    def objective_loss(self, fake_samples):
    
        binary_sum = tf.reduce_mean(tf.reduce_sum(fake_samples, axis=1))
        mse = (8-binary_sum)**2
        sum_penalty = tf.maximum(mse, 0)

        return sum_penalty



    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.latent_dim, activation='relu'))
        model.add(layers.Dense(64, input_dim=128, activation='relu'))
        model.add(layers.Dense(32, input_dim=64, activation='relu'))
        model.add(layers.Dense(16, input_dim=32, activation='relu'))
        model.add(layers.Dense(self.num_design_vars, activation='sigmoid'))
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.num_design_vars, activation='relu'))
        model.add(layers.Dense(64, input_dim=128, activation='relu'))
        model.add(layers.Dense(32, input_dim=64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def build_gan(self):
        # Freeze the discriminator's weights during GAN training
        self.discriminator.trainable = False

        # Build the GAN by stacking the generator and discriminator
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = Model(gan_input, gan_output)
        return gan

    """def generate_real_samples(self, num_samples):
        X = np.random.randint(2, size=(num_samples, self.num_design_vars))
        X = np.asarray([1 if sum(x[:5]) >= 3 else 0] * 5)
        y = np.ones((num_samples, 1))
        return X, y"""


    def generate_real_samples(self, num_samples):
      # Initialize the binary matrix
      binary_matrix = np.zeros((num_samples, self.num_design_vars))

      # Generate each row of the binary matrix
      for i in range(num_samples):
          row_sum = 0
          while row_sum != 8:
              # Generate a random binary vector
              binary_vector = np.random.randint(2, size=self.num_design_vars)

              # Calculate the sum of the binary vector
              row_sum = np.sum(binary_vector)

          # Add the binary vector to the binary matrix
          binary_matrix[i,:] = binary_vector

      y = np.ones((num_samples, 1))
      return binary_matrix,y

    def generate_latent_points(self, num_samples):
        X = np.random.normal(0, 1, (int(num_samples), self.latent_dim))
        return X


    """def generate_fake_samples(self, training):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        fake_samples = self.generator(noise,training = training)
        y = np.zeros((self.batch_size, 1))
        fake_samples[fake_samples >= 0.5] = 1
        fake_samples[fake_samples < 0.5] = 0
        return fake_samples, y"""
    
    def generate_fake_samples(self, training):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        fake_samples = self.generator(noise, training=training)
        fake_samples_thresh = tf.where(fake_samples >= 0.5, tf.ones_like(fake_samples), tf.zeros_like(fake_samples))
        y = tf.zeros((self.batch_size, 1))
        return fake_samples, fake_samples_thresh 



    def create_batches(self,data):
        # create an array of indices for the dataset
        indices = np.arange(data.shape[0])
        # shuffle the indices
        np.random.shuffle(indices)
        batch_data = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch = data[batch_indices]
            batch_data.append(batch)
        return batch_data


        
    def train(self):
      # Initialize the GAN training process
        num_batches = int(self.num_examples / self.batch_size)

        """for epoch in range(self.num_epochs):
            for batch_idx in range(num_batches):
                # Train the discriminator on a batch of real and generated samples
                real_samples, real_labels = self.generate_real_samples(self.batch_size)
                fake_samples, fake_labels = self.generate_fake_samples(self.batch_size)
                discriminator_loss_real, discriminator_accuracy_real = self.discriminator.train_on_batch(real_samples, real_labels)
                discriminator_loss_fake, discriminator_accuracy_fake = self.discriminator.train_on_batch(fake_samples, fake_labels)
                discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
                discriminator_accuracy = 0.5 * np.add(discriminator_accuracy_real, discriminator_accuracy_fake)

                # Train the generator to trick the discriminator
                fake_samples,gen_labels = self.generate_fake_samples(self.batch_size)
                fake_output = self.discriminator.predict(fake_samples)
                generator_loss, generator_accuracy = self.gan.train_on_batch()"""

        d_losses = []
        g_losses = []
        obj_losses = []
        gan_losses = []
        real_data, real_labels = self.generate_real_samples(self.num_examples)
        real_data_sliced = self.create_batches(real_data)

        
        for epoch in range(self.num_epochs):
            g_losses_batch = []
            d_losses_batch = []
            obj_losses_batch = []
            gan_losses_batch = []
            for batch in real_data_sliced:
            
                with tf.GradientTape() as tape:
                    #generated_samples = self.generator(noise, training=False)
                    generated_samples, fake_samples_thresh=self.generate_fake_samples(training=False)
                    
                    real_output = self.discriminator(batch, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)
                    d_loss = self.discriminator_loss(real_output, fake_output)
                    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                    self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
                
                # Train the generator
            
                with tf.GradientTape() as tape:
                    #generated_samples = self.generator(noise, training=True)
                    generated_samples, generated_samples_thresh=self.generate_fake_samples(training=True)
                    fake_output = self.discriminator(generated_samples, training=False)
                    g_loss = self.generator_loss(fake_output)
                    obj_loss = self.objective_loss(generated_samples)
                    g_loss_obj = g_loss + self.lambda_s*obj_loss
                    grads = tape.gradient(g_loss_obj, self.generator.trainable_weights)
                    self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

                g_losses_batch.append(g_loss)
                obj_losses_batch.append(obj_loss)
                gan_losses_batch.append(g_loss_obj)
                d_losses_batch.append(d_loss)


            d_losses_m = tf.reduce_mean(d_losses_batch)
            g_losses_m = tf.reduce_mean(g_losses_batch)
            obj_losses_m = tf.reduce_mean(obj_losses_batch)
            gan_losses_m = tf.reduce_mean(gan_losses_batch)

            g_losses.append(g_losses_m)
            obj_losses.append(obj_losses_m)
            gan_losses.append(gan_losses_m)
            d_losses.append(d_losses_m)


            print(f"Epoch: {epoch + 1}/{self.num_epochs}, Discriminator Loss: {d_losses_m}, Generator Loss: {g_losses_m}, Obj_loss = {obj_losses_m}")
            if epoch==self.num_epochs-1:
                print(generated_samples_thresh)


        plt.figure('G losses')
        plt.plot(g_losses, label='Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


        plt.figure('D losses')
        plt.plot(d_losses, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


        plt.figure('Obj losses')
        plt.plot(obj_losses, label='Objective loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


        plt.figure('GAN losses')
        plt.plot(obj_losses, label='GAN loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
                

      #print(f"Epoch: {epoch + 1}/{self.num_epochs}, Discriminator Loss: {d_loss}, Discriminator Accuracy: {discriminator_accuracy}, Generator Loss: {g_loss}, Generator Accuracy: {generator_accuracy}")

      



                




GAN = BinaryDesignExample()
GAN.train()
