import tensorflow as tf
import numpy as np
'''
We are looking for a cost function in thus form:
min_G max_D V(D, G) = E[x ~ p_data(x)][log D(x)] + lambda_s * E[z ~ p_z(z)][log(1 - D(G(z)))] - lambda_c * C(G(z))

lambda_c will control the influence that the lifecycle cost has on the total loss function so the GAN, 
apart from trying to imitate the training examples, it will also take into account the cost function,
which will be calculated following the VASSAR model.

'''

# Define the hyperparameters
num_epochs = 1000
batch_size = 64
learning_rate = 0.0002
beta1 = 0.5
lambda_s = 0.1

# Define the number of orbits in the constellation
num_orbits = 5

# Define the science benefit and lifecycle cost functions
def science_benefit(design):
    # Compute the science benefit for the given satellite design
    return np.sum(design)

def lifecycle_cost(design):
    # Compute the lifecycle cost for the given satellite design
    return np.sum(design)

# Define the generator network
def generator(z, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # Create a fully connected layer with 128 neurons and a ReLU activation function
        fc1 = tf.layers.dense(z, 128, activation=tf.nn.relu)
        
        # Create a fully connected layer with 256 neurons and a ReLU activation function
        fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.relu)
        
        # Create a fully connected layer with num_orbits neurons and a sigmoid activation function
        out = tf.layers.dense(fc2, num_orbits, activation=tf.nn.sigmoid)
        
        return out

# Define the discriminator network
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Create a fully connected layer with 128 neurons and a LeakyReLU activation function
        fc1 = tf.layers.dense(x, 128, activation=tf.nn.leaky_relu)
        
        # Create a fully connected layer with 256 neurons and a LeakyReLU activation function
        fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.leaky_relu)
        
        # Create a fully connected layer with 1 neuron and a sigmoid activation function
        out = tf.layers.dense(fc2, 1, activation=tf.nn.sigmoid)
        
        return out

# Define the placeholders for the input data
real_data = tf.placeholder(tf.float32, shape=[None, num_orbits])
z = tf.placeholder(tf.float32, shape=[None, 100])

# Generate fake data using the generator network
fake_data = generator(z)

# Compute the discriminator's output for the real and fake data
D_real = discriminator(real_data)
D_fake = discriminator(fake_data, reuse=True)

# Compute the generator and discriminator losses
gen_loss = -tf.reduce_mean(tf.log(D_fake) + lambda_s * science_benefit(fake_data))
disc_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))

# Define the optimizer for the generator and discriminator networks
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
gen_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(gen_loss, var_list=gen_vars)
disc_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(disc_loss, var_list=disc_vars)

# Initialize the variables
init = tf.global_variables_initializer()



# Define the discriminator's optimizer
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Define the generator's optimizer
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, science_benefit):
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    benefit_loss = tf.reduce_sum(science_benefit * fake_output) / tf.reduce_sum(fake_output)
    total_loss = fake_loss - lambda_s * benefit_loss
    return total_loss

# Define the training loop
@tf.function
def train_step(designs, science_benefit):
    # Generate noise
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    # Train the discriminator
    with tf.GradientTape() as d_tape:
        # Generate a batch of fake designs
        fake_designs = generator(noise, training=True)
        
        # Evaluate the discriminator on real and fake designs
        real_output = discriminator(designs, training=True)
        fake_output = discriminator(fake_designs, training=True)
        
        # Calculate the discriminator's loss
        d_loss = discriminator_loss(real_output, fake_output)
        
    # Calculate the gradients and apply them to the discriminator's parameters
    gradients_of_discriminator = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Train the generator
    with tf.GradientTape() as g_tape:
        # Generate a batch of fake designs
        fake_designs = generator(noise, training=True)
        
        # Evaluate the discriminator on fake designs
        fake_output = discriminator(fake_designs, training=True)
        
        # Calculate the generator's loss
        g_loss = generator_loss(fake_output, science_benefit)
        
    # Calculate the gradients and apply them to the generator's parameters
    gradients_of_generator = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

# Train the GAN
def train(dataset, epochs):
    for epoch in range(epochs):
        for designs, science_benefit in dataset:
            train_step(designs, science_benefit)
            
        # Print the progress
        if epoch % 10 == 0:
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            fake_designs = generator(noise, training=False)
            print(f"Epoch {epoch}, Benefit: {evaluate_designs(fake_designs)}")
