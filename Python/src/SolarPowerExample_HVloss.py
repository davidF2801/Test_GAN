import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
import pandas as pd
import math
from pymoo.indicators.hv import HV

# Separate loss functions for generator and discriminator
import itertools




# Binary design variables example
class SolarPowerExample:
    def __init__(self):
        self.num_design_vars = 15
        self.num_examples = 3000
        self.latent_dim = 128
        self.batch_size = 1
        self.num_epochs = 100
        self.num_episodes = 7
        self.learning_rate = 0.0002
        self.beta1 = 0.1
        # Define problem parameters
        self.budget = 4000
        # self.component_costs = np.array([1, 2, 3, 4, 5])
        # self.component_power = np.array([3, 5, 2, 4, 1])
        self.component_costs = [200, 150, 300, 400, 250, 200, 350, 300, 250, 400,100,200,300,400, 150]
        self.component_power = [10, 12, 8, 9, 11, 13, 7, 6, 10, 12, 15, 7, 11, 14, 10]
        self.budget_penalty = 0.01
        self.power_penalty = 0.1
        self.budget_tolerance = 0.05
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.number_function_evaluations=0
        self.max_power = tf.reduce_sum(self.component_power)
        self.max_cost = tf.reduce_sum(self.component_costs)


        # Build the discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.discriminator_loss,
                                   optimizer=self.discriminator_optimizer,
                                   metrics=['accuracy'])

        # Build the generator model
        self.generator = self.build_generator()
        self.generator.compile(loss=self.generator_total_loss,
                               optimizer=self.generator_optimizer)

        # Build the GAN model
        self.gan = self.build_gan()
        self.gan.compile(loss=self.generator_total_loss, 
                         optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta1))





  



    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.latent_dim, activation='relu'))
        model.add(layers.Dense(64, input_dim=128, activation='relu'))
        model.add(layers.Dense(32, input_dim=64, activation='relu'))
        model.add(layers.Dense(16, input_dim=32, activation='relu'))
        model.add(layers.Dense(self.num_design_vars, activation='sigmoid'))
        return model

    #def build_discriminator(self):
    #    model = tf.keras.Sequential()
    #    model.add(layers.Dense(128, input_dim=self.num_design_vars, activation='relu'))
    #    model.add(layers.Dense(64, input_dim=128, activation='relu'))
    #    model.add(layers.Dense(32, input_dim=64, activation='relu'))
    #    model.add(layers.Dense(1, activation='sigmoid'))
    #    return model


    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.num_design_vars, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dropout(0.3))
        #model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation='relu'))
        #model.add(layers.Dropout(0.3))
        #model.add(layers.BatchNormalization())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def build_gan(self):
        # Freeze the discriminator's weights during GAN training
        self.discriminator.trainable = True
        # Build the GAN by stacking the generator and discriminator
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = Model(gan_input, gan_output)
        return gan

 

    def fitness(self,design):
        power = tf.reduce_sum(tf.multiply(design, self.component_power), axis=-1)
        cost = tf.reduce_sum(tf.multiply(design, self.component_costs), axis=-1)
        fitness = power/cost
        return power,cost,fitness



    def pareto_front_calculator(self, solutions, show, save = False):
        savings = []
        powers = []
        best_designs = []
        max_cost = tf.reduce_sum(self.component_costs)

        for sol in solutions:
            power, cost, fitness = self.fitness(sol)
            saving = float(max_cost) - float(cost)
            dominated = False
            for i, design in enumerate(best_designs):
                p_power, p_cost, _ = self.fitness(design)
                if (cost>= p_cost and power<=p_power) :
                    dominated = True
                    break

            if dominated==False:
                    for i, design in enumerate(best_designs):
                        p_power, p_cost, _ = self.fitness(design)
                        if cost<=p_cost and power>=p_power:
                            best_designs.pop(i)
                            savings.pop(i)
                            powers.pop(i)

                    best_designs.append(sol)
                    savings.append(saving)
                    powers.append(power)
                  
        if show:
            if save:
                savings = np.array(savings)
                powers = np.array(powers)
                np.save('optimal_pareto.npy', [savings, powers])


            data = np.load('optimal_pareto.npy')
            savings_opt = data[0]
            powers_opt = data[1]

            plt.figure('Pareto Front')
            plt.scatter(savings, powers,c='blue', label='Achieved Pareto')
            plt.scatter(savings_opt, powers_opt, c='red', label='Optimal Pareto')
            plt.xlabel('Savings (Max_cost - cost)')
            plt.ylabel('Generated Power')
            plt.legend()
            plt.show()


        return np.array(best_designs)



  
    #def generator_loss(self, fake_output):

    #   return tf.reduce_mean(tf.math.log(1 - fake_output))

    def generator_loss(self,fake_output):
        return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)


    def generator_total_loss(self, fake_output, objectives):

        HV = self.hypervolume_loss(objectives)
        return self.generator_loss(fake_output) + HV



    def hypervolume_loss(self, solution_set):
        
        hv = HV(ref_point=(1, 1))
        hv_value = hv.do(solution_set)
        return -hv_value




    def discriminator_loss(self, real_output, fake_output):
        d_loss_real = binary_crossentropy(tf.ones_like(real_output), real_output)
        d_loss_fake = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        d_loss = (d_loss_real + d_loss_fake) / 2
        return d_loss


    def generate_real_samples(self,num_samples):
        X = np.zeros((num_samples, self.num_design_vars))
        for i in range(num_samples):
            # Generate random binary array
            design = np.random.randint(2, size=self.num_design_vars)
            cost = np.dot(design, self.component_costs)
            # Ensure the cost constraint is satisfied
            while cost > self.budget or np.any(np.all(X == design, axis=1)) :
                design = np.random.randint(2, size=self.num_design_vars)
                cost = np.dot(design, self.component_costs)
            X[i] = design
        return X
    


    # Define the loss function as the negative hypervolume

    
    def binarize(continuous_samples):
        fake_samples_thresh = tf.where(continuous_samples >= 0.5, tf.ones_like(continuous_samples), tf.zeros_like(continuous_samples))

    def percentage_budget_fulfilled(self, design_space):
        # Count the number of designs that fulfill the budget constraint
        count = 0
        for design in design_space:
            cost = np.dot(design, self.component_costs)
            if cost <= self.budget:
                count += 1
        # Calculate the percentage of designs that fulfill the budget constraint
        percentage = count / len(design_space) * 100
        return percentage

    
    def generate_fake_samples(self, training):
        noise = tf.random.normal([8, self.latent_dim])
        fake_samples = self.generator(noise, training=training)
        fake_samples_thresh = tf.where(fake_samples >= 0.5, tf.ones_like(fake_samples), tf.zeros_like(fake_samples))
      
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
        d_losses = []
        g_losses = []
        per_constraint = []
        power = []
        costs = []
        data = self.generate_real_samples(self.num_examples)
        #real_data = dataArray
        





        for nep in range(self.num_episodes):

            print(f"Episode {nep}")
            real_data = self.pareto_front_calculator(data,nep==0)
            real_data_sliced = self.create_batches(real_data)
            data = real_data
        
            for epoch in range(self.num_epochs):
                g_losses_batch = []
                d_losses_batch = []
                per_constraint_batch = []
                power_batch = []
                costs_batch = []


                for batch in real_data_sliced:
            
                    with tf.GradientTape(persistent=True) as dtape:
                        #generated_samples = self.generator(noise, training=False)
                        generated_samples, generated_samples_thresh=self.generate_fake_samples(training=False)
                        real_output = self.discriminator(batch, training=True)
                        fake_output = self.discriminator(generated_samples_thresh, training=True)
                        d_loss = self.discriminator_loss(real_output, fake_output)
                        grads = dtape.gradient(d_loss, self.discriminator.trainable_weights)
                        if len(grads) == 0:
                            grads = [tf.random.normal(w.shape) for w in self.generator.trainable_weights]
                        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
                
                    # Train the generator
                    with tf.GradientTape() as gtape:
                        #generated_samples = self.generator(noise, training=True)
                        generated_samples, generated_samples_thresh=self.generate_fake_samples(training=True)
                        #generated_samples_thresh=tf.constant(generated_samples_thresh)
                        #gst = np.expand_dims(generated_samples_thresh,axis=0)
                        
                        power_generated,cost,_ = self.fitness(generated_samples_thresh)
                        
                        no_power = []
                        normalized_cost =[]
                        for p in power_generated:
                            nop = (float(self.max_power)-p )/float(self.max_power)
                            no_power.append(nop)

                        for c in cost:
                            nc = c/float(self.max_cost)
                            normalized_cost.append(nc)

                            


                        self.number_function_evaluations+=1
                        #data = np.concatenate(data,generated_samples_thresh)
                        objectives = np.array((no_power,normalized_cost ))
                        objectives = np.transpose(objectives)
                        fake_output = self.discriminator(generated_samples, training=False)
                        generator_loss = self.generator_total_loss(fake_output,objectives)
                        grads = gtape.gradient(generator_loss, self.generator.trainable_weights)
                        if len(grads) == 0 or grads[0] is None:
                            grads = [tf.random.normal(w.shape) for w in self.generator.trainable_weights]

                        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
                        data = np.vstack((data,generated_samples_thresh))
                        


                    per_constraint_batch.append(self.percentage_budget_fulfilled(generated_samples_thresh))
                    g_losses_batch.append(tf.reduce_mean(generator_loss))
                    #obj_losses_batch.append(obj_loss)
                    #gan_losses_batch.append(g_loss_obj)
                    d_losses_batch.append(tf.reduce_mean(d_loss))
                    power_batch.append(power_generated)
                    costs_batch.append(cost)

                per_constraint_m = tf.reduce_mean(per_constraint_batch)
                d_losses_m = tf.reduce_mean(d_losses_batch)
                g_losses_m = tf.reduce_mean(g_losses_batch)
                #obj_losses_m = tf.reduce_mean(obj_losses_batch)
                #gan_losses_m = tf.reduce_mean(gan_losses_batch)
                power_m = tf.reduce_mean(power_batch)
                cost_m = tf.reduce_mean(costs_batch)



                costs.append(cost_m)
                g_losses.append(g_losses_m)
                d_losses.append(d_losses_m)
                per_constraint.append(per_constraint_m)
                power.append(power_m)

                print(f"Epoch: {epoch + 1}/{self.num_epochs}, Discriminator Loss: {d_losses_m}, Generator Loss: {g_losses_m}, Percentage fulfilling = {per_constraint_m}, Generated power =  {power_m}, Cost = {cost_m}")
                if epoch==self.num_epochs-1:
                    print(generated_samples_thresh)




        print(self.number_function_evaluations)



        self.pareto_front_calculator(data,True)
        plt.figure('G losses')
        plt.plot(g_losses, label='Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\Results\Pareto_Loss_Solar_Power\gl.png')
        plt.show()

        plt.figure('Costs')
        plt.plot(costs, label='Costs')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\Results\Pareto_Loss_Solar_Power\cost.png')
        plt.show()

        plt.figure('Generated Power')
        plt.plot(power, label='Generated Power')
        plt.xlabel('Epochs')
        plt.ylabel('Power (Watts)')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\Results\Pareto_Loss_Solar_Power\pow.png')
        plt.show()

        plt.figure('Percentage constraint')
        plt.plot(per_constraint, label='Percentage of designs that fulfill the constraint')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\Results\Pareto_Loss_Solar_Power\per.png')
        plt.show()
        


        plt.figure('D losses')
        plt.plot(d_losses, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\Results\Pareto_Loss_Solar_Power\dl.png')
        plt.show()
       


      #print(f"Epoch: {epoch + 1}/{self.num_epochs}, Discriminator Loss: {d_loss}, Discriminator Accuracy: {discriminator_accuracy}, Generator Loss: {g_loss}, Generator Accuracy: {generator_accuracy}")

      



                




GAN = SolarPowerExample()
design_space = list(itertools.product([0, 1], repeat=15))
# dataArray = np.array(design_space)
np.savetxt('design_space.csv', design_space, delimiter=',')
# Load the design space from the file
#design_space = np.loadtxt('design_space.csv', delimiter=',')
percentage = GAN.percentage_budget_fulfilled(design_space)

print(percentage)
# Save the design space to a file

#optimal_pareto = GAN.pareto_front_calculator(design_space,True)

GAN.train()

GAN.generator.save('generator_model_HV1.h5')



## Load the saved generator model
#generator = load_model('generator_model_1.h5', custom_objects={'generator_total_loss': GAN.generator_total_loss, 'discriminator_loss':GAN.discriminator_loss, 'genarator_loss':GAN.generator_loss})
generator = load_model('generator_model_3.h5', custom_objects={'generator_total_loss': GAN.generator_total_loss, 'discriminator_loss':GAN.discriminator_loss, 'genarator_loss':GAN.generator_loss})
# Generate designs
num_designs = input('Number of designs: ')


for i in range(int(num_designs)):
    noise = tf.random.normal([1, GAN.latent_dim])
    designs = generator.predict(noise)
    binary_designs = np.round(designs)
    print(binary_designs)
    costs = np.sum(binary_designs * GAN.component_costs, axis=1)
    generated_power = np.sum(binary_designs * GAN.component_power, axis=1)
    print(costs)
    print(generated_power)




# create an empty dataframe to store the data
df = pd.DataFrame(columns=['instrument_{}'.format(i+1) for i in range(15)] + ['cost', 'generated_power'])

# loop through the designs and append to the dataframe
for i in range(int(num_designs)):
    noise = tf.random.normal([1, GAN.latent_dim])
    designs = generator.predict(noise)
    binary_designs = np.round(designs)
    print(binary_designs)
    costs = np.sum(binary_designs * GAN.component_costs, axis=1)
    generated_power = np.sum(binary_designs * GAN.component_power, axis=1)
    print(costs)
    print(generated_power)
    
    # create a new row of data and append to the dataframe
    new_row = np.hstack((binary_designs, np.array([costs, generated_power]).T))
    df = df.append(pd.DataFrame(new_row, columns=['instrument_{}'.format(i+1) for i in range(15)] + ['cost', 'generated_power']), ignore_index=True)
    #df = df.append(new_row, ignore_index=True)

# save the dataframe to csv file
df.to_csv('generated_designs_1.csv', index=False)



