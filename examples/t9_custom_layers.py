'''
Demonstration of custom population layers
'''

import numpy as np
import covasim as cv

# Create the first sim
orig_sim = cv.Sim(pop_type='hybrid', label='Default hybrid population')
orig_sim.initialize() # Initialize the population

# Create the second sim
sim = orig_sim.copy()

# Define the new layer, 'q'
n_people = len(sim.people)
n_contacts_per_person = 0.5
n_contacts = int(n_contacts_per_person*n_people)
contacts_p1 = cv.choose(max_n=n_people, n=n_contacts)
contacts_p2 = cv.choose(max_n=n_people, n=n_contacts)
beta = np.ones(n_people)
layer = cv.Layer(p1=contacts_p1, p2=contacts_p2, beta=beta)

# Add this layer in and re-initialize the sim
sim.people.contacts['q'] = layer
sim.reset_layer_pars() # Automatically add layer 'q' to the parameters using default values
sim.initialize() # Reinitialize
sim.label = f'Extra contact layer with {n_contacts_per_person} contacts per person'

orig_sim.run()
sim.run()

# Run and compare
# msim = cv.MultiSim([orig_sim, sim])
# msim.run()
# msim.plot()