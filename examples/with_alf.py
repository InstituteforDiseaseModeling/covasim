import covasim as cv
import sciris as sc
import numpy as np
sim = cv.Sim(pop_type='hybrid')
sim.initialize()

# Define ALF parameters
# According to LTC 2015-2016 data for all of Washington:
# 83% of residents in nursing homes are 65> and 95% of residents in residential care communities are 65>.
# Therefore we set the minimum age to 70.
# The average size of a nursing home is 76 and average size of a residential care community is 18.
# There are 200 nursing homes in all of Washington and 2,000 residential care communities, so we
# set the average size of an ALF to 25.
# There are approximately 51,200 people residing in ALFs in Washington. We compared this to the number of
# people living above 60 in Washington State (132,021) to conclude that 38% of these people are living in ALFs.
# In the 2,200 combined facilities, there are 23,569 nursing and social work staff employees.
# Therefore, we assume each facility has on average 10 health care staff.

min_alf_age = 70
n_alf_contacts = 25
alf_prob = 0.38
n_alf_staff = 10

# Set alf parameters to look like household
for key in ['contacts', 'dynam_layer', 'beta_layer', 'quar_eff']:
    sim[key]['alf'] = sim[key]['h']

pop_size = sim.pars['pop_size']

# Create empty contact list for alf residents
# Create the contacts
layer_keys = ['alf']
alf_inds = cv.binomial_filter(alf_prob, sc.findinds(sim.people.age >= min_alf_age))
contacts_list = [{key:[] for key in layer_keys} for i in range(pop_size)]
alf_contacts, _ = cv.make_microstructured_contacts(len(alf_inds), {'alf':n_alf_contacts}, sim)

# get the dictionary of ALFs
alf_dict = sim.contactdict['alf']

for i, ind in enumerate(alf_inds):
    contacts_list[ind]['alf'] = alf_inds[alf_contacts[i]['alf']] # Copy over alf contacts

    # find the cluster that individual is in and replace with correct index
i = 0
for id, _ in alf_dict.items():
    alf_dict[id] = list(alf_inds[alf_contacts[i]['alf']])
    i += 1

# Create list of health care workers
health_care_worker_age = [22, 55]
health_care_worker_inds = sc.findinds((sim.people.age >= health_care_worker_age[0]) * (sim.people.age < health_care_worker_age[1]))

# Loop through all ALFs, select a number of health care workers from health_care_worker_inds, add them to ALF and
# to list of contacts, then remove them from health_care_worker_inds

for id, cluster_members in alf_dict.items():
    num_health_care_workers = cv.poisson(n_alf_staff)
    health_care_workers_to_add = np.random.choice(health_care_worker_inds, n_alf_staff, replace=False)
    for person in np.nditer(health_care_workers_to_add):
        contacts_list[person]['alf'] = contacts_list[person]['alf'] + cluster_members
        cluster_members.append(person)
        for alf_member in cluster_members:
            contacts_list[alf_member]['alf'] = contacts_list[alf_member]['alf'] + person
    for person in range(len(health_care_workers_to_add)):
        health_care_worker_inds = health_care_worker_inds[health_care_worker_inds != health_care_workers_to_add[person]]

sim.contactdict['alf'] = alf_dict
sim.people.add_contacts(contacts_list, lkey='alf')
print('Total alf contacts: ', len(sim.people.contacts['alf']))

# Run
sim.run()
sim.plot()