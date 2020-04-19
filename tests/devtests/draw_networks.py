"""
Show contacts within different layers
"""

import networkx as nx
import covasim as cv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)

pars = {
    'pop_size': 1e2, # start with a small pool
    'pop_type': 'hybrid', # synthpops, hybrid
    'pop_infected': 0, # Infect none for starters
    'n_days': 100, # 40d is long enough for everything to play out
    'contacts': {'h': 4.0, 's': 10, 'w': 10, 'c': 10},
    'beta_layer': {'h': 1, 's': 1, 'w': 1, 'c': 1},
    'quar_eff': {'h': 1, 's': 1, 'w': 1, 'c': 1},
}

# Create sim
sim = cv.Sim(pars=pars)
sim.initialize()

fig = plt.figure(figsize=(16,16))
mapping = dict(h='Households', s='Schools', w='Work', c='Community')
for i, layer in enumerate(['h', 's', 'w', 'c']):
    ax = plt.subplot(2,2,i+1)
    hdf = sim.people.contacts[layer].to_df()

    G = nx.Graph()
    G.add_nodes_from(set(list(hdf['p1'].unique()) + list(hdf['p2'].unique())))
    f = hdf['p1']
    t = hdf['p2']
    G.add_edges_from(zip(f,t))
    print('Nodes:', G.number_of_nodes())
    print('Edges:', G.number_of_edges())

    nx.draw(G, ax=ax, node_size=10, width=0.1)
    ax.set_title(mapping[layer])
plt.show()
