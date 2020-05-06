"""
Show contacts within different layers
"""

import covasim as cv
import networkx as nx
import pylab as pl
import numpy as np
import seaborn as sns
import sciris as sc
sns.set(font_scale=2)

pop_size = 200
pop_type = 'synthpops'

contacts = dict(
    random = {'a':20},
    hybrid = {'h': 2.0, 's': 4, 'w': 6, 'c': 10},
    synthpops = {'h': 2.0, 's': 4, 'w': 6, 'c': 10},
    )

pars = {
    'pop_size': pop_size, # start with a small pool
    'pop_type': pop_type, # synthpops, hybrid
    'contacts': contacts[pop_type],
}

# Create sim
sim = cv.Sim(pars=pars)
sim.initialize()

fig = pl.figure(figsize=(16,16), dpi=120)
mapping = dict(a='All', h='Households', s='Schools', w='Work', c='Community')
colors = sc.vectocolor(sim.people.age, cmap='turbo')

keys = list(contacts[pop_type].keys())
nrowcol = np.ceil(np.sqrt(len(keys)))

for i, layer in enumerate(keys):
    ax = pl.subplot(nrowcol,nrowcol,i+1)
    hdf = sim.people.contacts[layer].to_df()

    inds = list(set(list(hdf['p1'].unique()) + list(hdf['p2'].unique())))
    color = colors[inds]

    G = nx.DiGraph()
    G.add_nodes_from(inds)
    f = hdf['p1']
    t = hdf['p2']
    G.add_edges_from(zip(f,t))
    print(f'Layer: {layer}, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}')

    pos = nx.spring_layout(G, k=0.2)
    nx.draw(G, pos=pos, ax=ax, node_size=40, width=0.1, arrows=True, node_color=color)
    ax.set_title(mapping[layer])
pl.show()
