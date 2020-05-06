"""
Show contacts within different layers
"""

import covasim as cv
import networkx as nx
import pylab as pl
import seaborn as sns
import sciris as sc
sns.set(font_scale=2)

pars = {
    'pop_size': 200, # start with a small pool
    'pop_type': 'hybrid', # synthpops, hybrid
    'pop_infected': 0, # Infect none for starters
    'n_days': 100, # 40d is long enough for everything to play out
    'contacts': {'h': 2.0, 's': 4, 'w': 6, 'c': 10},
    'beta_layer': {'h': 1, 's': 1, 'w': 1, 'c': 1},
    'quar_eff': {'h': 1, 's': 1, 'w': 1, 'c': 1},
}

# Create sim
sim = cv.Sim(pars=pars)
sim.initialize()

fig = pl.figure(figsize=(16,16), dpi=120)
mapping = dict(h='Households', s='Schools', w='Work', c='Community')
colors = sc.vectocolor(sim.people.age, cmap='turbo')

for i, layer in enumerate(['h', 's', 'w', 'c']):
    ax = pl.subplot(2,2,i+1)
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
