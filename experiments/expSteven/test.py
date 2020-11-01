import covasim as cv
import numpy as np


if __name__ == '__main__':

    sim = cv.Sim()
    sim.initialize()

    contacts = sim.people.contacts['a'].find_contacts(np.int64(0))

    for ind in contacts:
        print(sim.people.contacts['a'].get(inds))
