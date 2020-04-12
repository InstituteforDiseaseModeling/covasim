import pylab as pl
import sciris as sc
import covasim as cv


if __name__ == '__main__':

    sc.tic()

    people = cv.People(pop_size=20000)

    plist = people.to_people()

    ppl2 = cv.People()
    ppl2.from_people(plist)

    ppl3 = people + ppl2

    sim = cv.Sim()
    cv.make_people(sim)
    ppl4 = sim.people

    df = ppl4.to_df()
    arr = ppl4.to_arr()

    sc.toc()