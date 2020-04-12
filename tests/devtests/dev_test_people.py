import sciris as sc
import covasim as cv


if __name__ == '__main__':

    sc.tic()

    people = cv.People(pop_size=20000)

    sc.toc(label='default')

    plist = people.to_people()

    sc.toc(label='to people')

    ppl2 = cv.People()
    ppl2.from_people(plist)

    sc.toc(label='from people')

    ppl3 = people + ppl2

    sim = cv.Sim(pop_type='synthpops')
    cv.make_people(sim)
    ppl4 = sim.people

    sc.toc(label='as sim')

    df = ppl4.to_df()
    arr = ppl4.to_arr()

    sc.toc(label='to df/arr')

    sc.toc()