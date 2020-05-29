import sciris as sc
import covasim as cv

s = cv.Sim(label='sim1', verbose=0)
s.run()

sc.heading('print')
print(s)

sc.heading('summarize')
s.summarize()

sc.heading('brief')
s.brief()

sc.heading('multisim')
msim = cv.MultiSim(s, verbose=0)
msim.run(reduce=True)
msim.summarize()