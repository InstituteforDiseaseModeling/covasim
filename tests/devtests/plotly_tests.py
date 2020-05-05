import plotly.io as pio
import covasim as cv


pio.renderers.default = "browser"

sim = cv.Sim(pop_size=10e3)

sim.run()
fig = cv.animate_people(sim, do_show=True)
