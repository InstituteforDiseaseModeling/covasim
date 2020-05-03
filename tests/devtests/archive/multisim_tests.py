import covasim as cv

sim = cv.Sim() # Create the sim
msim = cv.MultiSim(sim, n_runs=5) # Create the multisim
msim.run() # Run them in parallel
msim.combine() # Combine into one sim
msim.plot() # Plot results

sim = cv.Sim() # Create the sim
msim = cv.MultiSim(sim, n_runs=11, noise=0.1) # Set up a multisim with noise
msim.run() # Run
msim.reduce() # Compute statistics
msim.plot() # Plot

sims = [cv.Sim(beta=0.015*(1+0.02*i)) for i in range(5)] # Create sims
for sim in sims: sim.run() # Run sims in serial
msim = cv.MultiSim(sims) # Convert to multisim
msim.plot() # Plot as single sim