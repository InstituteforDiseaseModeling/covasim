import pylab as pl


#%% Load data




#%% Plotting
n = 84
arr = pl.arange(n)/n

x = pl.sin(arr*2*pl.pi)
y = pl.cos(arr*2*pl.pi)

pl.scatter(x,y)