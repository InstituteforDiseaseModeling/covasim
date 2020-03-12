import sciris as sc
import pylab as pl

doplot = True
folder = '/home/cliffk/idm/covid_abm/covid_seattle/results_2020mar12/'
fn = folder + 'seattle-capacity_2020mar12_v0.obj'
fig_fn =  folder + 'seattle-capacity_2020mar12_v0.png'
scale = 100 # from parameters.py

data = sc.loadobj(fn)

reskey = 'cum_exposed'
for key,valdict in data.items():
    arr = valdict[reskey]*scale
    pl.savetxt(folder + f'seattle-capacity_2020mar12_{key}.csv', arr, fmt='%0.0f', delimiter=',')


if doplot:
    fig = pl.figure(figsize=(25,15))

    for k,key in enumerate(data.keys()):
        pl.subplot(2,2,k+1)
        arr = data[key][reskey]*scale
        for i in range(arr.shape[1]):
            pl.plot(arr[:,i])
        if key == 'Baseline':
            pl.title(key)
        else:
            pl.title(f'{key}% reduction in transmission')
        pl.xlabel('Day from March 12th')
        pl.ylabel('Cumulative infections')
        sc.commaticks()
        pl.ylim([0,1.5e6])

    pl.savefig(fig_fn)
    pl.show()

print('Done')