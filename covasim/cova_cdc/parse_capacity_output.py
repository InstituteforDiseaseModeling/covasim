import sciris as sc
import pylab as pl

doplot   = 1
version  = 'v5'
date     = '2020mar15'
folder   = f'results_{date}'
basename = f'{folder}/cdc-projections_{date}_{version}'
fn_obj   = f'{basename}.obj'
fn_fig   = f'{basename}.png'

typekeys = ['best','low', 'high']

data = sc.loadobj(fn_obj)

reskey = 'cum_exposed'
for typekey in typekeys:
    for key,valdict in data.items():
        arr = valdict[typekey][reskey]
        pl.savetxt(f'{basename}_{key}_{typekey}.csv', arr, fmt='%0.0f', delimiter=',')


if doplot:
    fig = pl.figure(figsize=(25,15))

    for k,key in enumerate(data.keys()):
        pl.subplot(2,3,k+1)
        for typekey in typekeys:
            arr = data[key][typekey][reskey]
            pl.plot(arr)
        # for i in range(arr.shape[1]):
            # pl.plot(arr[:,i])
        pl.title(key)
        # else:
        #     pl.title(f'{key}% reduction in transmission')
        pl.xlabel('Day from Feb. 21')
        pl.ylabel('Cumulative infections')
        sc.commaticks()
        # pl.ylim([0,1.5e6])

    pl.savefig(fn_fig)
    pl.show()

print('Done')