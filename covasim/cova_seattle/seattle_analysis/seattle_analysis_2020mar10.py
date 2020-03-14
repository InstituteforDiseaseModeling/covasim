import sciris as sc

fn = 'seattle-projection-results_v4e.obj'

reskeys = ['cum_exposed', 
           'n_exposed']

final = sc.loadobj(fn)

for k in list(final.keys()):
    for key in reskeys:
        print(f'{k} {key}: {final[k].best[key][-1]:0.0f}')
        

print('Done.')
