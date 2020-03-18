import sciris as sc
import pandas as pd

folder = '/home/cliffk/idm/covid_abm/covid_seattle/results_2020mar10/'
fn = folder + 'seattle-projection-results_v4e.obj'
data = sc.loadobj(fn)

for key,arrdict in data.items():
    csv_fn = folder + f'seattle-projection-results_v4e_{key}.csv'
    best = arrdict['best']
    df = pd.DataFrame(best)
    df.to_csv(csv_fn)

print('Done')