import covasim.cova_cdc as cova

version  = 'v5'
date     = '2020mar15'
folder   = f'results_{date}'
basename = f'{folder}/cdc-projections_{date}_{version}'
fn_obj   = f'{basename}.obj'
fn_fig   = f'{basename}_hs.png'

healthsystems = cova.HealthSystem(filename=fn_obj)