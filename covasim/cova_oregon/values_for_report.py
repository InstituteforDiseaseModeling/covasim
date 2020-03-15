'''
Write the paragraph for the report.
'''

import sciris as sc

version = 'v3'
folder = 'results_2020mar15'
fn_obj = f'{folder}/oregon-projection-2020mar14_{version}.obj'

data = sc.loadobj(fn_obj)

def f(scen, lim, reskey='cum_exposed'):
    val = data[scen][lim][reskey][-1]
    output = sc.sigfig(val, sigfigs=2)
    return output

def g(scen, reskey='n_exposed'):
    outputs = {}
    for lim in ['best', 'low', 'high']:
        val = data[scen][lim][reskey][-1]
        outputs[lim] = sc.sigfig(val, sigfigs=2)
    output = f'{outputs["best"]} [{outputs["low"]}, {outputs["high"]}]'
    return output


string = f'''In the absence of interventions, we estimate that by April 11th, there will be
{f('baseline', 'best')}
(80% confidence interval:
[{f('baseline', 'low')}, {f('baseline', 'high')}])
cumulative infections. With proposed interventions beginning March 16th, and schools reopening on March 31st,
we estimate
{f('reopen', 'best')} [{f('reopen', 'low')}, {f('reopen', 'high')}]
infections. If schools instead remain closed, we estimate
{f('closed', 'best')} [{f('closed', 'low')}, {f('closed', 'high')}]
infections. If aggressive interventions are implemented instead, we estimate
{f('aggressive', 'best')} [{f('aggressive', 'low')}, {f('aggressive', 'high')}]
infections. Critically, the number of active infections by April 11th in the business-as-usual,
schools-reopen, schools-remain-closed, and aggressive-intervention scenarios are
{g('baseline')},
{g('reopen')},
{g('closed')}, and
{g('aggressive')}.
'''

string = string.replace('\n',' ')

print(string)