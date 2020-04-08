#!/bin/bash

set -e
cd $(dirname "$0")

CMD='python sim.py'
echo "running ${CMD}"; eval $CMD

CMD='python sim.py --interv=True'
echo "running ${CMD}"; eval $CMD

CMD='python sim.py --pars "{pop_size:20000, pop_infected:1, n_days:360, rand_seed:1}"'
echo "running ${CMD}"; eval $CMD

CMD='python sim.py --interv=True --do_save=True'
echo "running ${CMD}"; eval $CMD

CMD='python scenarios.py'
echo "running ${CMD}"; eval $CMD

CMD="python scenarios.py --basepars "{pop_size: 5000}" --metapars \"{n_runs:3, noise:0.1, noisepar:'beta', rand_seed:1, quantiles:{'low':0.1, 'high':0.9}}\""
echo "running ${CMD}"; eval $CMD

CMD='python scenarios.py --interv_day=35 --interv_eff=0.7'
echo "running ${CMD}"; eval $CMD

CMD='python scenarios.py -do_save=True'
echo "running ${CMD}"; eval $CMD