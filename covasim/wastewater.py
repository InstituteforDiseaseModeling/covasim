'''
Wastewater sampling analyzer for Covasim.

WastewaterSampler is a cv.Analyzer that, at specified time points, snapshots
the set of circulating viral haplotypes and their relative proportions weighted
by individual viral shedding — i.e. what a wastewater sample would look like.
Haplotypes are stored as frozensets of (site_0indexed, ref_nt_int, alt_nt_int)
mutation tuples relative to the simulation reference sequence.
Snapshots can be exported as FASTA or fed directly to Bygul's
simulate_proportions to generate synthetic sequencing reads.

Requires pars['evo_pars']['enable'] = True.
'''

import dataclasses
from collections import defaultdict

import numpy as np

from .analysis import Analyzer
from .sequence_evolution import agent_mutations as _agent_mutations
from .sequence_evolution import decode_mutations as _decode_mutations

__all__ = ['WastewaterSampler', 'WastewaterSample']


@dataclasses.dataclass
class WastewaterSample:
    '''Snapshot of circulating haplotypes and their viral-shedding-weighted proportions.'''
    day:              int
    date:             str
    mutation_sets:    list  # deduplicated frozensets of (site_0idx, ref_nt_int, alt_nt_int), one per distinct haplotype
    variant_labels:   list  # variant label string for each distinct haplotype
    proportions:      list  # normalized viral-shedding fractions, sums to 1.0
    raw_loads:        list  # un-normalized total viral shedding per haplotype
    n_infectious:     int   # number of currently infectious agents
    n_genotypes:      int   # number of distinct haplotypes present
    variant_shedding: dict = None  # {variant_label: total_shedding} aggregated by named variant


class WastewaterSampler(Analyzer):
    '''
    Analyzer that snapshots the viral haplotype mixture present in a wastewater
    sample at one or more time points.

    For each sampled day:
      1. Identifies all currently infectious agents.
      2. Reads each agent's viral shedding from ``sim.people.viral_shedding``
         (``viral_load × rel_trans``), which is updated by ``sim.step()`` before
         analyzers are called.
      3. Retrieves each agent's mutation frozenset from LineageSequenceTracker.
      4. Groups identical haplotypes (by frozenset identity) and sums their
         viral shedding contributions.
      5. Normalizes to proportions.

    Snapshots are stored in self.samples (dict: int day → WastewaterSample).
    Use to_fasta(day) to get a multi-FASTA string, or simulate_sample(day, ...)
    to pass the mixture directly to Bygul.

    Requires pars['evo_pars']['enable'] = True.

    Args:
        days  (list): simulation days (int) or calendar date strings to sample.
        label (str):  optional label for the analyzer.
        sid   (int):  optional sewershed ID; None samples everyone (legacy behavior),
                      -1 samples people outside all supplied polygons. Use one analyzer per sid.

    Example::

        ww = cv.WastewaterSampler(days=[56, 120])
        sim = cv.Sim(pars, analyzers=[ww])
        sim.run()
        print(ww.to_fasta(56))
    '''

    def __init__(self, days, label=None, sid=None):
        super().__init__(label=label)
        if sid is not None and (isinstance(sid, (bool, np.bool_)) or not isinstance(sid, (int, np.integer)) or sid < -1):
            raise ValueError('sid must be an integer >= -1, or None for the entire population')
        self.sid = sid
        self._days_input = days  # raw user input; converted in initialize()
        self.samples     = {}
        self._reference  = None  # set in initialize()

    def initialize(self, sim):
        super().initialize(sim)
        if not sim['evo_pars'].get('enable', False):
            raise ValueError(
                "WastewaterSampler requires pars['evo_pars']['enable'] = True"
            )
        self._reference = sim.people.sequence_tracker.reference
        # Convert string dates to integer days
        self._day_set = set()
        for d in self._days_input:
            self._day_set.add(sim.day(d) if isinstance(d, str) else int(d))

    def apply(self, sim):
        if sim.t not in self._day_set:
            return
        self._take_sample(sim)

    def _take_sample(self, sim):
        # Infectious agents only
        eligible = sim.people.infectious
        if self.sid is not None:
            eligible = eligible & (sim.people.sewershed == self.sid)
        inds = np.nonzero(eligible)[0]

        if len(inds) == 0:
            self.samples[sim.t] = None
            return

        loads       = sim.people.viral_shedding[inds]
        tracker     = sim.people.sequence_tracker
        ref         = self._reference
        variant_map = sim.pars.get('variant_map', {})

        load_by_muts  = defaultdict(float)
        label_by_muts = {}   # first-seen variant label per distinct haplotype
        for idx, load in zip(inds, loads):
            i    = int(idx)
            muts = _agent_mutations(tracker, i, ref)
            load_by_muts[muts] += float(load)
            if muts not in label_by_muts:
                v = sim.people.infectious_variant[i]
                label_by_muts[muts] = (
                    variant_map.get(int(v), 'wild') if not np.isnan(v) else 'wild'
                )

        total         = sum(load_by_muts.values())
        if total <= 0:
            self.samples[sim.t] = None
            return
        mutation_sets = list(load_by_muts.keys())
        raw_loads     = [load_by_muts[m] for m in mutation_sets]
        proportions   = [v / total for v in raw_loads]
        proportions[-1] = 1.0 - sum(proportions[:-1])

        # Aggregate shedding by named variant (mirrors old group_by='variant' logic)
        variant_indices = sim.people.infectious_variant[inds]
        load_by_variant = defaultdict(float)
        for vi, load in zip(variant_indices, loads):
            vname = variant_map.get(int(vi), f'variant_{int(vi)}') if not np.isnan(vi) else 'wild'
            load_by_variant[vname] += float(load)

        self.samples[sim.t] = WastewaterSample(
            day              = sim.t,
            date             = sim.date(sim.t),
            mutation_sets    = mutation_sets,
            variant_labels   = [label_by_muts[m] for m in mutation_sets],
            proportions      = proportions,
            raw_loads        = raw_loads,
            n_infectious     = int(len(inds)),
            n_genotypes      = len(mutation_sets),
            variant_shedding = dict(load_by_variant),
        )

    def to_fasta(self, day):
        '''Return a multi-FASTA string for the haplotype mixture on the given day.'''
        sample = self.samples[day]
        if sample is None:
            return ''
        ref   = self._reference
        lines = []
        for i, (muts, prop, vlabel) in enumerate(zip(sample.mutation_sets, sample.proportions, sample.variant_labels)):
            lines.append(f'>genotype_{i}  proportion={prop:.8f}  day={day}  date={sample.date}  variant={vlabel}')
            lines.append(_decode_mutations(muts, ref))
        return '\n'.join(lines)

    def simulate_sample(self, day, primers, reference, outdir='./reads/', readcnt=500, redo=False, **bygul_kwargs):
            '''
            Write a multi-FASTA file and a proportions CSV, then invoke Bygul's simulate_proportions CLI.

            Bygul's simulate_proportions command is a Click CLI that expects genome
            sequences inside a multi-FASTA file and their relative abundances in a CSV.
            This method writes temporary files, builds the required CLI argument list, 
            and calls the command via Click's standalone_mode=False interface.

            Args:
                day       (int):  simulation day to sample (must be in self.samples)
                primers   (str):  path to primer BED file (required by Bygul)
                reference (str):  path to reference FASTA file (required by Bygul)
                outdir    (str):  output directory for Bygul reads (default '.') (NOTE: If redo=True, Bygul will delete all contents of outdir)
                readcnt   (int):  number of reads per amplicon (default 500)
                redo      (bool): re-run even if output files already exist (default False)
                **bygul_kwargs:   additional CLI options forwarded as --key value pairs
                                (e.g. wgsim_read_length=150, wgsim_error_rate=0.0001,
                                simulation_mode='amplicon', seed=42)

            Requires the `bygul` package to be installed.
            '''
            import os
            import tempfile
            import csv

            try:
                from bygul._cli import simulate_proportions
            except ImportError as e:
                raise ImportError(
                    "The 'bygul' package is required for WastewaterSampler.simulate(). "
                    "Install it from the Andersen Lab: https://github.com/andersen-lab/Bygul"
                ) from e

            sample = self.samples[day]
            if sample is None:
                raise ValueError(f"No infectious agents were present on day {day}; cannot simulate.")

            with tempfile.TemporaryDirectory() as tmpdir:
                multifasta_path = os.path.join(tmpdir, f'wastewater_day{day}.fasta')
                csv_path = os.path.join(tmpdir, 'proportions.csv')

                ref = self._reference

                # Build proportions list that sums to exactly 1.0 after float parsing.
                # Adjust the last value to absorb any rounding error.
                props = sample.proportions
                rounded = [round(p, 10) for p in props[:-1]]
                rounded.append(round(1.0 - sum(rounded), 10))

                # Write both the multi-FASTA and the proportions CSV
                with open(multifasta_path, 'w') as f_fasta, open(csv_path, 'w', newline='') as f_csv:
                    csv_writer = csv.writer(f_csv)
                    csv_writer.writerow(['sample_name', 'proportion'])

                    for i, muts in enumerate(sample.mutation_sets):
                        seq_name = f'genotype_{i}'
                        seq = _decode_mutations(muts, ref)
                        
                        # Append sequence to the multi-FASTA
                        f_fasta.write(f'>{seq_name}\n{seq}\n')
                        
                        # Write row to the CSV mapping name to proportion
                        csv_writer.writerow([seq_name, rounded[i]])

                # Build Click CLI args list using the new --multifasta and --csv parameters
                args = [
                    '--multifasta', multifasta_path,
                    '--csv', csv_path,
                    '--primers', primers,
                    '--reference', reference,
                    '--outdir', outdir,
                    '--readcnt', str(readcnt),
                ]

                for key, val in bygul_kwargs.items():
                    args.extend([f'--{key}', str(val)])

                if redo:
                    args.extend(['--redo'])

                # standalone_mode=False returns the result instead of calling sys.exit
                simulate_proportions.main(args=args, standalone_mode=False)