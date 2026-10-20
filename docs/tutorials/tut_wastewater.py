"""
Covasim for wastewater: step by step tutorial (script version of tut_wastewater.ipynb).

Runs from any directory; paths are resolved relative to this file. Cells are marked with
"# %%" so editors such as VS Code and Spyder can run them one at a time.
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__))) # So the relative data paths below resolve

# %% [markdown]
# # Covasim for wastewater step by step tutorial
# 1. To run the following code, first make sure covasim and its dependencies are installed.
# Please follow the instructions here https://github.com/andersen-lab/covasim4wastewater to install.
# (Full covasim tutorial available at : https://docs.covasim.org/tutorials/)
# 2. To start, we run a basic simulation: 20,000 people with 20 initial infections for 100 days. The contact network is "hybrid", meaning people interact at workplaces, schools, households and also randomly. There are no imported cases.
# 3. People are placed on a map using the example San Diego files in `san-diego-data/`:
#    * `boundaries.geojson` — regions, each with `region_code`, `region_name` and `population` (used to assign people to regions and to draw their home coordinates).
#    * `sewersheds.geojson` — catchment polygons, each with a `sid` property. Each person is assigned the sewershed containing their home.
#
#    Both are optional (`boundary_shapefile=None`, `sewershed_file=None` by default, giving one region and one catchment). Supply your own files in the same format to model a different place.
#
# > **Note:** Run this notebook from the `docs/tutorials/` directory so the relative paths below resolve.

# %%
import covasim as cv
import numpy as np
import matplotlib.pyplot as plt

sim = cv.Sim(
    pop_size=20_000,
    pop_type='hybrid',
    pop_infected=20,
    n_days=100,
    n_imports=0,
    boundary_shapefile='san-diego-data/boundaries.geojson',
    sewershed_file='san-diego-data/sewersheds.geojson',
)
sim.run()

# %% [markdown]
# ## Wastewater viral loads
#
# To estimate how much each infected individual is shedding to wastewater,
#  we use the shedding hub https://shedding-hub.github.io, "woelfel2020virological" dataset.
#
# For each individual, the full `pop_size × n_days` shedding matrix is stored in `sim.results['viral_shedding_hist']`
# (memory grows with population × days, so keep this in mind for very large runs).
# The following plot shows how much each agent is shedding (cell color) over the simulation (x axis time, y axis agent).

# %%
viral_shedding = sim.results['viral_shedding_hist']
im = plt.imshow(
    viral_shedding,
    aspect='auto',   # prevents squashing
    origin='lower',  # agent 0 at bottom
    cmap='viridis',
)
plt.colorbar(im, label='Viral shedding')
plt.xlabel('time')
plt.ylabel('agent')
plt.title('Viral shedding by agent and time')
plt.tight_layout()
plt.show()

# %% [markdown]
# We then look at the overall shedding (total for all agents) and the shedding in each sewershed at each time point. This is the signal a wastewater treatment plant would see for its catchment.

# %%
sewersheds = np.asarray(sim.people.sewershed)
plt.figure(figsize=(12, 6))
for sid in np.unique(sewersheds):
    mask = sewersheds == sid
    plt.plot(viral_shedding[mask].sum(axis=0), label=f'Sewershed {sid} ({mask.sum()} people)')
plt.plot(viral_shedding.sum(axis=0), linewidth=2, color='red', label='Total')
plt.xlabel('Day')
plt.ylabel('Total viral shedding')
plt.title('Total viral shedding by sewershed')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The cumulative shedding in each sewershed can also be shown on a map:

# %%
import geopandas as gpd

sewershed_gdf = gpd.read_file('san-diego-data/sewersheds.geojson')
# sid strings like "SWR-01" become the integer 1 in sim.people.sewershed
sewershed_gdf['sid_num'] = sewershed_gdf['sid'].str.extract(r'(\d+)', expand=False).astype(int)

agent_total = viral_shedding.sum(axis=1)  # cumulative shedding per agent
totals = {sid: float(agent_total[sewersheds == sid].sum()) for sid in np.unique(sewersheds)}
sewershed_gdf['total_viral_shedding'] = sewershed_gdf['sid_num'].map(totals)

fig, ax = plt.subplots(figsize=(8, 8))
sewershed_gdf.plot(column='total_viral_shedding', cmap='viridis', edgecolor='0.3', legend=True,
                   legend_kwds={'label': 'Cumulative viral shedding', 'orientation': 'horizontal', 'shrink': 0.7}, ax=ax)
for _, row in sewershed_gdf.iterrows():
    c = row.geometry.centroid
    ax.annotate(row['sewershed_name'], xy=(c.x, c.y), ha='center', fontsize=8, color='white', fontweight='bold')
ax.set_title('San Diego example: cumulative viral shedding by sewershed')
ax.set_axis_off()
plt.show()

# %% [markdown]
# ## Genomic simulation
# In this section, we use Covasim to simulate how different SARS-CoV-2 variants—specifically **wild-type, AY.4, BA.2, and XBB.1.5**—spread over time. We track these variants using consensus FASTA files and plot their infectious counts on both absolute and proportional scales.
#
# ### How it works
#
# * **Seeding variants:** We introduce each new variant using a `founding_fasta` file, which defines its specific mutation profile when it enters the population.
# * **Tracking infections:** To track variant proportions over time, we pull daily counts directly from `sim.results['variant']['n_infectious_by_variant']`—Covasim's built-in tracker for daily infectious cases per variant.
# * **Wastewater sampling:** The `WastewaterSampler` tool isn't needed to plot these infection numbers; it's used only to generate the haplotype-resolved data required for Bygul (covered in the next section).
#
# > **Note:** Make sure to run this notebook from the `docs/tutorials/` directory so Covasim can find the FASTA files located in `sars-2-data/variants/`.

# %%


# Paths are relative to docs/tutorials/ — run the notebook from that directory.
ref_fasta         = 'sars-2-data/NC_045512_Hu-1.fasta'
fitness_data_path = '../../data/nt_fitness.csv'
ay4_fasta         = 'sars-2-data/variants/AY.4_consensus.fasta'
ba2_fasta         = 'sars-2-data/variants/BA.2.9_consensus.fasta'
xbb15_fasta       = 'sars-2-data/variants/XBB.1.5.18_consensus.fasta'
artic_primers     = 'sars-2-data/ARTIC_V4-1.bed'

# Variants — rel_beta is NOT hard-coded.  evo_pars (below) derives the
# transmission advantage from each variant's founding mutations by looking up
# every SNP in the Bloom lab nucleotide fitness landscape and computing
# rel_beta = exp(Σ log_fitness_i × fitness_scale).
ay4 = cv.variant(
    label          = 'AY.4',
    days           = 30, # introduce AY.4 on day 30
    n_imports      = 50,
    founding_fasta = ay4_fasta,
)
ba2 = cv.variant(
    label          = 'BA.2',
    days           = 60, # introduce BA.2 on day 60
    n_imports      = 50,
    founding_fasta = ba2_fasta,
)
xbb15 = cv.variant(
    label          = 'XBB.1.5',
    days           = 100, # introduce XBB.1.5 on day 100
    n_imports      = 50,
    founding_fasta = xbb15_fasta, # the variant's sequence when first introduced
)

# WastewaterSampler captures each unique evolved haplotype so simulate_sample()
# can feed the mixture to Bygul for amplicon-read synthesis.
# Sampled after XBB.1.5's introduction (day 100) while all four variants are
# likely still co-circulating.
_haplotype_days = list(range(110, 151, 5))   # days 110, 115, 120 … 150
ww = cv.WastewaterSampler(
    days  = _haplotype_days,
    label = 'WastewaterHaplotype',
)

sim = cv.Sim(
    dict(
        pop_size     = 20_000,
        pop_infected = 20, #initial infections
        start_day    = '2022-10-01',
        end_day      = '2023-04-01',
        evo_pars = dict( #evolutionary parameters for the simulation
            enable            = True,
            reference         = ref_fasta,
            mol_clock_rate    = 1e-6,
            fitness_model     = 'bloom_nt',           # Bloom lab nt_fitness.csv https://github.com/jbloomlab/SARS2-mut-fitness
            fitness_data_path = fitness_data_path,
            fitness_scale     = 0.05,                 # multiplies log-fitness sum before exp()
        ),
    ),
    variants  = [ay4, ba2, xbb15],
    analyzers = [ww],
    verbose   = 0,
)
sim.run()

# %% [markdown]
# Read more about sequence evolution : https://github.com/andersen-lab/covasim4wastewater/blob/main/docs/tutorials/tut_sequence_evol.ipynb
# The following code calculates the relative beta for each variant and shows the SNP counts compared to the founding sequence.

# %%
# Show the rel_beta multiplier that was derived from each variant's founding mutations.
# This makes the FASTA → fitness → transmissibility pathway explicit.
tracker       = sim.people.sequence_tracker
fitness_model = tracker.fitness_model

print(f'{"variant":12s}  {"founding SNPs":>13s}  {"rel_beta (×wt)":>15s}')
print('-' * 46)
# Wild-type carries no founding mutations → reference fitness = 1.0
for label, muts in [('wild', frozenset())] + list(tracker.variant_founding_mutations.items()):
    mult = fitness_model.compute_fitness(muts)
    print(f'{label:12s}  {len(muts):>13d}  {mult:>15.4f}')

# %% [markdown]
# The following code generates plots that show variant specific case counts at each time point. We can also see the prportions of infected people by each variant. The plot shows how each specific variant takes over based on the day they are introduced. e.g. 'AY.4': 30, 'BA.2': 60, 'XBB.1.5': 100 (dashed lines)

# %%
variant_names = list(sim['variant_map'].values())   # e.g. ['wild', 'AY.4', 'BA.2', 'XBB.1.5']
n_steps = sim.npts

# Per-variant infectious counts from the standard Covasim result.
# Shape: (n_variants, n_steps).
n_inf = np.array(sim.results['variant']['n_infectious_by_variant'].values)
total = n_inf.sum(axis=0)
total[total == 0] = 1  # avoid division by zero on days with no infections

loads = {v: n_inf[i] for i, v in enumerate(variant_names)}
props = {v: n_inf[i] / total for i, v in enumerate(variant_names)}

# Color palette (extend automatically for any extra variants)
palette = {'wild': '#4C72B0', 'AY.4': '#DD8452', 'BA.2': '#C44E52', 'XBB.1.5': '#55A868'}
_defaults = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, v in enumerate(variant_names):
    palette.setdefault(v, _defaults[i % len(_defaults)])

tvec       = np.arange(n_steps)
intro_days = {'AY.4': 30, 'BA.2': 60, 'XBB.1.5': 100}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# -- Absolute infectious counts -----------------------------------------------
for vname in variant_names:
    ax1.plot(tvec, loads[vname], label=vname, color=palette[vname], linewidth=1.8)
for vname, day in intro_days.items():
    ax1.axvline(day, color=palette[vname], linestyle=':', linewidth=1.2, alpha=0.7)
ax1.set_ylabel('Infectious agents')
ax1.set_title('Infectious count by variant — absolute')
ax1.legend(framealpha=0.9)

# -- Scaled to 1 (stacked area) -----------------------------------------------
stacked = np.array([props[v] for v in variant_names])
ax2.stackplot(tvec, stacked,
              labels=variant_names,
              colors=[palette[v] for v in variant_names],
              alpha=0.85)
for vname, day in intro_days.items():
    ax2.axvline(day, color='white', linestyle=':', linewidth=1.5, alpha=0.9)
ax2.set_xlabel('Simulation day')
ax2.set_ylabel('Proportion of infectious agents')
ax2.set_title('Variant proportion over time — scaled to 1')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Amplicon sequencing snapshot with Bygul
#
# `WastewaterSampler` records every unique evolved haplotype present in the
# infectious pool on each sampling day.  At a time-point where wild-type, AY.4,
# BA.2, and XBB.1.5 all co-circulate the mixture mirrors what a real wastewater
# sequencing lab would see.
#
# `simulate_sample()` writes each haplotype to a temporary FASTA file and calls
# [Bygul](https://github.com/andersen-lab/Bygul)'s `simulate_proportions` CLI,
# which tiles ARTIC V4.1 amplicons across each genome and produces synthetic
# paired-end reads at the observed mixture proportions — giving a realistic
# amplicon-sequencing dataset that reflects the simulated epidemiology.
#
# > **Requires** the `bygul` package: `pip install bygul`  (installing through conda is recommended conda install bioconda::bygul)
# > BED file and reference FASTA are read from `sars-2-data/` via the paths
# > set in the earlier cell (`artic_primers`, `ref_fasta`).

# %%
import glob, os, dataclasses

# From the candidate haplotype days, pick the first where all four variants
# each hold >1% of signal.  Fall back to day 120 if none qualifies.
ww = sim.get_analyzer('WastewaterHaplotype')
mixed_day = next(
    (t for t in sorted(ww.samples)
     if ww.samples[t] is not None
     and all(props[v][t] > 0.01 for v in ['wild', 'AY.4', 'BA.2', 'XBB.1.5'])),
    120,
)

print(f'Sampling day {mixed_day}  ({sim.date(mixed_day)})')
print('  Variant-level proportions at that day:')
for v in ['wild', 'AY.4', 'BA.2', 'XBB.1.5']:
    print(f'    {v:10s}  {props[v][mixed_day]:.1%}')

sample = ww.samples.get(mixed_day)
if sample is None:
    print('\nNo infectious agents were present on that day.')
else:
    print(f'\n  Unique haplotypes in pool : {sample.n_genotypes}')

    # Rank haplotypes by proportion and keep the top 5.
    # Renormalize so the subset proportions still sum to 1.0 — Bygul requires this.
    order     = sorted(range(sample.n_genotypes),
                       key=lambda i: sample.proportions[i], reverse=True)[:5]
    top_seqs  = [sample.mutation_sets[i]   for i in order]
    top_loads = [sample.raw_loads[i]   for i in order]
    top_props = [sample.proportions[i] for i in order]
    top_total = sum(top_props)
    top_props = [p / top_total for p in top_props]
    top_props[-1] = 1.0 - sum(top_props[:-1])   # absorb floating-point residual

    print(f'  Top-5 proportions (renormalized): {[round(p, 4) for p in top_props]}')
    print(f'  These cover {top_total:.1%} of the full pool by viral shedding')

    # Replace the stored sample with the trimmed version so simulate_sample() sees it
    ww.samples[mixed_day] = dataclasses.replace(
        sample,
        mutation_sets = top_seqs,
        proportions   = top_props,
        raw_loads       = top_loads,
        n_genotypes     = len(top_seqs),
    )

# artic_primers and ref_fasta were set in the earlier cell
outdir = './amplicon_reads/'

try:
    ww.simulate_sample(
        day             = mixed_day,
        primers         = artic_primers,
        reference       = ref_fasta,
        outdir          = outdir,
        readcnt         = 300,
        redo            = True,
        simulation_mode = 'amplicon',
        seed            = 42,
    )
    fastq_files = (
        glob.glob(os.path.join(outdir, '**/*.fastq*'), recursive=True) +
        glob.glob(os.path.join(outdir, '*.fastq*'))
    )
    print(f'\nBygul wrote {len(fastq_files)} FASTQ file(s) to {outdir!r}:')
    for f in sorted(set(fastq_files))[:12]:
        print(f'  {os.path.relpath(f)}  ({os.path.getsize(f)/1024:.1f} kB)')
except ImportError as exc:
    print('\n[Bygul not installed — skipping read simulation]')
    print(str(exc))
    print('Install: pip install bygul   (https://github.com/andersen-lab/Bygul)')

# %% [markdown]
# ## Clinical and wastewater data streams together
#
# The same simulation can produce both a clinical stream (sequences from diagnosed cases) and a wastewater stream (shedding-weighted haplotype mixtures per sewershed). Here we add testing, a `ClinicalSequencer`, and one `WastewaterSampler` per sewershed (`sid=`; `sid=None` samples everyone).
#
# > Covasim copies analyzers when the sim is created, so always read results back with `sim.get_analyzer(label)`, not from the original objects.

# %%
sample_days = list(range(40, 100, 10))
dual = cv.Sim(
    dict(
        pop_size     = 20_000,
        pop_infected = 20,
        n_days       = 100,
        evo_pars     = dict(enable=True, reference=ref_fasta, mol_clock_rate=1e-6,
                            fitness_model='bloom_nt', fitness_data_path=fitness_data_path, fitness_scale=0.05),
        boundary_shapefile = 'san-diego-data/boundaries.geojson',
        sewershed_file     = 'san-diego-data/sewersheds.geojson',
    ),
    variants      = [ay4],
    interventions = [cv.test_prob(symp_prob=0.2, start_day=0)],
    analyzers     = [cv.ClinicalSequencer(days=sample_days, n_samples=50, label='clinical')]
                  + [cv.WastewaterSampler(days=sample_days, sid=s, label=f'ww_sewershed_{s}') for s in (1, 2, 3)],
    verbose       = 0,
)
dual.run()

clinical = dual.get_analyzer('clinical')
for day in sample_days:
    row = f'day {day}: clinical n={len(clinical.samples.get(day, [])):3d}'
    for s in (1, 2, 3):
        smp = dual.get_analyzer(f'ww_sewershed_{s}').samples.get(day)
        row += f' | sewershed {s}: ' + (f'{smp.n_genotypes} haplotypes' if smp else 'no shedders')
    print(row)
