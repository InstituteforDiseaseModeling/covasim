'''
Set the parameters for Covasim.
'''

import numpy as np
import sciris as sc
from .settings import options as cvo # For setting global options
from . import misc as cvm
from . import defaults as cvd

__all__ = ['make_pars', 'reset_layer_pars', 'get_prognoses', 'get_variant_choices', 'get_vaccine_choices']


def make_pars(set_prognoses=False, prog_by_age=True, version=None, **kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = cv.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        set_prognoses (bool): whether or not to create prognoses (else, added when the population is created)
        prog_by_age   (bool): whether or not to use age-based severity, mortality etc.
        kwargs        (dict): any additional kwargs are interpreted as parameter names
        version       (str):  if supplied, use parameters from this Covasim version

    Returns:
        pars (dict): the parameters of the simulation
    '''

#LocationElements
public class popPara implements LocationElements{    # Population parameters
  public String elements(){
  return "popPara";
  }
}
public class tempPara implements LocationElements{
  public String elements(){
  return "tempPara";
  }
}

public class covidPara implements LocationElements{
  public String elements(){
  return "covidPara";
  }
}

#PersonalElements
public class netPara implements PersonalElements{
  public String elements(){
  return "netPara";
  }
}

public class Age implements PersonalElements{
  public String elements(){
  return "Age";
  }
}

public class sexual implements PersonalElements{
  public String elements(){
  return "sexual";
  }
}

public class basicDisPara implements PersonalElements{
  public String elements(){
  return "basicDisPara";
  }
}

public class calImmuPara implements PersonalElements{ # Parameters used to calculate immunity
  public String elements(){
  return "calImmuPara";
  }
}

public class varSpecPara implements PersonalElements{     # Variant-specific disease transmission parameters. By default, these are set up for a single variant, but can all be modified for multiple variants
  public String elements(){
  return "varSpecPara";
  }
}

#TechElements
public class serviPara implements TechElements{    # Severity parameters: probabilities of symptom progression
  public String elements(){
  return "serviPara";
  }
}

public class effiPara implements TechElements{    # Efficacy of protection measures
  public String elements(){
  return "effiPara";
  }
}

public class eventPara implements TechElements{    # Events and interventions
  public String elements(){
  return "eventPara";
  }
}

public class heathPara implements TechElements{    # Health system parameters
  public String elements(){
  return "heathPara";
  }
}

public class variPara implements TechElements{    # Handle vaccine and variant parameters
  public String elements(){
  return "variPara";
  }
}


#TimeDuration

public class proPara implements TimeDuration{    # Duration parameters: time for disease progression
  public String elements(){
  return "proPara";
  }
}
public class recPara implements TimeDuration{    # Duration parameters: time for disease recovery
  public String elements(){
  return "recPara";
  }
}
public class multiVarPara implements TimeDuration{     # Parameters that control settings and defaults for multi-variant runs
  public String elements(){
  return "multiVarPara";
  }
}
public class simPara implements TimeDuration{         # Simulation parameters
  public String elements(){
  return "simPara";
  }
}

public class resPara implements TimeDuration{    # Rescaling parameters
  public String elements(){
  return "resPara";
  }
}

#location Details

public class popPara implements LocationElements{    # Population parameters.
  @override
  public String elements(){
  return "popPara";
  }
  @override
  public abstract float percetangeWeight();
}

public class popSize extends popPara {

   @Override
   public int number() {
      return 20e3;
   }

   @Override
   public String name() {
      return "pop_size";
   }
}

public class popiInfected extends popPara {

   @Override
   public int number() {
      return 20;
   }

   @Override
   public String name() {
      return "pop_infected";
   }
}
public class popType extends popPara {

   @Override
   public int number() {
      return 'random';
   }

   @Override
   public String name() {
      return "pop_type";
   }
}

public class location extends popPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "location";
   }
}

public class tempPara implements LocationElements{
  @override
  public String elements(){
  return "tempPara";
  }
  @override
  public abstract float percetangeWeight();
}

public class covidPara implements LocationElements{
  @override
  public String elements(){
  return "covidPara";
  }
  @override
  public abstract float percetangeWeight();
}


#personal Details

public class calImmuPara implements PersonalElements{ # Parameters used to calculate immunity
  public String elements(){
  return "calImmuPara";
  }
}

public class useWaning extends calImmuPara {

   @Override
   public int number() {
      return False;
   }

   @Override
   public String name() {
      return "use_waning";
   }
}

public class nabInit extends calImmuPara {

   @Override
   public int number() {
      return dict(dist='normal', par1=0, par2=2);
   }

   @Override
   public String name() {
      return "nab_init";
   }
}

public class nabDecay extends calImmuPara {

   @Override
   public int number() {
      return dict(form='nab_growth_decay', growth_time=22, decay_rate1=np.log(2)/100, decay_time1=250, decay_rate2=np.log(2)/3650, decay_time2=365);
   }

   @Override
   public String name() {
      return "nab_decay";
   }
}

public class nabKin extends calImmuPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "nab_kin";
   }
}

public class nabBoost extends calImmuPara {

   @Override
   public int number() {
      return 1.5;
   }

   @Override
   public String name() {
      return "nab_boost";
   }
}

public class nabEff extends calImmuPara {

   @Override
   public int number() {
      return dict(alpha_inf=3.5, beta_inf=1.219, alpha_symp_inf=-1.06, beta_symp_inf=0.867, alpha_sev_symp=0.268, beta_sev_symp=3.4);
   }

   @Override
   public String name() {
      return "nab_eff";
   }
}
public class relImmSymp extends calImmuPara {

   @Override
   public int number() {
      return dict(asymp=0.85, mild=1, severe=1.5);
   }

   @Override
   public String name() {
      return "rel_imm_symp";
   }
}
public class immunity extends calImmuPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "immunity";
   }
}

public class varSpecPara implements PersonalElements{     # Variant-specific disease transmission parameters. By default, these are set up for a single variant, but can all be modified for multiple variants
  public String elements(){
  return "varSpecPara";
  }
}

public class relBeta extends calImmuPara {

   @Override
   public int number() {
      return 1.0;
   }

   @Override
   public String name() {
      return "rel_beta";
   }
}

public class relImmVariant extends calImmuPara {

   @Override
   public int number() {
      return 1.0;
   }

   @Override
   public String name() {
      return "rel_imm_variant";
   }
}

public class netPara implements PersonalElements{
  public String elements(){
  return "netPara";
  }
}

public class Age implements PersonalElements{
  public String elements(){
  return "Age";
  }
}

public class sexual implements PersonalElements{
  public String elements(){
  return "sexual";
  }
}

public class basicDisPara implements PersonalElements{
  public String elements(){
  return "basicDisPara";
  }
}


#Tech Details

public class serviPara implements TechElements{    # Severity parameters: probabilities of symptom progression
  public String elements(){
  return "serviPara";
  }
}

public class relSympProb extends serviPara {

   @Override
   public int number() {
      return 1.0;
   }

   @Override
   public String name() {
      return "rel_symp_prob";
   }
}

public class relSevereProb extends serviPara {

   @Override
   public int number() {
      return 1.0;
   }

   @Override
   public String name() {
      return "rel_severe_prob";
   }
}


public class relCritProb extends serviPara {

   @Override
   public int number() {
      return 1.0;
   }

   @Override
   public String name() {
      return "rel_crit_prob";
   }
}

public class relDeathProb extends serviPara {

   @Override
   public int number() {
      return 1.0;
   }

   @Override
   public String name() {
      return "rel_death_prob";
   }
}

public class progByAge extends serviPara {

   @Override
   public char prog() {
      return prog_by_age;
   }

   @Override
   public String name() {
      return "prog_by_age";
   }
}

public class prognoses extends serviPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "prognoses";
   }
}


public class effiPara implements TechElements{    # Efficacy of protection measures
  public String elements(){
  return "effiPara";
  }
}

public class isoFactor extends effiPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "iso_factor";
   }
}

public class quarFactor extends effiPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "quar_factor";
   }
}

public class quarperiod extends effiPara {

   @Override
   public int number() {
      return 14;
   }

   @Override
   public String name() {
      return "quar_period";
   }
}


public class eventPara implements TechElements{    # Events and interventions
  public String elements(){
  return "eventPara";
  }
}


public class interventions extends eventPara {

   @Override
   public int number() {
      return [];
   }

   @Override
   public String name() {
      return "interventions";
   }
}

public class analyzers extends eventPara {

   @Override
   public int number() {
      return [];
   }

   @Override
   public String name() {
      return "analyzers";
   }
}

public class timelimit extends eventPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "timelimit";
   }
}

public class stoppingFunc extends eventPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "stopping_func";
   }
}

public class heathPara implements TechElements{    # Health system parameters
  public String elements(){
  return "heathPara";
  }
}

public class nBedsHosp extends heathPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "n_beds_hosp";
   }
}

public class nBedsIcu extends heathPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "n_beds_icu";
   }
}

public class noHospFactor extends heathPara {

   @Override
   public int number() {
      return 2.0;
   }

   @Override
   public String name() {
      return "n_beds_icu";
   }
}

public class noIcuFactor extends heathPara {

   @Override
   public int number() {
      return 2.0;
   }

   @Override
   public String name() {
      return "noIcuFactor";
   }
}


public class variPara implements TechElements{    # Handle vaccine and variant parameters
  public String elements(){
  return "variPara";
  }
}

public class vaccinePars extends variPara {

   @Override
   public int number() {
      return {};
   }

   @Override
   public String name() {
      return "vaccine_pars";
   }
}

public class vaccineMap extends variPara {

   @Override
   public int number() {
      return {};
   }

   @Override
   public String name() {
      return "vaccine_map";
   }
}

public class variants extends variPara {

   @Override
   public int number() {
      return [];
   }

   @Override
   public String name() {
      return "variants";
   }
}

public class variantMap extends variPara {

   @Override
   public int number() {
      return {0:'wild'};
   }

   @Override
   public String name() {
      return "variant_map";
   }
}

public class variantPars extends variPara {

   @Override
   public int number() {
      return dict(wild={});
   }

   @Override
   public String name() {
      return "variant_pars";
   }
}

#Time Details

public class proPara implements TimeDuration{    # Duration parameters: time for disease progression
  public String elements(){
  return "proPara";
  }
}

public class dur extends proPara {

   @Override
   public int number() {
      return {};
   }

   @Override
   public String name() {
      return "dur";
   }
}

public class dur_exp2inf extends proPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=4.5, par2=1.5);
   }

   @Override
   public String name() {
      return "['dur']['exp2inf']";
   }
}

public class dur_inf2sym extends proPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=1.1, par2=0.9);
   }

   @Override
   public String name() {
      return "['dur']['inf2sym']";
   }
}

public class dur_sym2sev extends proPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=6.6, par2=4.9);
   }

   @Override
   public String name() {
      return "['dur']['sym2sev']";
   }
}

public class dur_sev2crit extends proPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=1.5, par2=2.0);
   }

   @Override
   public String name() {
      return "['dur']['sev2crit']";
   }
}


public class recPara implements TimeDuration{    # Duration parameters: time for disease recovery
  public String elements(){
  return "recPara";
  }
}

public class dur_asym2rec extends recPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=8.0,  par2=2.0);
   }

   @Override
   public String name() {
      return "['dur']['asym2rec']";
   }
}

public class dur_mild2rec extends recPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=8.0,  par2=2.0);
   }

   @Override
   public String name() {
      return "['dur']['mild2rec']";
   }
}

public class dur_sev2rec extends recPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=18.1,  par2=2.3);
   }

   @Override
   public String name() {
      return "['dur']['sev2rec']";
   }
}

public class dur_crit2rec extends recPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=18.1,  par2=2.3);
   }

   @Override
   public String name() {
      return "['dur']['crit2rec']";
   }
}

public class dur_crit2die extends recPara {

   @Override
   public int number() {
      return dict(dist='lognormal_int', par1=10.7,  par2=4.8);
   }

   @Override
   public String name() {
      return "['dur']['crit2die']";
   }
}


public class multiVarPara implements TimeDuration{     # Parameters that control settings and defaults for multi-variant runs
  public String elements(){
  return "multiVarPara";
  }
}

public class nImports extends multiVarPara {

   @Override
   public int number() {
      return 0;
   }

   @Override
   public String name() {
      return 'n_imports';
   }
}

public class nVariants extends multiVarPara {

   @Override
   public int number() {
      return 1;
   }

   @Override
   public String name() {
      return 'n_variants';
   }
}


public class simPara implements TimeDuration{         # Simulation parameters
  public String elements(){
  return "simPara";
  }
}

public class startDay extends simPara {

   @Override
   public int number() {
      return '2020-03-01';
   }

   @Override
   public String name() {
      return "start_day";
   }
}

public class endDay extends simPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "end_day";
   }
}

public class nDays extends simPara {

   @Override
   public int number() {
      return 60;
   }

   @Override
   public String name() {
      return "n_days";
   }
}

public class randSeed extends simPara {

   @Override
   public int number() {
      return 1;
   }

   @Override
   public String name() {
      return "rand_seed";
   }
}

public class verbose extends simPara {

   @Override
   public char verbose() {
      return cvo.verbose;
   }

   @Override
   public String name() {
      return "verbose";
   }
}


public class resPara implements TimeDuration{    # Rescaling parameters
  public String elements(){
  return "resPara";
  }
}

public class popScale extends resPara {

   @Override
   public int number() {
      return 1;
   }

   @Override
   public String name() {
      return "pop_scale";
   }
}

public class scaledPop extends resPara {

   @Override
   public int number() {
      return None;
   }

   @Override
   public String name() {
      return "scaled_pop";
   }
}

public class rescale extends resPara {

   @Override
   public int number() {
      return Ture;
   }

   @Override
   public String name() {
      return "rescale";
   }
}

public class rescaleThreshold extends resPara {

   @Override
   public int number() {
      return 0.05;
   }

   @Override
   public String name() {
      return "rescale_threshold";
   }
}

public class rescaleFactor extends resPara {

   @Override
   public int number() {
      return 1.2;
   }

   @Override
   public String name() {
      return "rescale_factor";
   }
}

public class fracSusceptible extends resPara {

   @Override
   public int number() {
      return 1.0;
   }

   @Override
   public String name() {
      return "frac_susceptible";
   }
}

    for sp in cvd.variant_pars:
        if sp in pars.keys():
            pars['variant_pars']['wild'][sp] = pars[sp]

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)
    if set_prognoses: # If not set here, gets set when the population is initialized
        pars['prognoses'] = get_prognoses(pars['prog_by_age'], version=version) # Default to age-specific prognoses

    # If version is specified, load old parameters
    if version is not None:
        version_pars = cvm.get_version_pars(version, verbose=pars['verbose'])
        for key in pars.keys(): # Only loop over keys that have been populated
            if key in version_pars: # Only replace keys that exist in the old version
                pars[key] = version_pars[key]

        # Handle code change migration
        if sc.compareversions(version, '2.1.0') == -1 and 'migrate_lognormal' not in pars:
            cvm.migrate_lognormal(pars, verbose=pars['verbose'])

    return pars


# Define which parameters need to be specified as a dictionary by layer -- define here so it's available at the module level for sim.py
layer_pars = ['beta_layer', 'contacts', 'dynam_layer', 'iso_factor', 'quar_factor']


def reset_layer_pars(pars, layer_keys=None, force=False):
    '''
    Helper function to set layer-specific parameters. If layer keys are not provided,
    then set them based on the population type. This function is not usually called
    directly by the user, although it can sometimes be used to fix layer key mismatches
    (i.e. if the contact layers in the population do not match the parameters). More
    commonly, however, mismatches need to be fixed explicitly.

    Args:
        pars (dict): the parameters dictionary
        layer_keys (list): the layer keys of the population, if available
        force (bool): reset the parameters even if they already exist
    '''

    # Specify defaults for random -- layer 'a' for 'all'
    layer_defaults = {}
    layer_defaults['random'] = dict(
        beta_layer  = dict(a=1.0), # Default beta
        contacts    = dict(a=20),  # Default number of contacts
        dynam_layer = dict(a=0),   # Do not use dynamic layers by default
        iso_factor  = dict(a=0.2), # Assumed isolation factor
        quar_factor = dict(a=0.3), # Assumed quarantine factor
    )

    # Specify defaults for hybrid -- household, school, work, and community layers (h, s, w, c)
    layer_defaults['hybrid'] = dict(
        beta_layer  = dict(h=3.0, s=0.6, w=0.6, c=0.3),  # Per-population beta weights; relative; in part based on Table S14 of https://science.sciencemag.org/content/sci/suppl/2020/04/28/science.abb8001.DC1/abb8001_Zhang_SM.pdf
        contacts    = dict(h=2.0, s=20,  w=16,  c=20),   # Number of contacts per person per day, estimated
        dynam_layer = dict(h=0,   s=0,   w=0,   c=0),    # Which layers are dynamic -- none by default
        iso_factor  = dict(h=0.3, s=0.1, w=0.1, c=0.1),  # Multiply beta by this factor for people in isolation
        quar_factor = dict(h=0.6, s=0.2, w=0.2, c=0.2),  # Multiply beta by this factor for people in quarantine
    )

    # Specify defaults for SynthPops -- same as hybrid except for LTCF layer (l)
    l_pars = dict(beta_layer=1.5,
                  contacts=10,
                  dynam_layer=0,
                  iso_factor=0.2,
                  quar_factor=0.3
    )
    layer_defaults['synthpops'] = sc.dcp(layer_defaults['hybrid'])
    for key,val in l_pars.items():
        layer_defaults['synthpops'][key]['l'] = val

    # Choose the parameter defaults based on the population type, and get the layer keys
    try:
        defaults = layer_defaults[pars['pop_type']]
    except Exception as E:
        errormsg = f'Cannot load defaults for population type "{pars["pop_type"]}": must be hybrid, random, or synthpops'
        raise ValueError(errormsg) from E
    default_layer_keys = list(defaults['beta_layer'].keys()) # All layers should be the same, but use beta_layer for convenience

    # Actually set the parameters
    for pkey in layer_pars:
        par = {} # Initialize this parameter
        default_val = layer_defaults['random'][pkey]['a'] # Get the default value for this parameter

        # If forcing, we overwrite any existing parameter values
        if force:
            par_dict = defaults[pkey] # Just use defaults
        else:
            par_dict = sc.mergedicts(defaults[pkey], pars.get(pkey, None)) # Use user-supplied parameters if available, else default

        # Figure out what the layer keys for this parameter are (may be different between parameters)
        if layer_keys:
            par_layer_keys = layer_keys # Use supplied layer keys
        else:
            par_layer_keys = list(sc.odict.fromkeys(default_layer_keys + list(par_dict.keys())))  # If not supplied, use the defaults, plus any extra from the par_dict; adapted from https://www.askpython.com/python/remove-duplicate-elements-from-list-python

        # Construct this parameter, layer by layer
        for lkey in par_layer_keys: # Loop over layers
            par[lkey] = par_dict.get(lkey, default_val) # Get the value for this layer if available, else use the default for random
        pars[pkey] = par # Save this parameter to the dictionary

    return


def get_prognoses(by_age=True, version=None):
    '''
    Return the default parameter values for prognoses

    The prognosis probabilities are conditional given the previous disease state.

    Args:
        by_age (bool): whether to use age-specific values (default true)

    Returns:
        prog_pars (dict): the dictionary of prognosis probabilities
    '''

    if not by_age: # All rough estimates -- almost always, prognoses by age (below) are used instead
        prognoses = dict(
            age_cutoffs   = np.array([0]),
            sus_ORs       = np.array([1.00]),
            trans_ORs     = np.array([1.00]),
            symp_probs    = np.array([0.75]),
            comorbidities = np.array([1.00]),
            severe_probs  = np.array([0.10]),
            crit_probs    = np.array([0.04]),
            death_probs   = np.array([0.01]),
        )
    else:
        prognoses = dict(
            age_cutoffs   = np.array([0,       10,      20,      30,      40,      50,      60,      70,      80,      90,]),     # Age cutoffs (lower limits)
            sus_ORs       = np.array([0.34,    0.67,    1.00,    1.00,    1.00,    1.00,    1.24,    1.47,    1.47,    1.47]),    # Odds ratios for relative susceptibility -- from Zhang et al., https://science.sciencemag.org/content/early/2020/05/04/science.abb8001; 10-20 and 60-70 bins are the average across the ORs
            trans_ORs     = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Odds ratios for relative transmissibility -- no evidence of differences
            comorbidities = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Comorbidities by age -- set to 1 by default since already included in disease progression rates
            symp_probs    = np.array([0.50,    0.55,    0.60,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90,    0.90]),    # Overall probability of developing symptoms (based on https://www.medrxiv.org/content/10.1101/2020.03.24.20043018v1.full.pdf, scaled for overall symptomaticity)
            severe_probs  = np.array([0.00050, 0.00165, 0.00720, 0.02080, 0.03430, 0.07650, 0.13280, 0.20655, 0.24570, 0.24570]), # Overall probability of developing severe symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            crit_probs    = np.array([0.00003, 0.00008, 0.00036, 0.00104, 0.00216, 0.00933, 0.03639, 0.08923, 0.17420, 0.17420]), # Overall probability of developing critical symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            death_probs   = np.array([0.00002, 0.00002, 0.00010, 0.00032, 0.00098, 0.00265, 0.00766, 0.02439, 0.08292, 0.16190]), # Overall probability of dying -- from O'Driscoll et al., https://www.nature.com/articles/s41586-020-2918-0; last data point from Brazeau et al., https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-34-ifr/
        )
    prognoses = relative_prognoses(prognoses) # Convert to conditional probabilities

    # If version is specified, load old parameters
    if by_age and version is not None:
        version_prognoses = cvm.get_version_pars(version, verbose=False)['prognoses']
        for key in version_prognoses.keys(): # Only loop over keys that have been populated
            if key in version_prognoses: # Only replace keys that exist in the old version
                prognoses[key] = np.array(version_prognoses[key])

    # Check that lengths match
    expected_len = len(prognoses['age_cutoffs'])
    for key,val in prognoses.items():
        this_len = len(prognoses[key])
        if this_len != expected_len: # pragma: no cover
            errormsg = f'Lengths mismatch in prognoses: {expected_len} age bins specified, but key "{key}" has {this_len} entries'
            raise ValueError(errormsg)

    return prognoses


def relative_prognoses(prognoses):
    '''
    Convenience function to revert absolute prognoses into relative (conditional)
    ones. Internally, Covasim uses relative prognoses.
    '''
    out = sc.dcp(prognoses)
    out['death_probs']  /= out['crit_probs']   # Conditional probability of dying, given critical symptoms
    out['crit_probs']   /= out['severe_probs'] # Conditional probability of symptoms becoming critical, given severe
    out['severe_probs'] /= out['symp_probs']   # Conditional probability of symptoms becoming severe, given symptomatic
    return out


def absolute_prognoses(prognoses):
    '''
    Convenience function to revert relative (conditional) prognoses into absolute
    ones. Used to convert internally used relative prognoses into more readable
    absolute ones.

    **Example**::

        sim = cv.Sim()
        abs_progs = cv.parameters.absolute_prognoses(sim['prognoses'])
    '''
    out = sc.dcp(prognoses)
    out['severe_probs'] *= out['symp_probs']   # Absolute probability of severe symptoms
    out['crit_probs']   *= out['severe_probs'] # Absolute probability of critical symptoms
    out['death_probs']  *= out['crit_probs']   # Absolute probability of dying
    return out


#%% Variant, vaccine, and immunity parameters and functions

def get_variant_choices():
    '''
    Define valid pre-defined variant names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'wild':   ['wild', 'default', 'pre-existing', 'original'],
        'b117':   ['alpha', 'b117', 'uk', 'united kingdom', 'kent'],
        'b1351':  ['beta', 'b1351', 'sa', 'south africa'],
        'p1':     ['gamma', 'p1', 'b11248', 'brazil'],
        'b16172': ['delta', 'b16172', 'india'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping


def get_vaccine_choices():
    '''
    Define valid pre-defined vaccine names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'default': ['default', None],
        'pfizer':  ['pfizer', 'biontech', 'pfizer-biontech', 'pf', 'pfz', 'pz'],
        'moderna': ['moderna', 'md'],
        'novavax': ['novavax', 'nova', 'covovax', 'nvx', 'nv'],
        'az':      ['astrazeneca', 'az', 'covishield', 'oxford', 'vaxzevria'],
        'jj':      ['jnj', 'johnson & johnson', 'janssen', 'jj'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping


def get_variant_pars(default=False):
    '''
    Define the default parameters for the different variants
    '''
    pars = dict(

        wild = dict(
            rel_beta        = 1.0, # Default values
            rel_symp_prob   = 1.0, # Default values
            rel_severe_prob = 1.0, # Default values
            rel_crit_prob   = 1.0, # Default values
            rel_death_prob  = 1.0, # Default values
        ),

        b117 = dict(
            rel_beta        = 1.67, # Midpoint of the range reported in https://science.sciencemag.org/content/372/6538/eabg3055
            rel_symp_prob   = 1.0,  # Inconclusive evidence on the likelihood of symptom development. See https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(21)00055-4/fulltext
            rel_severe_prob = 1.64, # From https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3792894, and consistent with https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.16.2100348 and https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/961042/S1095_NERVTAG_update_note_on_B.1.1.7_severity_20210211.pdf
            rel_crit_prob   = 1.0,  # Various studies have found increased mortality for B117 (summary here: https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(21)00201-2/fulltext#tbl1), but not necessarily when conditioned on having developed severe disease
            rel_death_prob  = 1.0,  # See comment above
        ),

        b1351 = dict(
            rel_beta        = 1.0, # No increase in transmissibility; B1351's fitness advantage comes from the reduction in neutralisation
            rel_symp_prob   = 1.0,
            rel_severe_prob = 3.6, # From https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.16.2100348
            rel_crit_prob   = 1.0,
            rel_death_prob  = 1.0,
        ),

        p1 = dict(
            rel_beta        = 2.05, # Estimated to be 1.7–2.4-fold more transmissible than wild-type: https://science.sciencemag.org/content/early/2021/04/13/science.abh2644
            rel_symp_prob   = 1.0,
            rel_severe_prob = 2.6, # From https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.16.2100348
            rel_crit_prob   = 1.0,
            rel_death_prob  = 1.0,
        ),

        b16172 = dict(
            rel_beta        = 2.2, # Estimated to be 1.25-1.6-fold more transmissible than B117: https://www.researchsquare.com/article/rs-637724/v1
            rel_symp_prob   = 1.0,
            rel_severe_prob = 3.2, # 2x more transmissible than alpha from https://mobile.twitter.com/dgurdasani1/status/1403293582279294983
            rel_crit_prob   = 1.0,
            rel_death_prob  = 1.0,
        )
    )

    if default:
        return pars['wild']
    else:
        return pars


def get_cross_immunity(default=False):
    '''
    Get the cross immunity between each variant in a sim
    '''
    pars = dict(

        wild = dict(
            wild   = 1.0, # Default for own-immunity
            b117   = 0.5, # Assumption
            b1351  = 0.5, # Assumption
            p1     = 0.5, # Assumption
            b16172 = 0.5, # Assumption
        ),

        b117 = dict(
            wild   = 0.5, # Assumption
            b117   = 1.0, # Default for own-immunity
            b1351  = 0.8, # Assumption
            p1     = 0.8, # Assumption
            b16172 = 0.8  # Assumption
        ),

        b1351 = dict(
            wild   = 0.066, # https://www.nature.com/articles/s41586-021-03471-w
            b117   = 0.5,   # Assumption
            b1351  = 1.0,   # Default for own-immunity
            p1     = 0.5,   # Assumption
            b16172 = 0.5    # Assumption
        ),

        p1 = dict(
            wild   = 0.34, # Previous (non-P.1) infection provides 54–79% of the protection against infection with P.1 that it provides against non-P.1 lineages: https://science.sciencemag.org/content/early/2021/04/13/science.abh2644
            b117   = 0.4,  # Assumption based on the above
            b1351  = 0.4,  # Assumption based on the above
            p1     = 1.0,  # Default for own-immunity
            b16172 = 0.8   # Assumption
        ),

        b16172=dict( # Parameters from https://www.cell.com/cell/fulltext/S0092-8674(21)00755-8
            wild   = 0.374,
            b117   = 0.689,
            b1351  = 0.086,
            p1     = 0.088,
            b16172 = 1.0 # Default for own-immunity
        ),
    )

    if default:
        return pars['wild']
    else:
        return pars


def get_vaccine_variant_pars(default=False):
    '''
    Define the effectiveness of each vaccine against each variant
    '''
    pars = dict(

        default = dict(
            wild   = 1.0,
            b117   = 1.0,
            b1351  = 1.0,
            p1     = 1.0,
            b16172 = 1.0,
        ),

        pfizer = dict(
            wild   = 1.0,
            b117   = 1/2.0,
            b1351  = 1/6.7,
            p1     = 1/6.5,
            b16172 = 1/2.9, # https://www.researchsquare.com/article/rs-637724/v1
        ),

        moderna = dict(
            wild   = 1.0,
            b117   = 1/1.8,
            b1351  = 1/4.5,
            p1     = 1/8.6,
            b16172 = 1/2.9,  # https://www.researchsquare.com/article/rs-637724/v1
        ),

        az = dict(
            wild   = 1.0,
            b117   = 1/2.3,
            b1351  = 1/9,
            p1     = 1/2.9,
            b16172 = 1/6.2,  # https://www.researchsquare.com/article/rs-637724/v1
        ),

        jj = dict(
            wild   = 1.0,
            b117   = 1.0,
            b1351  = 1/6.7,
            p1     = 1/8.6,
            b16172 = 1/6.2,  # Assumption, no data available yet
        ),

        novavax = dict( # Data from https://ir.novavax.com/news-releases/news-release-details/novavax-covid-19-vaccine-demonstrates-893-efficacy-uk-phase-3
            wild   = 1.0,
            b117   = 1/1.12,
            b1351  = 1/4.7,
            p1     = 1/8.6, # Assumption, no data available yet
            b16172 = 1/6.2, # Assumption, no data available yet
        ),
    )

    if default:
        return pars['default']
    else:
        return pars


def get_vaccine_dose_pars(default=False):
    '''
    Define the parameters for each vaccine
    '''

    # Default vaccine NAb efficacy is nearly identical to infection -- only alpha_inf differs
    default_nab_eff = dict(
        alpha_inf      =  1.11,
        beta_inf       =  1.219,
        alpha_symp_inf = -1.06,
        beta_symp_inf  =  0.867,
        alpha_sev_symp =  0.268,
        beta_sev_symp  =  3.4
    )

    pars = dict(

        default = dict(
            nab_eff   = sc.dcp(default_nab_eff),
            nab_init  = dict(dist='normal', par1=2, par2=2),
            nab_boost = 2,
            doses     = 1,
            interval  = None,
        ),

        pfizer = dict(
            nab_eff   = sc.dcp(default_nab_eff),
            nab_init  = dict(dist='normal', par1=2, par2=2),
            nab_boost = 3,
            doses     = 2,
            interval  = 21,
        ),

        moderna = dict(
            nab_eff   = sc.dcp(default_nab_eff),
            nab_init  = dict(dist='normal', par1=2, par2=2),
            nab_boost = 3,
            doses     = 2,
            interval  = 28,
        ),

        az = dict(
            nab_eff   = sc.dcp(default_nab_eff),
            nab_init  = dict(dist='normal', par1=-1, par2=2),
            nab_boost = 3,
            doses     = 2,
            interval  = 21,
        ),

        jj = dict(
            nab_eff   = sc.dcp(default_nab_eff),
            nab_init  = dict(dist='normal', par1=1, par2=2),
            nab_boost = 3,
            doses     = 1,
            interval  = None,
        ),

        novavax = dict(
            nab_eff   = sc.dcp(default_nab_eff),
            nab_init  = dict(dist='normal', par1=-0.9, par2=2),
            nab_boost = 3,
            doses     = 2,
            interval  = 21,
        ),
    )

    if default:
        return pars['default']
    else:
        return pars
