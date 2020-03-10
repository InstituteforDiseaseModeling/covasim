import scipy.stats as sps
import pylab as pl
import numpy as np
from covid_abm import utils as cov_ut

nSamples = 10000
DPY = 365.

def compare_new_infections():
    mult_vec = [1, 10/20., 1/20.]
    width=0.4
    fig,ax_vec = pl.subplots(1,len(mult_vec), figsize=(12,6), sharey=True)

    for i, mult in enumerate(mult_vec):
        ax = ax_vec[i]

        ## COVID-ABM - binomial model with n=20, p=2.5/200
        contacts = 20
        r_contact = 2.5/200
        contacts_mult = int(20 * mult)
        pvec = np.zeros(nSamples)
        for sample in range(nSamples):
            N = np.random.poisson(contacts_mult)
            pvec[sample] = np.random.binomial(n=N, p=r_contact)
        #d = sps.binom(n=contacts_mult, p=r_contact)
        K = range(5)
        #p = d.pmf(K)
        y,b = np.histogram(pvec, bins=K)
        y = y / y.sum()
        print(y)
        ax.bar(b[:-1], y, width, color='b', label='COVID-ABM')

        ## transGenEpi - heterogeneous beta ~ positiveNormal(mu=106/365, sigma=260/365)
        beta_config = {
            'type': 'positiveNormal',
            'params': {
                'mu': 106,
                'sigma': 260
            }
        }

        pvec = np.zeros(nSamples)
        for sample in range(nSamples):
            beta = cov_ut.sample(beta_config)
            pvec[sample] = 1-pl.exp(-beta * mult / DPY)
        mu = pvec.mean()
        ax.bar([width,1+width], [1-mu,mu], width, color='r', label='transGenEpi')

        ax.set_xlabel('Transmissions')
        if i == 0: ax.set_ylabel('Probability')
        if mult == 1:
            ax.set_title('Baseline')
        else:
            ax.set_title(f'R0 reduction: {1-mult}')
        pl.legend()

    pl.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Transmissions per infectious individual per day')

    pl.savefig('Compare_NewInfections.png')

def sample_config(config, samp=10000):
    ret = np.zeros(samp)
    for sample in range(samp):
        ret[sample] = cov_ut.sample(config)
    return ret

def compare_epi():
    fig,ax_vec = pl.subplots(1,2,figsize=(12,6), sharey=True)

    ### Latent period

    # transGenEpi
    transGenEpi = {
        'type': 'positiveNormal',
        'params': {
            'mu':          0.011 * DPY,
            'sigma':       0.0027 * DPY
        }
    }
    tge = np.ceil(sample_config(transGenEpi, samp=nSamples)) # Due to timestep

    # COVID-ABM
    covidabm = {
        'type': 'normal',
        'params': {
            'mu':          5,
            'sigma':       1
        }
    }
    covid = np.ceil(sample_config(covidabm, samp=nSamples)) # Due to timestep

    ax = ax_vec[0]
    ax.hist([covid, tge], bins = range(10), normed=True,
         color = ['b', 'r'], label=['COVID-ABM', 'transGenEpi'], align='left')

    ax.set_xlabel('Days')
    ax.set_ylabel('Probability')
    ax.set_title('Latent Period')


    ### Infectious period

    # transGenEpi
    transGenEpi = {
        'type': 'positiveNormal',
        'params': {
            'mu':          0.0219 * DPY,
            'sigma':       0.0055 * DPY
        }
    }
    tge = np.ceil(sample_config(transGenEpi)) # Due to timestep

    # COVID-ABM
    covidabm = {
        'type': 'normal',
        'params': {
            'mu':          10,
            'sigma':       3
        }
    }
    covid = np.ceil(sample_config(covidabm)) # Due to timestep

    ax = ax_vec[1]
    ax.hist([covid, tge], bins = range(20), normed=True,
         color = ['b', 'r'], label=['COVID-ABM', 'transGenEpi'], align='left')

    ax.set_xlabel('Days')
    ax.set_ylabel('Probability')
    ax.set_title('Infectious Period')
    ax.legend()
    pl.savefig('Compare_Epi.png')


compare_new_infections()
compare_epi()

pl.show()
