import covasim as cv
from unittest_support_classes import CovaSimTest


class SimulationKeys:
    ''' Define explicit mapping to simulation keys '''
    number_agents = 'n'
    population_scaling_factor = 'scale'
    initial_infected_count = 'n_infected'
    start_day = 'start_day'
    number_simulated_days = 'n_days'
    random_seed = 'seed'
    pass


class DiseaseProgressionKeys:
    ''' Define mapping to keys associated with disease progression '''
    modify_progression_by_age = 'prog_by_age'
    probability_of_infected_developing_symptoms = 'default_symp_prob'
    probability_of_symptoms_developing_severe = 'default_severe_prob'
    probability_of_severe_developing_death = 'default_death_prob'
    pass


class ResultsKeys:
    ''' Define keys for results '''
    cumulative_number_of_deaths = 'cum_deaths'
    pass


def define_base_parameters():
    ''' Define the basic parameters for a simulation -- these will rarely change between tests '''
    base_parameters_dict = {
        SimulationKeys.number_agents: 1000,
        SimulationKeys.initial_infected_count: 100,
        SimulationKeys.population_scaling_factor: 1,
        SimulationKeys.random_seed: 1,
        SimulationKeys.number_simulated_days: 60,
        }
    return base_parameters_dict


def BaseSim():
    ''' Create a base simulation to run tests on '''
    base_parameters_dict = define_base_parameters()
    base_sim = cv.Sim(pars=base_parameters_dict)
    return base_sim


class ExperimentalDiseaseMortalityTests(CovaSimTest):

    def test_zero_deaths(self):
        ''' Confirm that if mortality is set to zero, there are zero deaths '''
        sim = BaseSim() # Create the sim
        sim[DiseaseProgressionKeys.probability_of_severe_developing_death] = 0 # Change mortality rates to 0
        sim.run() # Run the simulation
        total_deaths = sim.results[ResultsKeys.cumulative_number_of_deaths][:][-1] # Get the total number of deaths (last value of the cumulative number)
        self.assertEqual(0, total_deaths,
                     msg="There should be no deaths"
                         "with {DiseaseProgressionKeys.probability_of_severe_developing_death} = 0."
                         "Channel {ResultsKeys.cumulative_number_of_deaths} had "
                         "bad data: {total_deaths}")


