import covasim as cv
from unittest_support_classes import CovaSimTest


class SimKeys:
    ''' Define mapping to simulation keys '''
    number_agents = 'n'
    initial_infected_count = 'n_infected'
    start_day = 'start_day'
    number_simulated_days = 'n_days'
    random_seed = 'seed'
    pass


class DiseaseKeys:
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
    ''' Define the basic parameters for a simulation -- these will sometimes, but rarely, change between tests '''
    base_parameters_dict = {
        SimKeys.number_agents: 1000, # Keep it small so they run faster
        SimKeys.initial_infected_count: 100, # Use a relatively large number to avoid stochastic effects
        SimKeys.random_seed: 1, # Ensure it's reproducible
        SimKeys.number_simulated_days: 60, # Don't run for too long for speed, but run for long enough
        }
    return base_parameters_dict


def BaseSim():
    ''' Create a base simulation to run tests on '''
    base_parameters_dict = define_base_parameters()
    base_sim = cv.Sim(pars=base_parameters_dict)
    return base_sim


class ExperimentalDiseaseMortalityTests(CovaSimTest):
    ''' Define the actual tests '''

    def test_zero_deaths(self):
        ''' Confirm that if mortality is set to zero, there are zero deaths '''

        # Create the sim
        sim = BaseSim()

        # Define test-secific configurations
        test_parameters = {
            DiseaseKeys.modify_progression_by_age: False, # Otherwise these parameters have no effect
            DiseaseKeys.probability_of_severe_developing_death: 0 # Change mortality rate to 0
            }

        # Run the simulation
        sim.update_pars(test_parameters)
        sim.run()

        # Check results
        total_deaths = sim.results[ResultsKeys.cumulative_number_of_deaths][:][-1] # Get the total number of deaths (last value of the cumulative number)
        self.assertEqual(0, total_deaths,
                     msg=f"There should be no deaths given parameters {test_parameters}. "
                         f"Channel {ResultsKeys.cumulative_number_of_deaths} had "
                         f"bad data: {total_deaths}")

        pass


    def test_full_deaths(self):
        ''' Confirm that if all progression parameters are set to 1, everyone dies'''

        # Create the sim
        sim = BaseSim()

        # Define test-secific configurations
        test_parameters = {
            SimKeys.initial_infected_count: sim[SimKeys.number_agents], # Ensure everyone is infected
            DiseaseKeys.modify_progression_by_age: False, # Otherwise these parameters have no effect
            DiseaseKeys.probability_of_infected_developing_symptoms: 1, # Everyone infected gets symptoms
            DiseaseKeys.probability_of_symptoms_developing_severe: 1, # Everyone with symptoms has severe disease
            DiseaseKeys.probability_of_severe_developing_death: 1, # Everyone with severe disease dies
        }

        # Run the simulation
        sim.update_pars(test_parameters)
        sim.run()

        # Check results
        total_deaths = sim.results[ResultsKeys.cumulative_number_of_deaths][:][-1] # Get the total number of deaths (last value of the cumulative number)
        self.assertEqual(sim[SimKeys.number_agents], total_deaths,
                     msg=f"Everyone should die with parameters {test_parameters}. "
                         f"Channel {ResultsKeys.cumulative_number_of_deaths} had "
                         f"bad data: {total_deaths} deaths vs. {sim[SimKeys.number_agents]} people.")

        pass


