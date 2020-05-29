"""
Classes that provide test content for tests later,
and easier configuration of tests to make tests
easier to red.

Test implementation is agnostic to model implementation
by design.
"""

import unittest
import json
import os
import numpy as np

from covasim import Sim, parameters, change_beta, test_prob, contact_tracing, sequence



class TestProperties:
    class ParameterKeys:
        class SimulationKeys:
            number_agents = 'pop_size'
            number_contacts = 'contacts'
            population_scaling_factor = 'pop_scale'
            population_rescaling = 'rescale'
            population_type = 'pop_type'
            initial_infected_count = 'pop_infected'
            start_day = 'start_day'
            number_simulated_days = 'n_days'
            random_seed = 'rand_seed'
            verbose = 'verbose'
            enable_synthpops = 'usepopdata'
            time_limit = 'timelimit'
            quarantine_effectiveness = 'quar_factor'
            # stopping_function = 'stop_func'
            pass

        class TransmissionKeys:
            beta = 'beta'
            asymptomatic_fraction = 'asym_prop'
            asymptomatic_transmission_multiplier = 'asym_factor'
            diagnosis_transmission_factor = 'iso_factor'
            contact_transmission_factor = 'cont_factor'
            contacts_per_agent = 'contacts'  # TODO: need to construct a population now
            beta_population_specific = 'beta_pop'
            contacts_population_specific = 'contacts_pop'
            pass

        class ProgressionKeys:
            durations = "dur"
            param_1 = "par1"
            param_2 = "par2"

            class DurationKeys:
                exposed_to_infectious = 'exp2inf'
                infectious_to_symptomatic = 'inf2sym'
                infectious_asymptomatic_to_recovered = 'asym2rec'
                infectious_symptomatic_to_recovered = 'mild2rec'
                symptomatic_to_severe = 'sym2sev'
                severe_to_critical = 'sev2crit'
                aymptomatic_to_recovered = 'asym2rec'
                severe_to_recovered = 'sev2rec'
                critical_to_recovered = 'crit2rec'
                critical_to_death = 'crit2die'
                pass

            class ProbabilityKeys:
                progression_by_age = 'prog_by_age'
                class RelativeProbKeys:
                    inf_to_symptomatic_probability = 'rel_symp_prob'
                    sym_to_severe_probability = 'rel_severe_prob'
                    sev_to_critical_probability = 'rel_crit_prob'
                    crt_to_death_probability = 'rel_death_prob'
                    pass
                class PrognosesListKeys:
                    symptomatic_probabilities = 'symp_probs'
                    severe_probabilities = 'severe_probs'
                    critical_probabilities = 'crit_probs'
                    death_probs = 'death_probs'
            pass

        class DiagnosticTestingKeys:
            number_daily_tests = 'daily_tests'
            daily_test_sensitivity = 'sensitivity'
            symptomatic_testing_multiplier = 'sympt_test'
            contacttrace_testing_multiplier = 'trace_test'
            pass
        pass

    class SpecializedSimulations:
        class Microsim:
            n = 10
            pop_infected = 1
            contacts = 2
            n_days = 10
            pass
        class Hightransmission:
            n = 500
            pop_infected = 10
            n_days = 30
            contacts = 3
            beta = 0.4
            serial = 2
            # serial_std = 0.5
            dur = 3
            pass
        class HighMortality:
            n = 1000
            cfr_by_age = False
            default_cfr = 0.2
            timetodie = 6
            # timetodie_std = 2
        pass

    class ResultsDataKeys:
        deaths_cumulative = 'cum_deaths'
        deaths_daily = 'new_deaths'
        diagnoses_cumulative = 'cum_diagnoses'
        diagnoses_at_timestep = 'new_diagnoses'
        exposed_at_timestep = 'n_exposed'
        susceptible_at_timestep = 'n_susceptible'
        infectious_at_timestep = 'n_infectious'
        symptomatic_at_timestep = 'n_symptomatic'
        symptomatic_cumulative = 'cum_symptomatic'
        symptomatic_new_timestep = 'new_symptomatic'
        recovered_at_timestep = 'new_recoveries'
        recovered_cumulative = 'cum_recoveries'
        infections_at_timestep = 'new_infections'
        infections_cumulative = 'cum_infections'
        tests_at_timestep = 'new_tests'
        tests_cumulative = 'cum_tests'
        quarantined_new = 'new_quarantined'
        GUESS_doubling_time_at_timestep = 'doubling_time'
        GUESS_r_effective_at_timestep = 'r_eff'

    pass


DurationKeys = TestProperties.ParameterKeys.ProgressionKeys.DurationKeys


class CovaSimTest(unittest.TestCase):
    def setUp(self):
        self.is_debugging = False

        self.simulation_parameters = None
        self.simulation_prognoses = None
        self.sim = None
        self.simulation_result = None
        self.interventions = None
        self.expected_result_filename = f"DEBUG_{self.id()}.json"
        if os.path.isfile(self.expected_result_filename):
            os.unlink(self.expected_result_filename)
        pass

    def tearDown(self):
        if not self.is_debugging:
            if os.path.isfile(self.expected_result_filename):
                os.unlink(self.expected_result_filename)
        pass

    # region configuration methods
    def set_simulation_parameters(self, params_dict=None):
        """
        Overrides all of the default sim parameters
        with the ones in the dictionary
        Args:
            params_dict: keys are param names, values are expected values to use

        Returns:
            None, sets self.simulation_params

        """
        if not self.simulation_parameters:
            self.simulation_parameters = parameters.make_pars(set_prognoses=True, prog_by_age=True)
        if params_dict:
            self.simulation_parameters.update(params_dict)
        pass

    def set_simulation_prognosis_probability(self, params_dict):
        """
        Allows for testing prognoses probability as absolute rather than relative.
        NOTE: You can only call this once per test or you will overwrite your stuff.
        """
        ProbKeys = TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys
        RelativeProbabilityKeys = ProbKeys.RelativeProbKeys
        supported_probabilities = [
            RelativeProbabilityKeys.inf_to_symptomatic_probability,
            RelativeProbabilityKeys.sym_to_severe_probability,
            RelativeProbabilityKeys.sev_to_critical_probability,
            RelativeProbabilityKeys.crt_to_death_probability
        ]
        if not self.simulation_parameters:
            self.set_simulation_parameters()
            pass

        if not self.simulation_prognoses:
            self.simulation_prognoses = parameters.get_prognoses(self.simulation_parameters[ProbKeys.progression_by_age])

        PrognosisKeys = ProbKeys.PrognosesListKeys
        for k in params_dict:
            prognosis_in_question = None
            expected_prob = params_dict[k]
            if k == RelativeProbabilityKeys.inf_to_symptomatic_probability:
                prognosis_in_question = PrognosisKeys.symptomatic_probabilities
            elif k == RelativeProbabilityKeys.sym_to_severe_probability:
                prognosis_in_question = PrognosisKeys.severe_probabilities
            elif k == RelativeProbabilityKeys.sev_to_critical_probability:
                prognosis_in_question = PrognosisKeys.critical_probabilities
            elif k == RelativeProbabilityKeys.crt_to_death_probability:
                prognosis_in_question = PrognosisKeys.death_probs
            else:
                raise KeyError(f"Key {k} not found in {supported_probabilities}.")
            old_probs = self.simulation_prognoses[prognosis_in_question]
            self.simulation_prognoses[prognosis_in_question] = np.array([expected_prob] * len(old_probs))
            pass
        pass

    def set_duration_distribution_parameters(self, duration_in_question,
                                             par1, par2):
        if not self.simulation_parameters:
            self.set_simulation_parameters()
            pass
        duration_node = self.simulation_parameters["dur"]
        duration_node[duration_in_question] = {
            "dist": "normal",
            "par1": par1,
            "par2": par2
        }
        params_dict = {
            "dur": duration_node
        }
        self.set_simulation_parameters(params_dict=params_dict)


    def run_sim(self, params_dict=None, write_results_json=False, population_type=None):
        if not self.simulation_parameters or params_dict: # If we need one, or have one here
            self.set_simulation_parameters(params_dict=params_dict)
            pass

        self.simulation_parameters['interventions'] = self.interventions

        self.sim = Sim(pars=self.simulation_parameters,
                       datafile=None)
        if not self.simulation_prognoses:
            self.simulation_prognoses = parameters.get_prognoses(
                self.simulation_parameters[TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.progression_by_age]
            )
            pass

        self.sim['prognoses'] = self.simulation_prognoses
        if population_type:
            self.sim.update_pars(pop_type=population_type)
        self.sim.run(verbose=0)
        self.simulation_result = self.sim.to_json(tostring=False)
        if write_results_json or self.is_debugging:
            with open(self.expected_result_filename, 'w') as outfile:
                json.dump(self.simulation_result, outfile, indent=4, sort_keys=True)
        pass
    # endregion

    # region simulation results support
    def get_full_result_channel(self, channel):
        result_data = self.simulation_result["results"][channel]
        return result_data

    def get_day_zero_channel_value(self, channel=TestProperties.ResultsDataKeys.susceptible_at_timestep):
        """

        Args:
            channel: timeseries channel to report ('n_susceptible')

        Returns: day zero value for channel

        """
        result_data = self.get_full_result_channel(channel=channel)
        return result_data[0]

    def get_day_final_channel_value(self, channel):
        channel = self.get_full_result_channel(channel=channel)
        return channel[-1]
    # endregion

    # region interventions support
    def intervention_set_changebeta(self,
                                    days_array,
                                    multiplier_array,
                                    layers = None):
        self.interventions = change_beta(days=days_array,
                                         changes=multiplier_array,
                                         layers=layers)
        pass

    def intervention_set_test_prob(self, symptomatic_prob=0, asymptomatic_prob=0,
                                   asymptomatic_quarantine_prob=0, symp_quar_prob=0,
                                   test_sensitivity=1.0, loss_prob=0.0, test_delay=1,
                                   start_day=0):
        self.interventions = test_prob(symp_prob=symptomatic_prob,
                                       asymp_prob=asymptomatic_prob,
                                       asymp_quar_prob=asymptomatic_quarantine_prob,
                                       symp_quar_prob=symp_quar_prob,
                                       test_sensitivity=test_sensitivity,
                                       loss_prob=loss_prob,
                                       test_delay=test_delay,
                                       start_day=start_day)
        pass

    def intervention_set_contact_tracing(self,
                                         start_day,
                                         trace_probabilities=None,
                                         trace_times=None):
        #  see ../tests/test_interventions_testing.test_tracedelay
        #  trace_probs = {'h': 1, 's': 1, 'w': 1, 'c': 1}
        #  trace_time = {'h': 0, 's': 1, 'w': 1, 'c': 2}
        #  pars.quar_period = 60 # 60 days
        if not trace_probabilities:
            trace_probabilities = {'h': 1, 's': 1, 'w': 1, 'c': 1}
            pass
        if not trace_times:
            trace_times = {'h': 1, 's': 1, 'w': 1, 'c': 1}
        self.interventions = contact_tracing(trace_probs=trace_probabilities,
                                             trace_time=trace_times,
                                             start_day=start_day)
        pass

    def intervention_build_sequence(self,
                                    day_list,
                                    intervention_list):
        my_sequence = sequence(days=day_list,
                               interventions=intervention_list)
        self.interventions = my_sequence
    # endregion

    # region specialized simulation methods
    def set_microsim(self):
        Simkeys = TestProperties.ParameterKeys.SimulationKeys
        Micro = TestProperties.SpecializedSimulations.Microsim
        microsim_parameters = {
            Simkeys.number_agents : Micro.n,
            Simkeys.initial_infected_count: Micro.pop_infected,
            Simkeys.number_simulated_days: Micro.n_days
        }
        self.set_simulation_parameters(microsim_parameters)
        pass

    def set_everyone_infected(self, agent_count=1000):
        Simkeys = TestProperties.ParameterKeys.SimulationKeys
        everyone_infected = {
            Simkeys.number_agents: agent_count,
            Simkeys.initial_infected_count: agent_count
        }
        self.set_simulation_parameters(params_dict=everyone_infected)
        pass

    DurationKeys = TestProperties.ParameterKeys.ProgressionKeys.DurationKeys

    def set_everyone_infectious_same_day(self, num_agents, days_to_infectious=1, num_days=60):
        """
        Args:
            num_agents: number of agents to create and infect
            days_to_infectious: days until all agents are infectious (1)
            num_days: days to simulate (60)
        """
        self.set_everyone_infected(agent_count=num_agents)
        prob_dict = {
            TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.inf_to_symptomatic_probability: 0
        }
        self.set_simulation_prognosis_probability(prob_dict)
        test_config = {
            TestProperties.ParameterKeys.SimulationKeys.number_simulated_days: num_days
        }
        self.set_duration_distribution_parameters(
            duration_in_question=DurationKeys.exposed_to_infectious,
            par1=days_to_infectious,
            par2=0
        )
        self.set_simulation_parameters(params_dict=test_config)
        pass

    def set_everyone_symptomatic(self, num_agents, constant_delay:int=None):
        """
        Cause all agents in the simulation to begin infected
        And proceed to symptomatic (but not severe or death)
        Args:
            num_agents: Number of agents to begin with
        """
        self.set_everyone_infectious_same_day(num_agents=num_agents,
                                              days_to_infectious=0)
        prob_dict = {
            TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.inf_to_symptomatic_probability: 1.0,
            TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.sym_to_severe_probability: 0
        }
        self.set_simulation_prognosis_probability(prob_dict)
        if constant_delay is not None:
            self.set_duration_distribution_parameters(
                duration_in_question=DurationKeys.infectious_to_symptomatic,
                par1=constant_delay,
                par2=0
            )
        pass

    def set_everyone_is_going_to_die(self, num_agents):
        """
        Cause all agents in the simulation to begin infected and die.
        Args:
            num_agents: Number of agents to simulate
        """
        ProbKeys = TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys
        self.set_everyone_infectious_same_day(num_agents=num_agents)
        prob_dict = {
            ProbKeys.inf_to_symptomatic_probability: 1,
            ProbKeys.sym_to_severe_probability: 1,
            ProbKeys.sev_to_critical_probability: 1,
            ProbKeys.crt_to_death_probability: 1
        }
        self.set_simulation_prognosis_probability(prob_dict)
        pass

    def set_everyone_severe(self, num_agents, constant_delay:int=None):
        self.set_everyone_symptomatic(num_agents=num_agents, constant_delay=constant_delay)
        prob_dict = {
            TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.sym_to_severe_probability: 1.0,
            TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.sev_to_critical_probability: 0.0
        }
        self.set_simulation_prognosis_probability(prob_dict)
        if constant_delay is not None:
            self.set_duration_distribution_parameters(
                duration_in_question=DurationKeys.symptomatic_to_severe,
                par1=constant_delay,
                par2=0
            )
        pass

    def set_everyone_critical(self, num_agents, constant_delay:int=None):
        """
        Causes all agents to become critically ill day 1
        """
        self.set_everyone_severe(num_agents=num_agents, constant_delay=constant_delay)
        prob_dict = {
            TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.sev_to_critical_probability: 1.0,
            TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.crt_to_death_probability: 0.0
        }
        self.set_simulation_prognosis_probability(prob_dict)
        if constant_delay is not None:
            self.set_duration_distribution_parameters(
                duration_in_question=DurationKeys.severe_to_critical,
                par1=constant_delay,
                par2=0
            )
        pass


    def set_smallpop_hightransmission(self):
        """
        Creates a small population with lots of transmission
        """
        Simkeys = TestProperties.ParameterKeys.SimulationKeys
        Transkeys = TestProperties.ParameterKeys.TransmissionKeys
        Hightrans = TestProperties.SpecializedSimulations.Hightransmission
        hightrans_parameters = {
            Simkeys.number_agents : Hightrans.n,
            Simkeys.initial_infected_count: Hightrans.pop_infected,
            Simkeys.number_simulated_days: Hightrans.n_days,
            Transkeys.beta : Hightrans.beta
        }
        self.set_simulation_parameters(hightrans_parameters)
        pass

    # endregion
    pass




class TestSupportTests(CovaSimTest):
    def test_run_vanilla_simulation(self):
        """
        Runs an uninteresting but predictable
        simulation, makes sure that results
        are created and json parsable
        """
        self.assertIsNone(self.sim)
        self.run_sim(write_results_json=True)
        json_file_found = os.path.isfile(self.expected_result_filename)
        self.assertTrue(json_file_found, msg=f"Expected {self.expected_result_filename} to be found.")
    pass

    @unittest.skip("Cannot easily specify number contacts to be small, need to construct a population")
    def test_run_microsim(self):
        """
        Runs a super short simulation
        Verifies that the microsim parameters were created and honored
        """
        self.assertIsNone(self.simulation_parameters)
        self.assertIsNone(self.sim)
        self.set_microsim()
        self.run_sim()
        result_data = self.simulation_result["results"]
        resultKeys = TestProperties.ResultsDataKeys
        microsimParams = TestProperties.SpecializedSimulations.Microsim
        self.assertEqual(len(result_data[resultKeys.recovered_at_timestep]),
                         microsimParams.n + 1)
        self.assertEqual(result_data[resultKeys.exposed_at_timestep][0],
                         microsimParams.pop_infected)
        pass

    def test_everyone_infected(self):
        """
        All agents start infected
        """

        total_agents = 500
        self.set_everyone_infected(agent_count=total_agents)
        self.run_sim()
        exposed_channel = TestProperties.ResultsDataKeys.exposed_at_timestep
        day_0_exposed = self.get_day_zero_channel_value(exposed_channel)
        self.assertEqual(day_0_exposed, total_agents)
        pass

    def test_run_small_hightransmission_sim(self):
        """
        Runs a small simulation with lots of transmission
        Verifies that there are lots of infections in
        a short time.
        """
        self.assertIsNone(self.simulation_parameters)
        self.assertIsNone(self.sim)
        self.set_smallpop_hightransmission()
        self.run_sim()

        self.assertIsNotNone(self.sim)
        self.assertIsNotNone(self.simulation_parameters)
        exposed_today_channel = self.get_full_result_channel(
            TestProperties.ResultsDataKeys.exposed_at_timestep
        )
        prev_exposed = exposed_today_channel[0]
        for t in range(1, 10):
            today_exposed = exposed_today_channel[t]
            self.assertGreaterEqual(today_exposed, prev_exposed,
                                    msg=f"The first 10 days should have increasing"
                                        f" exposure counts. At time {t}: {today_exposed} at"
                                        f" {t-1}: {prev_exposed}.")
            prev_exposed = today_exposed
            pass
        infections_channel = self.get_full_result_channel(
            TestProperties.ResultsDataKeys.infections_at_timestep
        )
        self.assertGreaterEqual(sum(infections_channel), 150,
                                msg=f"Should have at least 150 infections")
        pass
    pass



