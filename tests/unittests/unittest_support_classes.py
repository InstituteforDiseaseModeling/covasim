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

from covasim import Sim, parameters

class TestProperties:
    class ParameterKeys:
        class SimulationKeys:
            number_agents = 'n'
            population_scaling_factor = 'scale'
            initial_infected_count = 'n_infected'
            start_day = 'start_day'
            number_simulated_days = 'n_days'
            random_seed = 'seed'
            verbose = 'verbose'
            enable_synthpops = 'usepopdata'
            time_limit = 'timelimit'
            # stopping_function = 'stop_func'
            pass
        class TransmissionKeys:
            beta = 'beta'
            asymptomatic_fraction = 'asym_prop'
            asymptomatic_transmission_multiplier = 'asym_factor'
            diagnosis_transmission_factor = 'diag_factor'
            contact_transmission_factor = 'cont_factor'
            contacts_per_agent = 'contacts'
            beta_population_specific = 'beta_pop'
            contacts_population_specific = 'contacts_pop'
            pass
        class ProgressionKeys:
            exposed_to_infectious = 'serial'
            exposed_to_infectious_std = 'serial_std'
            exposed_to_symptomatic = 'incub'
            exposed_to_symptomatic_std = 'incub_std'
            infectiousness_duration = 'dur'
            infectiousness_duration_std = 'dur_std'
            pass
        class DiagnosticTestingKeys:
            number_daily_tests = 'daily_tests'
            daily_test_sensitivity = 'sensitivity'
            symptomatic_testing_multiplier = 'sympt_test'
            contacttrace_testing_multiplier = 'trace_test'
            pass
        class MortalityKeys:
            time_to_death = 'timetodie'
            time_to_death_std = 'timetodie_std'
            use_cfr_by_age = 'prog_by_age'
            prob_severe_death = 'default_death_prob'
            prob_symptomatic_severe = 'default_severe_prob'
            prob_infected_symptomatic = 'default_symp_prob'
            pass
        pass
    class SpecializedSimulations:
        class Microsim:
            n = 10
            n_infected = 1
            contacts = 2
            n_days = 10
            pass
        class Hightransmission:
            n = 500
            n_infected = 10
            n_days = 30
            contacts = 3
            beta = 0.4
            serial = 2
            serial_std = 0.5
            dur = 3
            pass
        class HighMortality:
            n = 1000
            cfr_by_age = False
            default_cfr = 0.2
            timetodie = 6
            timetodie_std = 2
        pass
    class ResultsDataKeys:
        deaths_cumulative = 'cum_deaths'
        deaths_daily = 'deaths'
        diagnoses_cumulative = 'cum_diagnosed'
        diagnoses_at_timestep = 'diagnoses'
        diagnostics_at_timestep = 'tests'
        diagnostics_cumulative = 'cum_tested'
        exposed_cumulative = 'cum_exposed'
        exposed_at_timestep = 'n_exposed'
        susceptible_at_timestep = 'n_susceptible'
        infectious_at_timestep = 'n_infectious'
        symptomatic_at_timestep = 'n_symptomatic'
        recovered_at_timestep = 'n_recovered'
        recovered_cumulative = 'cum_recoveries'
        recovered_at_timestep_WHAT = 'recoveries'
        infections_at_timestep = 'infections'
        GUESS_doubling_time_at_timestep = 'doubling_time'
        GUESS_r_effective_at_timestep = 'r_eff'

    pass

class CovaSimTest(unittest.TestCase):
    def setUp(self):
        self.is_debugging = False

        self.simulation_parameters = None
        self.sim = None
        self.simulation_result = None
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
            self.simulation_parameters = parameters.make_pars()
        if params_dict:
            self.simulation_parameters.update(params_dict)
        pass

    def run_sim(self, params_dict=None, write_results_json=True):
        if not self.simulation_parameters or params_dict: # If we need one, or have one here
            self.set_simulation_parameters(params_dict=params_dict)
            pass
        self.sim = Sim(pars=self.simulation_parameters,
                       datafile=None)
        self.sim.run(verbose=0)
        self.simulation_result = self.sim.to_json(tostring=False)
        if write_results_json:
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

    # region specialized simulation methods
    def set_microsim(self):
        Simkeys = TestProperties.ParameterKeys.SimulationKeys
        Transkeys = TestProperties.ParameterKeys.TransmissionKeys
        Micro = TestProperties.SpecializedSimulations.Microsim
        microsim_parameters = {
            Simkeys.number_agents : Micro.n,
            Simkeys.initial_infected_count: Micro.n_infected,
            Simkeys.number_simulated_days: Micro.n_days,
            Transkeys.contacts_per_agent: Micro.contacts
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

    def set_everyone_infectious_same_day(self, num_agents, days_to_infectious=1, num_days=60):
        """
        Args:
            num_agents: number of agents to create and infect
            days_to_infectious: days until all agents are infectious (1)
            num_days: days to simulate (60)
        """
        self.set_everyone_infected(agent_count=num_agents)
        test_config = {
            TestProperties.ParameterKeys.SimulationKeys.number_simulated_days: num_days,
            TestProperties.ParameterKeys.ProgressionKeys.exposed_to_infectious: days_to_infectious,
            TestProperties.ParameterKeys.ProgressionKeys.exposed_to_infectious_std: 0,
            TestProperties.ParameterKeys.MortalityKeys.use_cfr_by_age: False,
            TestProperties.ParameterKeys.MortalityKeys.prob_infected_symptomatic: 0
        }
        self.set_simulation_parameters(params_dict=test_config)
        pass

    def set_everyone_symptomatic(self, num_agents):
        """
        Cause all agents in the simulation to begin infected
        And proceed to symptomatic (but not severe or death)
        Args:
            num_agents: Number of agents to begin with
        """
        self.set_everyone_infectious_same_day(num_agents=num_agents,
                                              days_to_infectious=0)
        everyone_symptomatic_config = {
            TestProperties.ParameterKeys.MortalityKeys.prob_infected_symptomatic: 1.0,
            TestProperties.ParameterKeys.MortalityKeys.prob_symptomatic_severe: 0
        }
        self.set_simulation_parameters(params_dict=everyone_symptomatic_config)
        pass

    def set_everyone_is_going_to_die(self, num_agents):
        """
        Cause all agents in the simulation to begin infected and die.
        Args:
            num_agents: Number of agents to simulate
        """
        self.set_everyone_infectious_same_day(num_agents=num_agents)
        everyone_dying_config = {
            TestProperties.ParameterKeys.MortalityKeys.use_cfr_by_age: False,
            TestProperties.ParameterKeys.MortalityKeys.prob_infected_symptomatic: 1,  # Uh oh
            TestProperties.ParameterKeys.MortalityKeys.prob_symptomatic_severe: 1,  # Oh no
            TestProperties.ParameterKeys.MortalityKeys.prob_severe_death: 1,  # No recoveries
        }
        self.set_simulation_parameters(params_dict=everyone_dying_config)
        pass

    def set_smallpop_hightransmission(self):
        """
        Creates a small population with lots of transmission
        """
        Simkeys = TestProperties.ParameterKeys.SimulationKeys
        Transkeys = TestProperties.ParameterKeys.TransmissionKeys
        Progkeys = TestProperties.ParameterKeys.ProgressionKeys
        Hightrans = TestProperties.SpecializedSimulations.Hightransmission
        hightrans_parameters = {
            Simkeys.number_agents : Hightrans.n,
            Simkeys.initial_infected_count: Hightrans.n_infected,
            Simkeys.number_simulated_days: Hightrans.n_days,
            Transkeys.contacts_per_agent: Hightrans.contacts,
            Transkeys.beta : Hightrans.beta,
            Progkeys.exposed_to_infectious: Hightrans.serial,
            Progkeys.exposed_to_infectious_std: Hightrans.serial_std,
            Progkeys.infectiousness_duration: Hightrans.dur
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
        self.run_sim()
        json_file_found = os.path.isfile(self.expected_result_filename)
        self.assertTrue(json_file_found, msg=f"Expected {self.expected_result_filename} to be found.")
    pass

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
                         microsimParams.n_infected)
        pass

    def test_everyone_infected(self):
        """
        All agents start infected
        """
        self.is_debugging = True

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
        self.is_debugging = True
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



