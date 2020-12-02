"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest

from unittest_support_classes import CovaSimTest, TestProperties

ResKeys = TestProperties.ResultsDataKeys
ParamKeys = TestProperties.ParameterKeys


class DiseaseProgressionTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_exposure_to_infectiousness_delay_scaling(self):
        """
        Set exposure to infectiousness early simulation, mid simulation,
        late simulation. Set std_dev to zero. Verify move to infectiousness
        moves later as delay is longer.
        Depends on delay deviation test
        """
        total_agents = 500
        self.set_everyone_infected(total_agents)
        sim_dur = 60
        exposed_delays = [1, 2, 5, 15, 20, 25, 30]  # Keep values in order
        std_dev = 0
        for exposed_delay in exposed_delays:
            self.set_duration_distribution_parameters(
                duration_in_question=ParamKeys.ProgressionKeys.DurationKeys.exposed_to_infectious,
                par1=exposed_delay,
                par2=std_dev
            )
            prob_dict = {
                TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.inf_to_symptomatic_probability: 0
            }
            self.set_simulation_prognosis_probability(prob_dict)
            serial_delay = {
                TestProperties.ParameterKeys.SimulationKeys.number_simulated_days: sim_dur
            }
            self.run_sim(serial_delay)
            infectious_channel = self.get_full_result_channel(
                ResKeys.infectious_at_timestep
            )
            agents_on_infectious_day = infectious_channel[exposed_delay]
            if self.is_debugging:
                print(f"Delay: {exposed_delay}")
                print(f"Agents turned: {agents_on_infectious_day}")
                print(f"Infectious channel {infectious_channel}")
                pass
            for t in range(len(infectious_channel)):
                current_infectious = infectious_channel[t]
                if t < exposed_delay:
                    self.assertEqual(current_infectious, 0,
                                     msg=f"All {total_agents} should turn infectious at t: {exposed_delay}"
                                         f" instead got {current_infectious} at t: {t}")
                elif t == exposed_delay:
                    self.assertEqual(infectious_channel[exposed_delay], total_agents,
                                     msg=f"With stddev 0, all {total_agents} agents should turn infectious "
                                         f"on day {exposed_delay}, instead got {agents_on_infectious_day}. ")
        pass

    def test_mild_infection_duration_scaling(self):
        """
        Make sure that all initial infected cease being infected
        on following day. Std_dev 0 will help here
        """
        total_agents = 500
        exposed_delay = 1
        self.set_everyone_infectious_same_day(num_agents=total_agents,
                                              days_to_infectious=exposed_delay)
        prob_dict = {
            ParamKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.inf_to_symptomatic_probability: 0.0
        }
        self.set_simulation_prognosis_probability(prob_dict)
        infectious_durations = [1, 2, 5, 10, 20] # Keep values in order
        infectious_duration_stddev = 0
        for TEST_dur in infectious_durations:
            recovery_day = exposed_delay + TEST_dur
            self.set_duration_distribution_parameters(
                duration_in_question=ParamKeys.ProgressionKeys.DurationKeys.infectious_asymptomatic_to_recovered,
                par1=TEST_dur,
                par2=infectious_duration_stddev
            )
            self.run_sim()
            recoveries_channel = self.get_full_result_channel(
                TestProperties.ResultsDataKeys.recovered_at_timestep
            )
            recoveries_on_recovery_day = recoveries_channel[recovery_day]
            if self.is_debugging:
                print(f"Delay: {recovery_day}")
                print(f"Agents turned: {recoveries_on_recovery_day}")
                print(f"Recoveries channel {recoveries_channel}")
            self.assertEqual(recoveries_channel[recovery_day], total_agents,
                             msg=f"With stddev 0, all {total_agents} agents should turn infectious "
                                 f"on day {recovery_day}, instead got {recoveries_on_recovery_day}. ")

        pass

    def test_time_to_die_duration_scaling(self):
        total_agents = 500
        self.set_everyone_critical(num_agents=500, constant_delay=0)
        prob_dict = {
            ParamKeys.ProgressionKeys.ProbabilityKeys.RelativeProbKeys.crt_to_death_probability: 1.0
        }
        self.set_simulation_prognosis_probability(prob_dict)

        time_to_die_durations = [1, 2, 5, 10, 20]
        time_to_die_stddev = 0

        for TEST_dur in time_to_die_durations:
            self.set_duration_distribution_parameters(
                duration_in_question=ParamKeys.ProgressionKeys.DurationKeys.critical_to_death,
                par1=TEST_dur,
                par2=time_to_die_stddev
            )
            self.run_sim()
            deaths_today_channel = self.get_full_result_channel(
                TestProperties.ResultsDataKeys.deaths_daily
            )
            for t in range(len(deaths_today_channel)):
                curr_deaths = deaths_today_channel[t]
                if t < TEST_dur:
                    self.assertEqual(curr_deaths, 0,
                                     msg=f"With std 0, all {total_agents} agents should die on "
                                         f"t: {TEST_dur}. Got {curr_deaths} at t: {t}")
                elif t == TEST_dur:
                    self.assertEqual(curr_deaths, total_agents,
                                     msg=f"With std 0, all {total_agents} agents should die at t:"
                                         f" {TEST_dur}, got {curr_deaths} instead.")
                else:
                    self.assertEqual(curr_deaths, 0,
                                     msg=f"With std 0, all {total_agents} agents should die at t:"
                                         f" {TEST_dur}, got {curr_deaths} at t: {t}")
        pass

if __name__ == '__main__':
    unittest.main()