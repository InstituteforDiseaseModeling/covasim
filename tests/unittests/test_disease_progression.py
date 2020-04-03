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

    @unittest.skip("Not fixing deviation tests now")
    def test_exposure_to_infectiousness_delay_deviation_scaling(self):
        """
        Configure exposure to infectiousness delay to 1/2 sim
        length, and std_dev to 0. Verify that every n_infected
        at start goes infectious on the same day
        """
        total_agents = 500
        self.set_everyone_infected(total_agents)
        sim_dur = 60
        exposed_to_infectious_delay = 30
        std_devs = [0, .5, 1, 2, 4]  # Keep values in order

        prev_first_day = None
        prev_peak_val = None
        prev_last_day = None
        prev_stddev = None

        for TEST_stddev in std_devs:

            serial_dev_zero = {
                TestProperties.ParameterKeys.SimulationKeys.number_simulated_days: sim_dur,
                TestProperties.ParameterKeys.ProgressionKeys.exposed_to_infectious: exposed_to_infectious_delay,
                TestProperties.ParameterKeys.ProgressionKeys.exposed_to_infectious_std: TEST_stddev,
                TestProperties.ParameterKeys.MortalityKeys.use_cfr_by_age: False,
                TestProperties.ParameterKeys.MortalityKeys.prob_infected_symptomatic: 0
            }
            self.run_sim(serial_dev_zero)
            infectious_channel = self.get_full_result_channel(
                ResKeys.infectious_at_timestep
            )
            if TEST_stddev == 0:
                for t in range(0, exposed_to_infectious_delay):
                    today_infectious = infectious_channel[t]
                    self.assertEqual(today_infectious, 0,
                                     msg=f"With std_dev 0, there should be no infectious"
                                         f"prior to {exposed_to_infectious_delay} + 1. At {t} we had"
                                         f" {today_infectious}.")
                    pass
                prev_infectious = infectious_channel[exposed_to_infectious_delay]
                self.assertEqual(prev_infectious, total_agents,
                                 msg=f"At {exposed_to_infectious_delay} + 1, should have {total_agents}"
                                     f" infectious individuals with std_dev 0. Got {prev_infectious}.")
                prev_first_day = exposed_to_infectious_delay  # were zero infectious before this
                prev_peak_val = total_agents  # this is everyone
                prev_last_day = exposed_to_infectious_delay  # and there is nobody left
                for t in range(exposed_to_infectious_delay + 1, len(infectious_channel)):
                    today_infectious = infectious_channel[t]
                    self.assertLessEqual(today_infectious, prev_infectious,
                                         msg="With std_dev 0, after initial infectious conversion,"
                                             " infectiousness should only decline. Not the case"
                                             f" at t: {t}")
                    pass
                prev_stddev = TEST_stddev
            else:
                curr_first_day = None
                curr_peak_day = None
                curr_peak_val = 0
                curr_last_day = None
                for t in range(len(infectious_channel)):
                    today_infectious = infectious_channel[t]
                    if today_infectious > 0:
                        if not curr_first_day:
                            curr_first_day = t
                            curr_peak_day = t
                            curr_peak_val = today_infectious
                        elif today_infectious > curr_peak_val:
                            curr_peak_day = t
                            curr_peak_val = today_infectious
                        else:
                            curr_last_day = t
                            pass
                        pass
                    pass
                if self.is_debugging:
                    print(f"TEST_stddev: {TEST_stddev}")
                    print(f"current first day: {curr_first_day}")
                    print(f"current peak day: {curr_peak_day}, curr_peak_val: {curr_peak_val}")
                    print(f"current last day: {curr_last_day}")
                    print(f"infectious_channel: {infectious_channel}")
                self.assertLessEqual(curr_first_day, prev_first_day,
                                msg=f"With stddev {TEST_stddev}, first infectious day {curr_first_day}"
                                    f" should be lower than previous {prev_first_day}"
                                    f" that came with stddev {prev_stddev}.")
                self.assertLess(curr_peak_val, prev_peak_val,
                                 msg=f"With stddev {TEST_stddev}, the peak value {curr_peak_val}"
                                     f" from peak day {curr_peak_day} should be less than the previous"
                                     f" peak value {prev_peak_val} from stddev {prev_stddev}.")
                self.assertGreaterEqual(curr_last_day, prev_last_day,
                                   msg=f"With stddev {TEST_stddev}, last infectious conversion {curr_last_day}"
                                       f" should be greater than previous {prev_last_day}"
                                       f" that came with stddev {prev_stddev}.")
                # Having passed all assertions, reset expectations
                prev_first_day = curr_first_day
                prev_peak_val = curr_peak_val
                prev_last_day = curr_last_day
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
            serial_delay = {
                TestProperties.ParameterKeys.SimulationKeys.number_simulated_days: sim_dur,
                TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.use_progression_by_age: False,
                TestProperties.ParameterKeys.ProgressionKeys.ProbabilityKeys.inf_to_symptomatic_probability: 0
            }
            self.run_sim(serial_delay)
            infectious_channel = self.get_full_result_channel(
                ResKeys.infectious_at_timestep
            )
            prev_infectious = None
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
                    prev_infectious = current_infectious
        pass

    @unittest.skip("Not fixing deviation tests now")
    def test_infection_duration_deviation_scaling(self):
        """
        Like deviation zero test, but with expected duration midsim.
        Verification like the exposure_to_infectiouness_delay
        deviation scaling test (as std_dev goes up, first conversion
        earlier and last conversion later)
        """
        total_agents = 500
        exposed_delay = 1
        self.set_everyone_infectious_same_day(num_agents=total_agents,
                                              days_to_infectious=exposed_delay)
        infectious_duration = 30
        infectious_duration_stddevs = [0, 1, 2, 4]  # Keep values in order
        for TEST_std in infectious_duration_stddevs:
            test_config = {
                TestProperties.ParameterKeys.ProgressionKeys.infectiousness_duration: infectious_duration,
                TestProperties.ParameterKeys.ProgressionKeys.infectiousness_duration_std: TEST_std
            }
            self.run_sim(params_dict=test_config)
            recoveries_channel = self.get_full_result_channel(
                TestProperties.ResultsDataKeys.recovered_at_timestep_WHAT
            )
            if TEST_std == 0:
                # do the zero stuff
                everyone_recovers_day = infectious_duration + exposed_delay
                for t in range(exposed_delay, everyone_recovers_day):
                    today_recoveries = recoveries_channel[t]
                    self.assertEqual(today_recoveries, 0,
                                     msg=f"With std_dev 0, there should be no infectious"
                                         f"prior to {everyone_recovers_day}. At {t} we had"
                                         f" {today_recoveries}.")
                    pass
                prev_recoveries = recoveries_channel[infectious_duration + exposed_delay]
                self.assertEqual(prev_recoveries, total_agents,
                                 msg=f"At {everyone_recovers_day}, should have {total_agents}"
                                     f" recovered individuals with std_dev 0. Got {prev_recoveries}.")
                prev_first_day = everyone_recovers_day # were zero before this
                prev_peak_val = total_agents
                prev_last_day = everyone_recovers_day # and there is nobody left
                for t in range(everyone_recovers_day + 1, len(recoveries_channel)):
                    today_recoveries = recoveries_channel[t]
                    self.assertEqual(today_recoveries, 0,
                                     msg="With std_dev 0, after initial recovery day,"
                                         " recoveries should remain 0. Not the case"
                                         f" at t: {t} with recoveries: {today_recoveries}")
                    pass
                prev_stddev = TEST_std
                pass
            else:
                curr_first_day = None
                curr_peak_day = None
                curr_peak_val = 0
                curr_last_day = 0
                for t in range(infectious_duration, len(recoveries_channel)):
                    today_recoveries = recoveries_channel[t]
                    if today_recoveries > 0:
                        if not curr_first_day:
                            curr_first_day = t
                        elif today_recoveries > curr_peak_val:
                            curr_peak_day = t
                            curr_peak_val = today_recoveries
                        else:
                            curr_last_day = t
                            pass
                        pass
                    pass

                if self.is_debugging:
                    print(f"TEST_stddev: {TEST_std}")
                    print(f"current first day: {curr_first_day}")
                    print(f"current peak day: {curr_peak_day}, curr_peak_val: {curr_peak_val}")
                    print(f"current last day: {curr_last_day}")
                    print(f"recoveries_channel: {recoveries_channel}")
                self.assertLessEqual(curr_first_day, prev_first_day,
                                msg=f"With stddev {TEST_std}, first infectious day {curr_first_day}"
                                    f" should be lower than previous {prev_first_day}"
                                    f" that came with stddev {prev_stddev}.")
                self.assertLess(curr_peak_val, prev_peak_val,
                                 msg=f"With stddev {TEST_std}, the peak value {curr_peak_val}"
                                     f"from peak day {curr_peak_day} should be less than the previous"
                                     f"peak value {prev_peak_val} from stddev {prev_stddev}.")
                self.assertGreaterEqual(curr_last_day, prev_last_day,
                                   msg=f"With stddev {TEST_std}, last infectious conversion {curr_last_day}"
                                       f" should be greater than previous {prev_last_day}"
                                       f" that came with stddev {prev_stddev}.")
                pass
            pass
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
        only_mild_infections = {
            ParamKeys.ProgressionKeys.ProbabilityKeys.inf_to_symptomatic_probability: 0.0
        }
        self.set_simulation_parameters()
        infectious_durations = [1, 2, 5, 10, 20] # Keep values in order
        infectious_duration_stddev = 0
        for TEST_dur in infectious_durations:
            recovery_day = exposed_delay + TEST_dur
            self.set_duration_distribution_parameters(
                duration_in_question=ParamKeys.ProgressionKeys.DurationKeys.infectious_asymptomatic_to_recovered,
                par1=TEST_dur,
                par2=infectious_duration_stddev
            )
            self.run_sim(params_dict=only_mild_infections)
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

    @unittest.skip("Not fixing deviation tests now")
    def test_time_to_die_deviation_scaling(self):
        total_agents = 500
        self.set_everyone_is_going_to_die(num_agents=total_agents)

        time_to_die_duration = 30
        time_to_die_duration_stddevs = [0, 1, 2, 4]

        prev_first_death_day = None
        prev_peak_death_count = 0
        prev_last_death_day = None

        for TEST_std in time_to_die_duration_stddevs:
            test_config = {
                TestProperties.ParameterKeys.MortalityKeys.time_to_death: time_to_die_duration,
                TestProperties.ParameterKeys.MortalityKeys.time_to_death_std: TEST_std
            }
            self.run_sim(params_dict=test_config)
            deaths_today_channel = self.get_full_result_channel(
                TestProperties.ResultsDataKeys.deaths_daily
            )
            death_timer_starts = 0
            if TEST_std == 0:
                expected_death_day = death_timer_starts + time_to_die_duration
                for t in range(death_timer_starts, expected_death_day):
                    curr_deaths = deaths_today_channel[t]
                    self.assertEqual(curr_deaths, 0,
                                     msg=f"With std 0, expected no deaths until {expected_death_day},"
                                         f" but saw {curr_deaths} at time {t}.")
                    pass
                curr_deaths = deaths_today_channel[expected_death_day]
                self.assertEqual(curr_deaths, total_agents,
                                 msg=f"With std 0, expected {total_agents} deaths on"
                                     f" day {t}  but saw {curr_deaths} instead.")
                prev_first_death_day = expected_death_day # Last is curr day if rest were zero
                prev_peak_death_count = curr_deaths # this is everybody
                for t in range(expected_death_day + 1, len(deaths_today_channel)):
                    curr_deaths = deaths_today_channel[t]
                    self.assertEqual(curr_deaths, 0,
                                     msg=f"With std 0, expected no deaths after {expected_death_day},"
                                         f" but saw {curr_deaths} at time {t}.")
                    pass
                prev_last_death_day = prev_first_death_day # Last same as first if rest are 0
                pass
            else:
                curr_first_death_day = None
                curr_peak_death_count = 0
                curr_last_death_day = None

                for t in range(death_timer_starts, len(deaths_today_channel)):
                    today_deaths = deaths_today_channel[t]
                    if today_deaths > 0:
                        if not curr_first_death_day:
                            curr_first_death_day = t
                        elif today_deaths > curr_peak_death_count:
                            curr_peak_death_count = today_deaths
                        else:
                            curr_last_death_day = t
                            pass
                        pass
                    pass
                self.assertLessEqual(curr_first_death_day, prev_first_death_day)
                self.assertLess(curr_peak_death_count, prev_peak_death_count)
                self.assertGreaterEqual(curr_last_death_day, prev_last_death_day)
                prev_first_death_day = curr_first_death_day
                prev_peak_death_count = curr_peak_death_count
                prev_last_death_day = curr_last_death_day
                pass
        pass

    def test_time_to_die_duration_scaling(self):
        total_agents = 500
        self.set_everyone_critical(num_agents=500, constant_delay=0)
        all_critical_to_die = {
            ParamKeys.ProgressionKeys.ProbabilityKeys.crt_to_death_probability: 1.0
        }

        time_to_die_durations = [1, 2, 5, 10, 20]
        time_to_die_stddev = 0

        for TEST_dur in time_to_die_durations:
            self.set_duration_distribution_parameters(
                duration_in_question=ParamKeys.ProgressionKeys.DurationKeys.critical_to_death,
                par1=TEST_dur,
                par2=time_to_die_stddev
            )
            self.run_sim(params_dict=all_critical_to_die)
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

    @unittest.skip("NYI")
    def test_exposed_to_symptomatic_delay_deviation_scaling(self):
        total_agents = 500
        self.set_everyone_symptomatic(num_agents=total_agents)

        incubation_duration = 30
        incubation_duration_stds = [0, 1, 2, 4]
        prev_first_day_sympto = None
        peak_sympto_day_value = 0
        prev_last_day_sympto = 0
        for TEST_std in incubation_duration_stds:
            test_config = {
                TestProperties.ParameterKeys.ProgressionKeys.exposed_to_symptomatic: incubation_duration,
                TestProperties.ParameterKeys.ProgressionKeys.exposed_to_symptomatic_std: TEST_std
            }
            self.run_sim(params_dict=test_config)
            symptomatic_count_channel = self.get_full_result_channel(
                TestProperties.ResultsDataKeys.symptomatic_at_timestep
            )
            if TEST_std == 0:
                for t in range(1, len(symptomatic_count_channel)):
                    # get the people sympto at current ts, subtract from previous
                    # if that number is zero...
                    # proceed as normal / above
                    pass
            else:
                pass
            pass
        pass
    pass

if __name__ == '__main__':
    unittest.main()