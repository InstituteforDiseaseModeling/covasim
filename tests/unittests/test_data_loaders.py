import unittest

from covasim.data import loaders


class TestDataLoaders(unittest.TestCase):
    def test_country_households(self):
        ch = loaders.get_country_household_sizes('un_household_size_sample.xlsx')

        self.assertTrue(isinstance(ch, dict))
        self.assertGreater(len(ch), 5)

        self.assertTrue(ch['United States of America'] == ch['USA'])
        self.assertTrue(ch['Republic of Korea'] == ch['Korea'])
        self.assertTrue(ch['Republic of Korea'] == ch['South Korea'])

        self.assertTrue(2 <= ch['USA'] <= 5)
        self.assertTrue(1 <= ch['Korea'] <= 3)
        self.assertTrue(1 <= ch['Germany'] <= 3)
        self.assertTrue(5 <= ch['Senegal'] <= 10)


if __name__ == '__main__':
    unittest.main()
