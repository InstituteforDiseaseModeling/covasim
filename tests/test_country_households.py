import sys
import tempfile
from pathlib import Path

import pandas as pd
import sciris as sc

from covasim.data import loaders


def validate_country_households_range():
    """
    Validate household size is withing the expected range.
    """
    print('Loading data...')
    ch = loaders.get_country_household_sizes()

    assert 2 <= ch['USA'] <= 5
    assert 1 <= ch['Korea'] <= 3
    assert 1 <= ch['South Korea'] <= 3

    for c in ch:
        assert 0 < ch[c] < 10

    print('Household size values are in the expected range.')


def plot_validate_country_households():
    """
    Plot to help spot if some countries have unexpected values.
    """
    # Convert to dataframe for plotting.
    ch = loaders.get_country_household_sizes()
    df = pd.DataFrame.from_dict(ch, orient='index', columns=['size']).reset_index()
    df['size'] = df['size'].astype(float)
    df.columns = ['country', 'size']
    df = df.sort_values(by=['size'], ascending=False)

    # Plot
    ax = df.plot(kind='barh', x=df.columns[0], y=df.columns[1], figsize=(20, 50))
    fig = ax.get_figure()

    # Save plot to a temp file.
    png_path = Path(tempfile.TemporaryDirectory().name).joinpath('country_households_test_plot.png')
    png_path.parent.mkdir()
    fig.savefig(png_path)
    print(f'Plot path: {png_path}')


# %% Run as a script
if __name__ == '__main__':
    sc.tic()

    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot_validate_country_households()
    else:
        validate_country_households_range()

    sc.toc()

print('Done.')
