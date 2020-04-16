import pandas as pd

url = 'https://population.un.org/household/exceldata/population_division_UN_Houseshold_Size_and_Composition_2019.xlsx'

df = pd.read_excel(url, sheet_name=3, skiprows=4)
df = df.sort_values(['Country or area', 'Reference date (dd/mm/yyyy)'], ascending=[True, False])
size = {}

for index, row in df.iterrows():
    country = row['Country or area'].lower()
    if not country in size:
        size[country] = row['Average household size (number of members)']

print(size)

