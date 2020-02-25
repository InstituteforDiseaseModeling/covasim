#%% Housekeeping

import os
import sys
import pylab as pl
import pandas as pd
import sciris as sc

filename = 'SingaporeCOVID.csv'
cols = ['Case', 'age', 'sex', 'exposure_source', 'date_onset_symptoms', 'date_admission_hospital', 'date_confirmation']


#%% Load dataframe

raw_data = pd.read_csv(filename)
data = pd.DataFrame()
data = raw_data[cols]
N = len(data)


#%% Process date & day
def date_to_day(date):
    try:
        pieces = date.split('-')
        M = int(pieces[1])
        D = int(pieces[2])
        day = (M-1)*31 + D
    except:
        day = pl.inf
    return day

def day_to_date(day, first_day=0): # Sometimes this starts from Jan-01, sometimes Jan-20
    day_count = day + first_day
    if day_count > 31:
        month = 'Feb. '
        day_count -= 31
    else:
        month = 'Jan. '
    date = month + f'{day_count:2d}'
    return date

# Get dates
day = []
date = []
for i in range(N):
    row = data.loc[i]
    onset   = date_to_day(row['date_onset_symptoms'])
    admiss  = date_to_day(row['date_admission_hospital'])
    confirm = date_to_day(row['date_confirmation'])
    this_day = min([onset, admiss, confirm])
    day.append(this_day)
    date.append(day_to_date(this_day))
    
# Convert dates to days
day = pl.array(day)
first_day = day[0]
day = day - first_day + 1


data = data.assign(date=date)
data = data.assign(day=day)
# data = data.sort_values('day')
day_list = sorted(data.day.unique())
date_list = sorted(data.date.unique())
data = data.assign(ind=pl.arange(N)) # For indexing later

# Invert exposure source to people exposed
datadict = {}
for i in range(N):
    row = data.loc[i]
    datadict[row.ind] = row

# Map between cases and indices
mapping = {}
for ind,person in datadict.items():
    mapping[person.Case] = ind

contacts = {}
for ind in datadict.keys():
    contacts[ind] = []

for ind,person in datadict.items():
    if person.exposure_source != -1:
        contacts[mapping[person.exposure_source]].append(ind)


# Process into days
plotdata = {}
for i in range(N):
    row = data.loc[i]
    if row.day not in plotdata:
        plotdata[row.day] = []
    plotdata[row.day].append(row)
    

#%% Plotting
    
# Plot settings
delta = 1.05
yoff = 0.01
markersize = 300
linealpha = 0.1
dopause = 1 # Whether or not to plot live

# Movie options
dosave = 1
fps = 2 # Frames per second
figure_filename_template = 'tmp_singapore_' # This will produce e.g. tmp_singapore_13.png
movie_filename = 'singapore-covid-19.mp4'

# Configure axes
arr = pl.arange(N)/N
x = pl.sin(arr*2*pl.pi)
y = pl.cos(arr*2*pl.pi)
pl.figure(figsize=(18,18))
ax = pl.axes([0,0.15,0.9,0.8])
pl.xlim([-1,1])
pl.ylim([-1,1])
pl.axis('square')
pl.axis('off')
ages_for_color = data.age.to_numpy()
colors = sc.vectocolor(ages_for_color, cmap=sc.parulacolormap())

# Plot color legend
ax2 = pl.axes([0.86,0.98,0.08,0.88])
pl.axis('off')

# Plot total cases
ax3 = pl.axes([0.04,0.03,0.8,0.1])
ax3.set_xlabel('Day')
ax3.set_ylabel('Cumulative cses')
ax3.set_ylim([0, N])
ax3.set_xlim([day_list[0]-0.5, day_list[-1]+0.5])
sc.boxoff()
    

# Main plotting
frames = []
count = 0
max_day = max(day_list)+1
for day in range(1, max_day):
    print(f'Day {day} of {max_day}')
    n_cases = 0
    if day in day_list:
        thisday = plotdata[day]
        for person in thisday:
            n_cases += 1
            ind = person.ind
            marker = r'$♂️$' if person.sex == 'Male' else r'$♀️$'
            ax.scatter(x[ind], y[ind], s=markersize, c=[colors[ind]], marker=marker)
            ax.text(x[ind]*delta, y[ind]*delta-yoff, person.Case) 
            these_contacts = contacts[ind]
            for c in these_contacts:
                this_x = [x[ind], x[c]]
                this_y = [y[ind], y[c]]
                ax.plot(this_x, this_y, c='k', alpha=linealpha)
            
            ax2.text(0, -ind/N*1.10, f'Case {person.Case:2d}, {marker}, age {data.age[ind]:2.0f}', c=colors[ind], fontsize=16)
    
    count += n_cases
    ax3.bar(day, count, facecolor='k')
    
    ax.set_title(f'Day {day:2d}, {day_to_date(day, first_day=first_day)}, cases: {n_cases}', fontsize=24)
    
    if dosave:
        pl.savefig(f'tmp_singapore_{day}.png')
    
    if dopause:
        pl.pause(0.5)

if dosave:
    print('Saving as movie...')
    if sys.platform != 'linux':
        print('Warning! Saving as a movie on a non-Linux system is unlikely to work')
    os.system(f'ffmpeg -r {fps} -i {figure_filename_template}%d.png -vcodec mpeg4 -y {movie_filename}')
    os.system(f'rm -v {figure_filename_template}*.png')

print('Done.')