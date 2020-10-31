import covasim as cv

if __name__ == '__main__':

    msim1 = cv.MultiSim.load('pop1.msim')
    msim2 = cv.MultiSim.load('pop2.msim')
    msim3 = cv.MultiSim.load('pop3.msim')

    data1 = msim1.compare(output=True)
    data1 = data1.transpose()
    print('\nSeed 1')
    print('cum_infections-----------------------------')
    print(f"min:{ min(data1['cum_infections']) }")
    print(f"av: {sum(data1['cum_infections'])/11}")
    print(f"max: {max(data1['cum_infections'])}")

    print('cum_severe-----------------------------')
    print(f"min:{ min(data1['cum_severe']) }")
    print(f"av: {sum(data1['cum_severe'])/11}")
    print(f"max: {max(data1['cum_severe'])}")

    print('cum_critical-----------------------------')
    print(f"min:{ min(data1['cum_critical']) }")
    print(f"av: {sum(data1['cum_critical'])/11}")
    print(f"max: {max(data1['cum_critical'])}")

    print('cum_deaths-----------------------------')
    print(f"min:{ min(data1['cum_deaths']) }")
    print(f"av: {sum(data1['cum_deaths'])/11}")
    print(f"max: {max(data1['cum_deaths'])}")

    data2 = msim2.compare(output=True)
    data2 = data2.transpose()
    print('\nSeed 2')
    print('cum_infections-----------------------------')
    print(f"min:{ min(data2['cum_infections']) }")
    print(f"av: {sum(data2['cum_infections'])/11}")
    print(f"max: {max(data2['cum_infections'])}")

    print('cum_severe-----------------------------')
    print(f"min:{ min(data2['cum_severe']) }")
    print(f"av: {sum(data2['cum_severe'])/11}")
    print(f"max: {max(data2['cum_severe'])}")

    print('cum_critical-----------------------------')
    print(f"min:{ min(data2['cum_critical']) }")
    print(f"av: {sum(data2['cum_critical'])/11}")
    print(f"max: {max(data2['cum_critical'])}")

    print('cum_deaths-----------------------------')
    print(f"min:{ min(data2['cum_deaths']) }")
    print(f"av: {sum(data2['cum_deaths'])/11}")
    print(f"max: {max(data2['cum_deaths'])}")

    data3 = msim3.compare(output=True)
    data3 = data3.transpose()
    print('\nSeed 3')
    print('cum_infections-----------------------------')
    print(f"min:{ min(data3['cum_infections']) }")
    print(f"av: {sum(data3['cum_infections'])/11}")
    print(f"max: {max(data3['cum_infections'])}")

    print('cum_severe-----------------------------')
    print(f"min:{ min(data3['cum_severe']) }")
    print(f"av: {sum(data3['cum_severe'])/11}")
    print(f"max: {max(data3['cum_severe'])}")

    print('cum_critical-----------------------------')
    print(f"min:{ min(data3['cum_critical']) }")
    print(f"av: {sum(data3['cum_critical'])/11}")
    print(f"max: {max(data3['cum_critical'])}")

    print('cum_deaths-----------------------------')
    print(f"min:{ min(data3['cum_deaths']) }")
    print(f"av: {sum(data3['cum_deaths'])/11}")
    print(f"max: {max(data3['cum_deaths'])}")