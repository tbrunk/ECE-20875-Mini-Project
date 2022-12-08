import pandas
import numpy as np
import regex as re

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))

print(dataset_1.to_string()) #This line will print out your data

# Problem 1

print(dataset_1.mad())
print(dataset_1.median())
print(dataset_1.mean())
print(dataset_1.sort_index())
#print(dataset_1[0])
print('LOOK', dataset_1[:2])
data = np.array(dataset_1)

Bridge1 = data[:, 5]
Bridge2 = data[:, 6]
Bridge3 = data[:, 7]
Bridge4 = data[:, 8]

print(data[0][2])
print(Bridge1)

print("Brooklyn Bridge's std ->", Bridge1.std(), "\nBrooklyn Bridge's avg ->", Bridge1.mean())
print("Manhattan Bridge's std ->", Bridge2.std(), "\nManhattan Bridge's avg ->", Bridge2.mean())
print("Williamsburg Bridge's std ->", Bridge3.std(), "\nWilliamsburg Bridge's avg ->", Bridge3.mean())
print("Queensboro Bridge's std ->", Bridge4.std(), "\nQueensboro Bridge's avg ->", Bridge4.mean())

Bridges = [Bridge1, Bridge2, Bridge3, Bridge4]
NumEl = len(Bridge1)
AvgDeviation = []
for Bridge in Bridges:
    MeanDeviation = []
    for Datapoint in Bridge:
        MeanDeviation.append(abs(Datapoint - Bridge.mean()))
    AvgDeviation.append(sum(MeanDeviation)/NumEl)
print("Avg Deviation ->", AvgDeviation)

# Problem 2

HiTemp = data[:, 2]
LoTemp = data[:, 3]
Precip = data[:, 4]
TotalPopString = data[:, 9]
TotalPopInt = []
AvgTemp = np.zeros((len(HiTemp)))

for Day in range(len(HiTemp)):
    AvgTemp[Day] = (HiTemp[Day] + LoTemp[Day])/2
    TotalPopInt.append(int(re.sub(r',', '', TotalPopString[Day])))


NewData = np.column_stack((AvgTemp, Precip, TotalPopInt))
AvgOfAvgTemp = np.mean(AvgTemp)
HighTempNoPrecip = []
HighTempPrecip = []
LowTempNoPrecip = []
LowTempPrecip = []

for i in range(len(HiTemp)):
    if AvgTemp[i] >= AvgOfAvgTemp and Precip[i] <= .02:
        HighTempNoPrecip.append(TotalPopInt[i])
    elif AvgTemp[i] >= AvgOfAvgTemp and Precip[i] > .02:
        HighTempPrecip.append(TotalPopInt[i])
    elif AvgTemp[i] < AvgOfAvgTemp and Precip[i] <= .02:
        LowTempNoPrecip.append(TotalPopInt[i])
    elif AvgTemp[i] < AvgOfAvgTemp and Precip[i] > .02:
        LowTempPrecip.append(TotalPopInt[i])


print(NewData)
print(AvgOfAvgTemp)

print(HighTempNoPrecip)
print(HighTempPrecip)
print(LowTempNoPrecip)
print(LowTempPrecip)

print(np.mean(HighTempNoPrecip))
print(np.mean(HighTempPrecip))
print(np.mean(LowTempNoPrecip))
print(np.mean(LowTempPrecip))
