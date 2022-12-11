import pandas
import numpy as np
import regex as re
import math as ma
from sklearn.mixture import GaussianMixture

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

print("The average mean deviation of each of column of data")
print(dataset_1.mad())
print("Median of each column of data")
print(dataset_1.median())
print("Avg of each column of data")
print(dataset_1.mean())
print("Indexing the data appropriately")
print(dataset_1.sort_index())

data = np.array(dataset_1)

Bridge1 = data[:, 5]
Bridge2 = data[:, 6]
Bridge3 = data[:, 7]
Bridge4 = data[:, 8]

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

print("\nProblem 2\n")
print("The average temperature of our data set:", AvgOfAvgTemp)

print("Total people crossing the bridges with high temperature and low precipitation -> ", HighTempNoPrecip)
print("Total people crossing the bridges with high temperature and high precipitation -> ", HighTempPrecip)
print("Total people crossing the bridges with low temperature and low precipitation -> ", LowTempNoPrecip)
print("Total people crossing the bridges with low temperature and high precipitation -> ", LowTempPrecip)
print()

print("The average amount of bikers crossing the bridges with high temp and low precipitation -> ", np.mean(HighTempNoPrecip))
print("The average amount of bikers crossing the bridges with high temp and high precipitation -> ", np.mean(HighTempPrecip))
print("The average amount of bikers crossing the bridges with low temp and low precipitation -> ", np.mean(LowTempNoPrecip))
print("The average amount of bikers crossing the bridges with low temp and high precipitation -> ", np.mean(LowTempPrecip))

HiLoPrecip_NoPrecip = LowTempNoPrecip + HighTempPrecip

range1 = (np.mean(LowTempPrecip) + np.mean(HighTempPrecip)) / 2
range2 = (np.mean(HighTempPrecip) + np.mean(LowTempNoPrecip)) / 2
range3 = (np.mean(LowTempNoPrecip) + np.mean(HighTempNoPrecip)) / 2

conf_matrix = [[0]*4 for j in range(4)]

i = 0
while i < len(LowTempPrecip):
    if LowTempPrecip[i] < range1:
        conf_matrix[0][0] += 1
    elif LowTempPrecip[i] < range2 and LowTempPrecip[i] >= range1:
        conf_matrix[1][0] += 1
    elif LowTempPrecip[i] < range3 and LowTempPrecip[i] >= range2:
        conf_matrix[2][0] += 1
    elif LowTempPrecip[i] >= range3:
        conf_matrix[3][0] += 1
    i += 1

i = 0
while i < len(HighTempPrecip):
    if HighTempPrecip[i] < range1:
        conf_matrix[0][1] += 1
    elif HighTempPrecip[i] < range2 and HighTempPrecip[i] >= range1:
        conf_matrix[1][1] += 1
    elif HighTempPrecip[i] < range3 and HighTempPrecip[i] >= range2:
        conf_matrix[2][1] += 1
    elif HighTempPrecip[i] >= range3:
        conf_matrix[3][1] += 1
    i += 1

i = 0
while i < len(LowTempNoPrecip):
    if LowTempNoPrecip[i] < range1:
        conf_matrix[0][2] += 1
    elif LowTempNoPrecip[i] < range2 and LowTempNoPrecip[i] >= range1:
        conf_matrix[1][2] += 1
    elif LowTempNoPrecip[i] < range3 and LowTempNoPrecip[i] >= range2:
        conf_matrix[2][2] += 1
    elif LowTempNoPrecip[i] >= range3:
        conf_matrix[3][2] += 1
    i += 1

i = 0
while i < len(HighTempNoPrecip):
    if HighTempNoPrecip[i] < range1:
        conf_matrix[0][3] += 1
    elif HighTempNoPrecip[i] < range2 and HighTempNoPrecip[i] >= range1:
        conf_matrix[1][3] += 1
    elif HighTempNoPrecip[i] < range3 and HighTempNoPrecip[i] >= range2:
        conf_matrix[2][3] += 1
    elif HighTempNoPrecip[i] >= range3:
        conf_matrix[3][3] += 1
    i += 1

conf_matrix = np.array(conf_matrix)
print("Confusion matrix with 4 subgroups")
print(conf_matrix)

accuracy = (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2] + conf_matrix[3][3]) / sum(sum(conf_matrix))

print("The accuracy of our classifications into 4 subgroups", accuracy)

# New Confusion Matrix
new_range1 = (np.mean(HiLoPrecip_NoPrecip) + np.mean(LowTempPrecip)) / 2
new_range2 = (np.mean(HighTempNoPrecip) + np.mean(HiLoPrecip_NoPrecip)) / 2

new_conf_matrix = [[0]*3 for j in range(3)]

i = 0
while i < len(LowTempPrecip):
    if LowTempPrecip[i] < new_range1:
         new_conf_matrix[0][0] += 1
    elif LowTempPrecip[i] < new_range2 and LowTempPrecip[i] >= new_range1:
        new_conf_matrix[1][0] += 1
    elif LowTempPrecip[i] >= new_range2:
        new_conf_matrix[2][0] += 1
    i += 1

i = 0
while i < len(HiLoPrecip_NoPrecip):
    if HiLoPrecip_NoPrecip[i] < new_range1:
         new_conf_matrix[0][1] += 1
    elif HiLoPrecip_NoPrecip[i] < new_range2 and HiLoPrecip_NoPrecip[i] >= new_range1:
        new_conf_matrix[1][1] += 1
    elif HiLoPrecip_NoPrecip[i] >= new_range2:
        new_conf_matrix[2][1] += 1
    i += 1

i = 0
while i < len(HighTempNoPrecip):
    if HighTempNoPrecip[i] < new_range1:
         new_conf_matrix[0][2] += 1
    elif HighTempNoPrecip[i] < new_range2 and HighTempNoPrecip[i] >= new_range1:
        new_conf_matrix[1][2] += 1
    elif HighTempNoPrecip[i] >= new_range2:
        new_conf_matrix[2][2] += 1
    i += 1

new_conf_matrix = np.array(new_conf_matrix)
print("Confusion matrix with 3 subgroups")
print(new_conf_matrix)

new_accuracy = (new_conf_matrix[0][0] + new_conf_matrix[1][1] + new_conf_matrix[2][2]) / sum(sum(new_conf_matrix))

print("The accuracy of our classifications into 3 subgroups", new_accuracy)

print("Combining the average amount of bikers crossing the bridges of high temp and low precipitation and low temp and low precipitation -> ", np.mean(HiLoPrecip_NoPrecip))
print("The total amount of bikers crossing the bridges of high temp and low precipitation and low temp and low precipitation -> ", HiLoPrecip_NoPrecip)

print("\nProblem #3\n")

NumWeeks = ma.ceil(NumEl/7)

Fridays = np.array(data[0, :])
Saturdays = np.array(data[1, :])
Sundays = np.array(data[2, :])
Mondays = np.array(data[3, :])
Tuesdays = np.array(data[4, :])
Wednesdays = np.array(data[5, :])
Thursdays = np.array(data[6, :])

Zeros = np.zeros(NumEl)
Coords = np.column_stack((Zeros, TotalPopInt))

for z in range(NumEl):
    TotalPopInt[z] = [TotalPopInt[z], z % 7]

Tenth = ma.floor(NumEl / 10)
TestList = []

for o in range(Tenth):
    TenIndex = (10 * (o+1)) - (1*o) -1
    TestList.append(TotalPopInt[TenIndex])
    TotalPopInt.pop(TenIndex)
TestData = np.array(TestList)

Zeros = np.zeros(NumEl - 21)
Coords = np.array(TotalPopInt)

print("New shape of total pop and respective day to be evaluated as data ->", Coords.shape)

print("The data to be used as test data shown in the array below")
print(TestData)
print("Shape of test data ->", TestData.shape)

Gausian = GaussianMixture(n_components=7, random_state=0).fit(Coords)
Prediction = Gausian.predict(TestData)
print("The days that the model predicted ->", Prediction)

Prob3Accuracy = 0
for u in range(len(Prediction)):
    if Prediction[u] == TestData[u, 1]:
        Prob3Accuracy += 1
print("Accuracy of prediction percentage ->", Prob3Accuracy/len(Prediction))