import glob
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def readMyFile(filename):
    X = []
    Y = []

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            X.append(row[1])
            Y.append(row[2])

    return X, Y

path = './data/'
extension = 'csv'
os.chdir(path)
all_files = [i for i in glob.glob('*.{}'.format(extension))]
nrow = np.ceil(np.sqrt(len(all_files)))
ncol = nrow

print(ncol, nrow)
print(all_files)
for file in all_files:
    # Getting the file name without extension
    #file_name = os.path.splitext(os.path.basename(file))[0]
    # Reading the file content to create a DataFrame
    X, Y = readMyFile(file)

X = np.asarray([float(i) for i in X[1:]])-775
Y = np.asarray([float(i) for i in Y[1:]])

plt.subplot(nrow, ncol, 1)
plt.plot(X, Y)
plt.show()



# Example showing the Name in the print output

#      FirstYear  LastYear
# Name
# 0         1990      2007
# 1         2001      2001
# 2         2001      2008