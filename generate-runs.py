import numpy as np
import pickle
import os
import h5py
numExamples = 1000000
predictionAccuracy = [0.1, 0.9]

# Change path accordingly
path = '/Users/srinidhi/Downloads/prediction-storage/matrix'

#Types can be binary, integer, or float
def createData(numExamples, predictionAccuracy, numRuns, dataType):
    exampleHeader = np.arange(numExamples)
    first = True
    if dataType == 'binary':
        d_type = 'int8' # 1 byte
    elif dataType == 'integer':
        d_type = 'int16'
    else:
        d_type = 'float'
    if os.path.isfile(path+str(numRuns)+'-'+str(dataType)):
        # File exists already
        matrix = np.load(path+str(numRuns)+'-'+str(dataType))
    else:
        #Create dtypes array
        dtypeArray = [None] * numExamples
        for i in range(0, numExamples):
            dtypeArray[i] = (str(i), d_type)
        #Populate matrix with rows
        for i in range(0, numRuns):
            row = np.random.choice([0,1], size=numExamples, p=predictionAccuracy)
            row = np.array([tuple(row)], dtype=dtypeArray)
            if first:
                matrix = row
                first = False
            else:
                matrix = np.append(matrix, row)
        if dataType == 'binary':
            matrix = np.packbits(matrix.view(np.uint8))
        with open(path+str(numRuns)+'-'+str(dataType), 'wb') as f:
            # File doesn't exist so saving
            #pickle.dump(matrix, f)
            np.save(f, matrix)
    # H5py saving scheme
    h5f = h5py.File(path + 'hfpy-'+str(numRuns)+str(dataType), 'w')
    h5f.create_dataset('dataset_1', data=matrix, compression='gzip')

for dataType in ['binary', 'float', 'integer']:
    for numRuns in [1, 10, 100, 1000]:
        createData(numExamples, predictionAccuracy, numRuns, dataType)
        
