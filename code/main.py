# 1. DATA CREATION
from DataPreparation import TypesOfDatasets, CreateData

CreateData(TypesOfDatasets.VERIFIED_PRIMES, "data/verified-primes.txt", "R+")
for dataset in TypesOfDatasets.TRAINING_DATASETS:
    for i in range(1, 7):
        rangeOfTraining = f"ER{i}"
        CreateData(dataset, f"data/{dataset}{rangeOfTraining}.txt", rangeOfTraining)




# 2.1. BEST HYPERPARAMETER CONFIGURATION
from SequentialNNModel import SequentialNNModel
from DataPreparation import TypesOfDatasets
import time

for dataset in TypesOfDatasets.TRAINING_DATASETS:
    for i in range(1, 7):
        rangeOfTraining = f"ER{i}"
        print(f"CALCULATING: Best hyperparameters for the dataset {dataset.upper()} in the range {rangeOfTraining}")
        start = time.time()
        SNNM = SequentialNNModel(f"data/{dataset}{rangeOfTraining}.txt", "", sub_sequence_length=5)
        SNNM.getBestHyperparams(logPrefix="   ==> ")
        end = time.time()
        print(f"Elapsed Time: {end - start}(s)\n")
    print("#"*80, "\n")




# 2.2. TRAIN & STORE MODELS
from SequentialNNModel import SequentialNNModel
from DataPreparation import TypesOfDatasets, GlobalFunctions

for dataset in TypesOfDatasets.TRAINING_DATASETS:
    for i in range(1, 7):
        rangeOfTraining = f"ER{i}"
        bestHyperparams = GlobalFunctions.getBestHyperparams(dataset, rangeOfTraining)
        print(f"TRAINING & STORING: Neural Network for the dataset {dataset.upper()} in the range {rangeOfTraining}")

        start = time.time()
        model = SequentialNNModel(f"data/{dataset}{rangeOfTraining}.txt", f"Neural Network Model for {dataset} in {rangeOfTraining}", csvFilePath=f"results/{dataset}{rangeOfTraining}.csv")
        model.compile(units=bestHyperparams["units"], optimizer=bestHyperparams["optimizer"])
        _, result = model.train(epochs=bestHyperparams["epochs"], modelFileName=f"{dataset}{rangeOfTraining}", verbose=0)
        end = time.time()
        print(f"Detailed Loss: {result[0]}")
        print(f"Elapsed Time: {end - start}(s)\n")
    print("#"*80, "\n")






from ProcessAnalyser import ProcessAnalyser as PA

# Code for Sundaram's Sieve
#from SieveOfSundaram import SieveOfSundaram
#sos = SieveOfSundaram("Sieve of Sundaram", rangeOfTesting)
#sos.predict()
#sos.showGraph()


# Code for Neural Network
from SequentialNNModel import SequentialNNModel
primesNN = SequentialNNModel(f"data/primes{rangeOfTesting}.txt", "Prime Sequence NN", sub_sequence_length=5, csvFilePath=f"results/primes{rangeOfTesting}.txt")
primesNN.calculateBestHyperparams()
# primesNN.compile()
# primesNN.train(verbose=0, evaluate=False)

# newXValues = [11, 13, 17, 19, 23]
# result = primesNN.predict(newXValues)
# result, time = PA.measure(primesNN.predict, [newXValues], measureMemory=False)
# print(f"Predicted Prime(s): {result} || Time: {time}")
# print(primesNN.verifyResults(newXValues, result, writeToCSV = True))