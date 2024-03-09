# STEP 1: Create Dataset
# from DataPreparation import TypesOfDatasets, CreateData

# for dataset in [TypesOfDatasets.SUNDARAM_FACTORS]:
#     for i in range(1, 7):
#         rangeOfTraining = f"ER{i}"
#         CreateData(dataset, f"data/{dataset}{rangeOfTraining}.txt", rangeOfTraining)



# STEP 2: Get Best Hyperparams
# from SequentialNNModel import SequentialNNModel
# from DataPreparation import TypesOfDatasets
# import time

# for dataset in [TypesOfDatasets.SUNDARAM_FACTORS]:
#     for i in range(1, 7):
#         rangeOfTraining = f"ER{i}"
#         print(f"CALCULATING: Best hyperparameters for the dataset {dataset.upper()} in the range {rangeOfTraining}")
#         start = time.time()
#         SNNM = SequentialNNModel(f"data/{dataset}{rangeOfTraining}.txt", "", sub_sequence_length=5)
#         SNNM.calculateBestHyperparams(logPrefix="   ==> ")
#         end = time.time()
#         print(f"Elapsed Time: {end - start}(s)\n")
#     print("#"*80, "\n")


# STEP 3: Train and Store Models
from SequentialNNModel import SequentialNNModel
from DataPreparation import TypesOfDatasets, GlobalFunctions
import time

for dataset in [TypesOfDatasets.SUNDARAM_FACTORS]:
    for i in range(1, 7):
        rangeOfTraining = f"ER{i}"
        bestHyperparams = GlobalFunctions.getBestHyperparams(dataset, rangeOfTraining)
        print(f"TRAINING & STORING: Neural Network for the dataset {dataset.upper()} in the range {rangeOfTraining}")

        start = time.time()
        model = SequentialNNModel(f"data/{dataset}{rangeOfTraining}.txt", f"Neural Network Model for {dataset} in {rangeOfTraining}", csvFilePath=f"results/{dataset}{rangeOfTraining}.csv")
        model.compile(units=bestHyperparams["units"], optimizer=bestHyperparams["optimizer"])
        _, result = model.train(epochs=bestHyperparams["epochs"], modelFileName=f"{dataset}{rangeOfTraining}", verbose=0)
        end = time.time()
        print(f"Detailed Loss: {result}")
        print(f"Elapsed Time: {end - start}(s)\n")
    print("#"*80, "\n")


# STEP 4: Execute and Predict