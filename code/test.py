from DataPreparation import TypesOfDatasets, CreateData, GlobalFunctions
from SequentialNNModel import SequentialNNModel
from SequentialNNModelFromFile import SequentialNNModelFromFile

# models = dict(zip(TypesOfDatasets.TRAINING_DATASETS, [{"X": [], "y": []}] * len(TypesOfDatasets.TRAINING_DATASETS)))
# for dataset in TypesOfDatasets.TRAINING_DATASETS:
#     for i in range(1, 7):
#         rangeOfTraining = f"ER{i}"
#         # model = SequentialNNModel(f"data/{dataset}{rangeOfTraining}.txt", f"Neural Network Model for {dataset} in {rangeOfTraining}", csvFilePath="results/{dataset}{rangeOfTraining}.csv")
#         # bestHyperparams = model.bestHyperparams[dataset]
#         # model.compile(units=bestHyperparams["units"], optimizer=bestHyperparams["optimizer"])
#         # model.train(epochs=bestHyperparams["epochs"])
#         # models[dataset][rangeOfTraining] = model
#         X = []
#         y = []
#         for line in self.rawDataFile:
#             line = (line.strip()).split(" | ")
#             line = ((line[0].split("[")[1]).split("]")[0]).split(", ")
            
#             if line[0] == '0.5':
#                 # This covers the case for 2 of the sundarams_factors
#                 tempX = []
#                 tempX.append(float(line[0]))
#                 tempX = tempX + list(map(int, line[1:]))
#             else:
#                 X.append(list(map(int, line)))

#             y.append(int(line[1]))
        
#         if separateTrainingData:
#             return train_test_split(X, y, test_size=(1 - trainingSetSize), random_state=randomnessOfData)
#         models[dataset][rangeOfTraining]["X"] = []
#         models[dataset][rangeOfTraining]["y"] = []

# print(models)


# dataset = "primes"
# rangeOfTraining = f"ER{1}"
# model = SequentialNNModel(f"data/{dataset}{rangeOfTraining}.txt", f"Neural Network Model for {dataset} in {rangeOfTraining}", csvFilePath=f"results/{dataset}{rangeOfTraining}.csv")
# bestHyperparams = model.bestHyperparams[dataset]
# model.compile(units=bestHyperparams["units"], optimizer=bestHyperparams["optimizer"])
# #model.compile(units=bestHyperparams["units"], optimizer="adam")
# _, result = model.train(epochs=bestHyperparams["epochs"], modelFileName=f"{dataset}{rangeOfTraining}", verbose=0)
# print(f"Result: {result}")


for dataset in TypesOfDatasets.TRAINING_DATASETS:
    for i in range(1, 7):
        dataset = "sundaram_factors"
        rangeOfTraining = f"ER{i}"
        print(f"Training & Storing {dataset}{rangeOfTraining} model...")
        model = SequentialNNModel(f"data/{dataset}{rangeOfTraining}.txt", f"Neural Network Model for {dataset} in {rangeOfTraining}", csvFilePath=f"results/{dataset}{rangeOfTraining}.csv")
        bestHyperparams = GlobalFunctions.getBestHyperparams()
        model.compile(units=bestHyperparams["units"], optimizer=bestHyperparams["optimizer"])
        model.train(epochs=bestHyperparams["epochs"], modelFileName=f"{dataset}{rangeOfTraining}", verbose=0)
        print("\n")

# models[dataset][rangeOfTraining] = model


# print(models)



# TODO To import datasets
# datasetForEvaluation = "primes"
# rangeForEvaluation = "ER1"

# testImport = SequentialNNModelFromFile("primesER1", "primes", "ER1")
# print(testImport.evaluateImportedModel())

# for dataset in TypesOfDatasets.TRAINING_DATASETS:
#     for i in range(1, 7):
#         rangeOfTraining = f"ER{i}"





models = {'primes': {}, 'sum': {}, 'sundaram_factors': {}} 
print(models)
for dataset in TypesOfDatasets.TRAINING_DATASETS:
    for i in range(1, 7):        
        rangeOfTraining = f"ER{i}"
        models[dataset].update({
            rangeOfTraining: {"X": [], "y": []}
        })
        X = []
        y = []

        with open(f"data/{dataset}{rangeOfTraining}.txt", "r") as dataSet:
            for line in dataSet:
                line = (line.strip()).split(" | ")
                line = ((line[0].split("[")[1]).split("]")[0]).split(", ")
                
                X.append(list(map(int, line)))

                y.append(int(line[1]))

            models[dataset][rangeOfTraining]["X"] = X
            models[dataset][rangeOfTraining]["y"] = y
        


output = open(f"data/output.out", "w")
output.write(json.dumps(models))
#print(models)




############################################################################
################### TODO TODO TODO to be able to execute the needed code
#################### https://saturncloud.io/blog/how-to-enable-sse-and-avx-support-in-tensorflow-on-google-cloud-platform-console/
############################################################################

