from SequentialNNModel import SequentialNNModelFromFile, SequentialNNModel
from DataPreparation import TypesOfDatasets, GlobalFunctions
import time

dataset = "primes"
rangeOfTraining = "ER1"
rangeOfTesting = "TR1"
model = SequentialNNModelFromFile(dataset, rangeOfTraining)
model.recursiveRepeatForTestingRange(dataset, rangeOfTraining)
# testingDataX = [465, 477, 485, 489, 497]
# predictionsY = model.predict(testingDataX, lambda x: x + GlobalFunctions.getstandarisationValue(dataset, rangeOfTraining))
# print(predictionsY)
# print(model.verifyResults(testingDataX, predictionsY, rangeOfTraining))

# a = list(open("data/primesER1.txt", "r"))
# print(a[-1])
