import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import warnings
import time
from tensorflow.keras.models import model_from_json
from DataPreparation import GlobalFunctions, TypesOfDatasets
from memory_profiler import profile

def create_model(units=64, optimizer='adam', activation='relu'):
    sequence_length = 5

    model = keras.Sequential([
        keras.layers.Dense(units, activation=activation, input_shape=(sequence_length,)),
        keras.layers.Dense(1)
    ])
    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model


class SequentialNNModel():
    def __init__(self, rawDataFilePath: str, modelName: str, sub_sequence_length: int = 5, csvFilePath: str = ""):
        if csvFilePath != "":
            self.csvFile = open(csvFilePath, "a")
        
        self.rawDataFile = open(rawDataFilePath, "r")
        self.modelName = modelName
        self._isTrained = False
        self.verifiedPrimeList = [int(line.strip()) for line in open('data/verified-primes.txt').readlines()]

        self.sub_sequence_length = sub_sequence_length
        self.XTrain, self.XTest, self.yTrain, self.yTest = self._processRawData()


    def __create_model(self, units=64, optimizer='adam', activation='relu', sequence_length="") -> any:
        if sequence_length == "":
            sequence_length = self.sub_sequence_length

        model = keras.Sequential([
            keras.layers.Dense(units, activation=activation, input_shape=(sequence_length,)),
            keras.layers.Dense(1)
        ])
        # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        return model


    def _processRawData(self, separateTrainingData: bool = True, trainingSetSize: float = 0.8, randomnessOfData: int = 42) -> tuple[list]:
        X = []
        y = []
        for line in self.rawDataFile:
            line = (line.strip()).split(" | ")
            lineX = list(map(int, ((line[0].split("[")[1]).split("]")[0]).split(", ")))
            liney = int(line[1])
            #print(X, y)
            
            X.append(lineX)
            y.append(liney)

        if separateTrainingData:
            return train_test_split(X, y, test_size=(1 - trainingSetSize), random_state=randomnessOfData)

        shuffledData = shuffle(np.column_stack((X, y)), random_state=randomnessOfData)
        return shuffledData[:, :-1], shuffledData[:, -1]
        


    def compile(self, units: int = 64, optimizer: str = 'adam', activation = 'relu', sequence_length: int = 5) -> bool:
        self.model = self.__create_model(units, optimizer, activation, sequence_length)
        self._isCompiled = True
        return self._isCompiled
    

    def train(self, evaluate: bool = True, verbose: int = 1, epochs=64, saveModel: bool = True, modelFileName: str = "") -> bool or any:
        # TODO: Adjust epochs, batch_size
        # print(self.XTrain, self.yTrain)
        self.model.fit(self.XTrain, self.yTrain, epochs=epochs, batch_size=2, verbose=verbose)
        self._isTrained = True
        
        if saveModel and modelFileName != "":
            modelJSON = self.model.to_json()
            with open(f"models/modelJSONs/{modelFileName}.json", "w") as modelJSONFile:
                modelJSONFile.write(modelJSON)
            self.model.save_weights(f"models/modelWeights/{modelFileName}.h5")
        
        if evaluate:
            return self.model.evaluate(self.XTest, self.yTest)

        return self._isTrained
    

    def calculateBestHyperparams(self, log: bool = True, logPrefix: str = "") -> tuple[float]:
        # Create a KerasRegressor for use with GridSearchCV
        testModel = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, epochs=50, verbose=0)

        # Define the hyperparameters to search
        param_grid = {
            'units': [8, 16, 32, 64, 128],
            'optimizer': ['adam', 'rmsprop'],
            'activation': ['relu', 'tanh'],
            'epochs': [50, 100, 150, 200, 400]
        }
        # 

        # Perform grid search with cross-validation
        cv = TimeSeriesSplit(n_splits=3)
        #print(cv)
        grid = GridSearchCV(estimator=testModel, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv)
        #print(self.XTrain, self.yTrain)
        grid_result = grid.fit(self.XTrain, self.yTrain)

        if log:
            print(f"{logPrefix}Best Parameters: {grid_result.best_params_}")
            print(f"{logPrefix}Best Negative Mean Squared Error: {grid_result.best_score_}")
        return (grid_result.best_params_, grid_result.best_score_)


    def predict(self, newX: list[int] | list[list[int]], standarisationFunction: callable = lambda value: value) -> list:
        if not self._isTrained:
            raise Exception("The model '" + self.modelName + " must be trained before it is able to predict new numbers")
        self._predictedXValues = newX

        self.standarisationFunction = standarisationFunction

        result = []
        # if there is only one sequence
        if type(newX[0]) is int:
            result.append(self.__predictIndividualValue(newX))
        # if there there are multiple sequences
        elif type(newX[0]) is list:
            for individualX in newX:
                result.append(self.__predictIndividualValue(individualX))
        
        self._predictedPrimes = result
        return self._predictedPrimes


    @profile
    def __predictIndividualValue(self, x: int, roundValues: bool = False) -> list:
        if roundValues:
            return self.standarisationFunction(round(self.model.predict(np.array([x]), verbose=0)[0][0]))
        return self.standarisationFunction(self.model.predict(np.array([x]), verbose=5)[0][0])
    
    
    # def recursiveRepeatForTestingRange(self, dataset: str, trainingRange: str, currentXData: list[int or float] = []):
    #     if len(currentXData) < 1:
    #         # loweBound, upperBound = TypesOfDatasets.getBounds(f"E{testingRange[1:]}")
    #         lastDataInDataFile = (list(open(f"data/{dataset}{trainingRange}.txt", "r"))[-1]).split("] | ")
    #         resultFirst = list(map(int, lastDataInDataFile[0].split(", ")[1:]))
    #         currentXData = resultFirst + [int(lastDataInDataFile[1])]
        
    #     standarisationValue = GlobalFunctions.getstandarisationValue(dataset, trainingRange)
    #     start = time.time()
    #     predictionsY = self.predict(currentXData, lambda x: x + standarisationValue)
    #     end = time.time()
    #     print(predictionsY)
    #     print(f"Prediction: {predictionsY}  ||  Verification: {self.verifyResults(currentXData, predictionsY, trainingRange)}  ||  Time Elapsed: {end - start}(s)")


    #     if self.verifiedPrimeList.index(currentXData[-1] + standarisationValue) < 595 * int(trainingRange[-1]):
    #         newXData = currentXData[1:] + [self.verifiedPrimeList[self.verifiedPrimeList.index(currentXData[-1] + standarisationValue) + 1] - standarisationValue]
    #         self.recursiveRepeatForTestingRange(dataset, trainingRange, newXData)
    

    def recursiveRepeatForTestingRange(self, dataset: str, trainingRange: str, currentXData: list[int or float] = [], desiredPrime: int = -1):
        if len(currentXData) < 1:
            # loweBound, upperBound = TypesOfDatasets.getBounds(f"E{testingRange[1:]}")
            lastDataInDataFile = (list(open(f"data/{dataset}{trainingRange}.txt", "r"))[-1]).split("] | ")
            resultFirst = list(map(int, lastDataInDataFile[0].split(", ")[1:]))
            currentXData = resultFirst + [int(lastDataInDataFile[1])]

        standarisationValue = GlobalFunctions.getstandarisationValue(dataset, trainingRange)
        print(standarisationValue)
        
        if desiredPrime == -1 and self.verifiedPrimeList.index(currentXData[-1] + standarisationValue) < 595 * int(trainingRange[-1]):
            desiredPrime = self.verifiedPrimeList[self.verifiedPrimeList.index(currentXData[-1] + standarisationValue) + 1]
        
        
        start = time.time()
        predictionsY = self.predict(currentXData, lambda x: x + standarisationValue)
        end = time.time()
        print(predictionsY)
        print(f"Prediction: {predictionsY}  ||  Verification: {self.verifyResults(currentXData, predictionsY, trainingRange, desiredPrime=desiredPrime)}  ||  Time Elapsed: {end - start}(s)")

        if self.verifiedPrimeList.index(desiredPrime) + 1 < 595 * int(trainingRange[-1]):
            newXData = currentXData[1:] + [int(predictionsY)] if isinstance(predictionsY, int) else currentXData[1:] + [int(predictionsY[0])]
            self.recursiveRepeatForTestingRange(dataset, trainingRange, newXData, desiredPrime=self.verifiedPrimeList[self.verifiedPrimeList.index(desiredPrime) + 1])


    def verifyResults(self, resultsX: list[int or float] or list[list[int or float]], resultsY: list[int or float] or int, dataRange: str, writeToCSV: bool = False, desiredPrime: int = -1) -> bool:
        '''
            Results are outputted as a list of the type 
            of dictionaries specified in the method
            self.__verifyIndividualResult().
        '''
        result = []
        # if there is only one sequence
        if type(resultsX[0]) is int or type(resultsX[0]) is float:
            resultY = resultsY[0] if type(resultsY) is list else resultsY
            result.append(self.__verifyIndividualResult(resultsX, resultY, dataRange, writeToCSV=writeToCSV, desiredPrime = desiredPrime))
        # if there there are multiple sequences
        elif type(resultsX[0]) is list:
            for idx in range(len(resultsX)):
                result.append(self.__verifyIndividualResult(resultsX[idx], resultsY[idx], dataRange, writeToCSV=writeToCSV, desiredPrime = desiredPrime))
        
        return result


    def __verifyIndividualResult(self, resultX: list[int or float], resultY: int or float, dataRange: str, writeToCSV: bool = False, desiredPrime: int = -1) -> bool:
        '''
            Results are outputted as dictionaries:
                result = {
                    'previous_primes': [2, 3, 5, 7, 11],
                    'desired_prime': 13,
                    'number_predicted': 13,
                    'offset': 0,
                    'percentage_error': 0
                }
        '''
        if desiredPrime == -1:
            desiredPrime = self.verifiedPrimeList[self.verifiedPrimeList.index(self.standarisationFunction(resultX[-1])) + 595 * (int(dataRange[-1]) - 1)]

        individualData = dict()
        individualData['previous_primes'] = resultX
        individualData['desired_prime'] = desiredPrime
        individualData['number_predicted'] = resultY
        individualData['offset'] = resultY - desiredPrime
        individualData['percentage_error'] = abs(individualData['offset']) / resultY * 100
        
        if writeToCSV:
            self.__writeLineToCSV(individualData)
        
        return individualData
    

    def writeToCSV(self, bunch: list[dict]) -> None:
        '''
            The data received must be a list of dictionaries 
            of the following form specified in the method 
            self.__writeLineToCSV()
        '''
        if not hasattr(self, 'csvFile'):
            raise Exception("Tried to write data in a non-specified CSV file.")
        
        warnings.warn("Performing a writeToCSV() with big lists of data is slower than setting writeToCSV=True when using the verifyResults() method", RuntimeWarning)
        for data in bunch:
            self.__writeLineToCSV(data)

    
    def __writeLineToCSV(self, data: dict) -> None:
        '''
            The data received must be a dictionary of the
            following form:
                dictionary = {
                    'prime_index': 370,
                    'desired_prime': 3531,
                    'number_predicted': 2530,
                    'offset': -1,
                    'percentage_error': 0.04
                }
            Or of the form:
                dictionary = {
                    'previous_primes': [431, 433, 439, 443, 449],
                    'desired_prime': 457,
                    'number_predicted': 459,
                    'offset': 2,
                    'percentage_error': 0.44
                }
        '''
        if not hasattr(self, 'csvFile'):
            raise Exception("Tried to write data in a non-specified CSV file.")
        
        if 'prime_index' in data:
            line = f"\n{data['prime_index']}, {data['desired_prime']}, {data['number_predicted']}, {data['offset']}, {data['percentage_error']}"
        elif 'previous_primes' in data:
            line = f"\n\"{data['previous_primes']}\", {data['desired_prime']}, {data['number_predicted']}, {data['offset']}, {data['percentage_error']}"
        self.csvFile.write(line)
    

    def graphResults(self):
        pass




class SequentialNNModelFromFile(SequentialNNModel):
    def __init__(self, trainedDataset: str, trainedRange: str, modelFileName: str = ""):
        self.trainedDataset = trainedDataset
        self.trainedRange = trainedRange
        self.modelFileName = modelFileName if modelFileName != "" else f"{trainedDataset}{trainedRange}"
        self.verifiedPrimeList = [int(line.strip()) for line in open('data/verified-primes.txt').readlines()]
        self._isTrained = True

        json_file = open(f"models/modelJSONs/{self.modelFileName}.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.compile(loss='mean_squared_error', optimizer=GlobalFunctions.getBestHyperparams(trainedDataset, trainedRange)["optimizer"])
        print(self.model)
        self.model.load_weights(f"models/modelWeights/{self.modelFileName}.h5")
    

    def evaluateImportedModel(self):
        self.rawDataFile = open(f"data/{self.trainedDataset}{self.trainedRange}.txt", "r")
        self.XTest, self.yTest = super()._processRawData(separateTrainingData=False)
        #print(f"self.XTest: {self.XTest}")
        #print(f"self.yTest: {self.yTest}")
        return self.model.evaluate(self.XTest, self.yTest)