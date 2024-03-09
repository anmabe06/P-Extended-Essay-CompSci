from sympy import primerange
import decimal
import json

class GlobalFunctions:
    def getBestHyperparams(datset: str, dataRange: str):
        with open('data/datasetRangeSpecificData.json','r') as json_File :
            return json.load(json_File)[datset][dataRange]["best_hyperparameters"]


    def getstandarisationValue(datset: str, dataRange: str):
        with open('data/datasetRangeSpecificData.json','r') as json_File :
            return json.load(json_File)[datset][dataRange]["data_standarisation_value"]




class TypesOfDatasets:
    PRIMES = "primes"
    SUM = "sum"
    SUNDARAM_FACTORS = "sundaram_factors"
    TRAINING_DATASETS = [PRIMES, SUM, SUNDARAM_FACTORS]
    VERIFIED_PRIMES = "verified_primes"

    def getBounds(dataRange: str) -> tuple[int] or list[tuple[int]]:
        translator = {
            "R1": (2, 500, 4358),
            "R2": (9157, 10**4, 14760),
            "R3": (10**5 - 1071, 10**5, 10**5 + 5830),
            "R4": (10**9 - 1919, 10**9, 10**9 + 10282),
            "R5": (10**35 - 6609, 10**35, 10**35 + 38978),
            "R6": (10**50 - 9281, 10**50, 10**50 + 56822)
        }

        if len(dataRange) == 2:
            return [tuple((translator[key][0], translator[key][2])) for key in translator.keys()]
        
        if dataRange[0] == "E":
            return translator[dataRange[1:]][:2]
        
        if dataRange[0] == "T":
            return translator[dataRange[1:]][1:]




class CreateData:
    def __init__(self, typeOfDataset: str, fileToWritePath: str, dataRange: str, overwrite: bool = True, sequenceSampleLength: int = 5):
        self.sequenceSampleLength = sequenceSampleLength
        self.typeOfDataset = typeOfDataset
        self.dataRange = dataRange
        
        if overwrite:
            self.fileToWrite = open(fileToWritePath, "w")
        else:
            self.fileToWrite = open(fileToWritePath, "a")

        if self.dataRange == "R+":
            for lb, ub in TypesOfDatasets.getBounds(self.dataRange):
                self.lowerBound = lb
                self.upperBound = ub
                self.__callGenerator(self.typeOfDataset)
        else:
            self.lowerBound, self.upperBound = TypesOfDatasets.getBounds(self.dataRange)
            self.__callGenerator(self.typeOfDataset)


    def __callGenerator(self, typeOfDataset: str) -> None:
        if typeOfDataset == "primes":
            self.__generatePrimesSequence()
            return
        
        if typeOfDataset == "sum":
            self.__generateSumSequence()
            return
        
        if typeOfDataset == "sundaram_factors":
            self.__generateSundaramsFactorSeries()
            return
        
        if typeOfDataset == "verified_primes":
            self.__generateVerifiedPrimes()
            return
        
        raise Exception(f"The variable typeOfData must be a string containning either 'primes' or 'polynomial'")
    

    def __formatListData(self, listData: list[int or float], standariseListData: bool = True, writeToFile: bool = True) -> None or list:
        '''
            This method's purpose is to transform a list of integers (i.e. the
            list from which a dataset is created) into its dataset form of
            X and y values.
        '''
        if standariseListData:
            listData = self.__standariseListData(listData)

        result = []
        X = [listData[i:i + self.sequenceSampleLength] for i in range(len(listData) - self.sequenceSampleLength)]
        y = [listData[i + self.sequenceSampleLength] for i in range(len(listData) - self.sequenceSampleLength)]
        # print(listData)

        for rowIdx in range(len(X)):
            rowStr = "["
            for xValue in X[rowIdx]:
                if X[rowIdx][len(X[rowIdx]) - 1] == xValue:
                    rowStr += f"{xValue}]"
                    break
                rowStr += f"{xValue}, "
            rowStr += f" | {y[rowIdx]}\n"
            if writeToFile:
                self.fileToWrite.write(rowStr)
            else:
                result.append(rowStr)
        return result
    

    def __standariseListData(self, data: list[int or float], customStandarisationValue: int or float = None):
        if customStandarisationValue != None:
            return [number - customStandarisationValue for number in data]
        
        standarisedData = []
        standarisationValue = GlobalFunctions.getstandarisationValue(self.typeOfDataset, self.dataRange)
        print(standarisationValue)

        #print(data[-1])
        for number in data:
            standarisedData.append((number - standarisationValue))
        #print(standarisedData)
        #print([a * (data[-1]-standarisationValue) + standarisationValue for a in standarisedData])
        
        return standarisedData
    
    
    def __generatePrimesSequence(self):
        primeSequence = list(primerange(self.lowerBound, self.upperBound))
        self.__formatListData(primeSequence, standariseListData = True, writeToFile = True)
    

    def __generateSumSequence(self):
        primeSequence = list(primerange(self.lowerBound, self.upperBound))
        sumSequence = []
        idx = prevPrimeSum = 0

        for prime in primeSequence:
            idx += 1
            prevPrimeSum += prime
            sumSequence.append(prevPrimeSum)

        self.__formatListData(sumSequence, standariseListData = True, writeToFile = True)

    
    def __generateSundaramsFactorSeries(self):
        decimal.getcontext().prec = 50
        factors = [int((decimal.Decimal(num - 1) / 2)) for num in primerange(self.lowerBound, self.upperBound)]
        self.__formatListData(factors, standariseListData = True, writeToFile = True)

    
    def __generateVerifiedPrimes(self):
        for prime in list(primerange(self.lowerBound, self.upperBound)):
            self.fileToWrite.write(f"{prime}\n")