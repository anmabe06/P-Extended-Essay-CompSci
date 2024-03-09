from matplotlib import pyplot as plt
from DataPreparation import TypesOfDatasets

class SieveOfSundaram():
    def __init__(self, modelName: str):
        self.__modelName = modelName
    

    def predict(self, rangeDT: str) -> list:
        self.lowerBound, self.upperBound = TypesOfDatasets.getBounds(rangeDT)
        
        result = []
        k = (self.upperBound - 2) // 2
        integers_list = [True for _ in range(k + 1)]

        for i in range(1, k + 1):
            j = i
            while i + j + 2 * i * j <= k:
                integers_list[i + j + 2 * i * j] = False
                j += 1
        
        if self.upperBound > 2:
            result.append(2)
        for i in range(1, k + 1):
            if integers_list[i] and 2 * i + 1 > self.lowerBound:
                result.append(2 * i + 1)

        self.yData = result
        return self.yData


    def graphResults(self, yValues: list[int] = []) -> None:
        if len(yValues) < 1:
            yValues = self.yData

        plt.plot([i for i in range(len(yValues))], yValues, 'o', label="Predicted Values", color="green")
        plt.plot([i for i in range(len(yValues))], yValues, 'x', label="Real Values", color="black")
        plt.xlabel('prime index') 
        plt.ylabel('prime number') 
        
        # displaying the title
        plt.title(self.__modelName)
        plt.legend()
        plt.show()