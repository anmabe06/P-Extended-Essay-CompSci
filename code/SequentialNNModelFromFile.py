from tensorflow.keras.models import model_from_json
from SequentialNNModel import SequentialNNModel
from DataPreparation import GlobalFunctions

class SequentialNNModelFromFile(SequentialNNModel):
    def __init__(self, trainedDataset: str, trainedRange: str, modelFileName: str = ""):
        self.trainedDataset = trainedDataset
        self.trainedRange = trainedRange
        self._isTrained = True
        self.modelFileName = modelFileName if modelFileName != "" else f"{trainedDataset}{trainedRange}"

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
        print(f"self.XTest: {self.XTest}")
        print(f"self.yTest: {self.yTest}")
        #return self.model.evaluate(self.XTest, self.yTest)