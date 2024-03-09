from DataPreparation import TypesOfDatasets, GlobalFunctions
import time
from SequentialNNModelFromFile import SequentialNNModelFromFile
from sympy import primerange
import decimal


# LSTM1 = SequentialNNModelFromFile(TypesOfDatasets.PRIMES, "ER1")
# result = LSTM1.predict([2, 3, 5, 7, 11])
# print(result)


verifiedPrimes = list(open("data/verified-primes.txt", "r"))
lastPrimes = {
    "ER1": [int(i.strip()) for i in verifiedPrimes[90:95]],
    "ER2": [int(i.strip()) for i in verifiedPrimes[685:690]],
    "ER3": [int(i.strip()) for i in verifiedPrimes[1279:1284]],
    "ER4": [int(i.strip()) for i in verifiedPrimes[1874:1879]],
    "ER5": [int(i.strip()) for i in verifiedPrimes[2469:2474]],
    "ER6": [int(i.strip()) for i in verifiedPrimes[3064:3069]],
}


decimal.getcontext().prec = 50
#dataset = TypesOfDatasets.SUM
for dataset in [TypesOfDatasets.SUNDARAM_FACTORS]:
    for i in range(1, 7):
        primeRange = f"ER{i}"
        primeRangeT = f"TR{i}"
        predictionsWriteFile = open(f"results-txt/{dataset}{primeRangeT}-predictions.txt", "a")
        timeWriteFile = open(f"results-txt/{dataset}{primeRangeT}-time.txt", "a")

        t1 = time.time()
        LSTM = SequentialNNModelFromFile(TypesOfDatasets.PRIMES, primeRange)
        t2 = time.time()
        print(f"Loading time for LSTM (dataset {dataset}, range {primeRangeT}): {t2 - t1}")

        standarisationValue = GlobalFunctions.getstandarisationValue(dataset, primeRange)
        bounds = TypesOfDatasets.getBounds(primeRangeT)
        entireData = lastPrimes[primeRange] + list(primerange(bounds[0], bounds[1]))
        for idx in range(0, 500):
            data = [int((decimal.Decimal(m - 1) / 2) - standarisationValue) for m in entireData[idx: idx + 5]]
            t1 = time.time()
            result = 2 * (int(LSTM.predict(data)[0]) + standarisationValue) + 1
            t2 = time.time()

            # Only because of excel's floating point limitation
            if i > 4:
                result = result - GlobalFunctions.getstandarisationValue(TypesOfDatasets.PRIMES, primeRange)
            
            predictionsWriteFile.write(f"{result}\n")
            timeWriteFile.write(f"{t2-t1}\n")

