from SieveOfSundaram import SieveOfSundaram
from DataPreparation import TypesOfDatasets
from sympy import primerange
from memory_profiler import profile
import time

verifiedPrimes = {
    "TR1": list(primerange(500, 4363)),
    "TR2": list(primerange(10**4, 14767)),
    "TR3": list(primerange(10**5, 10**5 + 5863)),
    "TR4": list(primerange(10**9, 10**9 + 10303)),
    "TR5": list(primerange(10**35, 10**35 + 38997)),
    "TR6": list(primerange(10**50, 10**50 + 56857))
}

@profile
def hey():
    for n in range(3, 4):
        primeRangeT = f"TR{n}"
        print(f"Starting Memory Measurements for {primeRangeT}...")
        primes = verifiedPrimes[primeRangeT]
        for i in range(0, 4):
            prediction = predict(primes[i], primes[i] + 1)[0]


def predict(lowerBound, upperBound):
    result = []
    k = (upperBound - 2) // 2
    integers_list = [True for _ in range(k + 1)]

    for i in range(1, k + 1):
        j = i
        while i + j + 2 * i * j <= k:
            integers_list[i + j + 2 * i * j] = False
            j += 1
    
    for i in range(1, k + 1):
        if integers_list[i] and 2 * i + 1 >= lowerBound:
            result.append(2 * i + 1)

    return result

hey()





# timeWriteFile = open(f"results-txt/sieve_of_sundaram{primeRangeT}-time.txt", "a")
    # t1 = time.perf_counter()
    # prediction = predict(primes[0], primes[1] + 1)[0]
    # t2 = time.perf_counter()
    # diff = "{:.19f}".format(t2 - t1)
    # print(f"  >> Time for 1 execution(s): {diff}(s)\n")

# SS = SieveOfSundaram("TR1")
# print(SS.predict("TR6"))