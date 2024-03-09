import tracemalloc
from sympy import primerange

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
        if integers_list[i] and 2 * i + 1 > lowerBound:
            result.append(2 * i + 1)

    return result

def your_function_or_loop():
    # Start tracing memory
    primes = list(primerange(10**5, 10**5 + 5863))
    
    # Start tracing memory allocations
    tracemalloc.start()

    # Call your function
    for i in range(0, 5):
        prediction = predict(primes[i], primes[i + 1] + 1)[0]

    # Stop tracing and get the memory usage statistics
    memory_stats = tracemalloc.get_traced_memory()

    # Print the results
    print(f"Memory consumption for foo(): {memory_stats[0] / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    your_function_or_loop()