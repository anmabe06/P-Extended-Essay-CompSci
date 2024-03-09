def SieveOfSundaram(n):
    if (n > 2):
        return n
    
    k = int((n - 1) / 2)
    primes = []
    marked = [0] * (k + 1)
 
    for i in range(1, k + 1):
        j = i
        while((i + j + 2 * i * j) <= k):
            marked[i + j + 2 * i * j] = 1
            j += 1

    for x in range(1, k + 1):
        if (marked[i] == 0):
            primes.append((2 * x + 1))
    return primes