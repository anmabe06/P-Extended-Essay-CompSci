from SequentialNNModel import SequentialNNModel

rangeOfTesting = "R1"
primesNN = SequentialNNModel(f"data/primes{rangeOfTesting}.txt", "Prime Sequence NN", sub_sequence_length=5, csvFilePath=f"results/primes{rangeOfTesting}.txt")
primesNN.compile(units)
primesNN.train(verbose=0, evaluate=False)