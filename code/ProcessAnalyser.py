import time
from datetime import datetime
import numpy as np
from memory_profiler import profile

class ProcessAnalyser:
    def measure(process: callable, params: list = [], measureTime: bool = True, measureMemory: bool = True, timeLog: bool = True, timeLogFilePath: str = "", optionalTimeLogHeader: str = "Time measured") -> tuple[any, float]:
        @profile
        def __helper(helperProcess: callable, helperParams: list = []):
            return helperProcess(*helperParams)
        

        def __logToFile(timeDelta):
            with open(timeLogFilePath) as timeLogFile:
                timeLogFile.write(f"[{datetime.now()}] {optionalTimeLogHeader}: {timeDelta}")

        timeMeasure = float()
        if measureTime:
            if measureMemory:
                initialTime = time.time()
                result = __helper(process, params)
            else:
                initialTime = time.time()
                result = process(*params)
            finalTime = time.time()
            timeMeasure = finalTime - initialTime
            if timeLog:
                __logToFile(f"{timeMeasure}ms")
        elif measureMemory:
            result = __helper(process, params)

        return result, timeMeasure
