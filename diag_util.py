import time as time

# Define wrapper to measure time
def measure(function):
    def wrapper(*args,**kwargs):
        start = time.time()
        print("\n--------------------------------")
        print("Calling function: {}".format(function.__name__))
        ret = function(*args,**kwargs)
        print("Exiting function: {}".format(function.__name__))
        elapsed = time.time() - start
        print("Time elapsed: {:.3f}s".format(elapsed))
        return ret
    return wrapper
