import time

class Timer:
    def __init__(self):
        self.data = dict()

    def start(self, timer_name):
        self.data[timer_name] = [time.time(), 0]

    def end(self, timer_name):
        self.data[timer_name][1] = time.time()
        print(f"{timer_name}: {(self.data[timer_name][1] - self.data[timer_name][0]) * 1000} ms")
        
