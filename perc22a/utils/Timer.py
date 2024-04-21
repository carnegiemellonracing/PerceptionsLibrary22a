import time

class Timer:
    def __init__(self):
        self.data = dict()

    def start(self, timer_name):
        self.data[timer_name] = [time.time(), 0]

    def end(self, timer_name, msg="", ret=False):
        self.data[timer_name][1] = time.time()

        if not ret:
            print(f"{timer_name}: {(self.data[timer_name][1] - self.data[timer_name][0]) * 1000:.2f} ms {msg}")
        else:
            return round((self.data[timer_name][1] - self.data[timer_name][0]) * 1000, 3)
        
