import time

"""
Timer.py
Creates a timer class that can be used to time the execution of code
- start: starts the timer
    - timer_name: name of the timer
- end: ends the timer and prints the time elapsed
    - timer_name: name of the timer
    - msg: message to print with the time elapsed
    - ret: whether to return the time elapsed
"""

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
        
