import random
from collections import deque

class AverageQueue:
    """
    calculates the average of the last maxlen values
    with a sliding window
    """
    def __init__(self, maxlen=1000):
        self.maxlen = maxlen
        self.queue = deque(maxlen=maxlen)
        self.avg = 0
        self.len = 0
        self.sum = 0

    def append(self, value):
        self.sum += value
        if self.len < self.queue.maxlen:
            self.len += 1
        else:
            self.sum -= self.queue.popleft()
        self.queue.append(value)
        self.avg = self.sum / self.maxlen
        return self.avg


"""avg_q= AverageQueue(10)
for i in range(1000):
    print(avg_q.append(i))"""