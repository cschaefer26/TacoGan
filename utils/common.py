class Averager:

    def __init__(self):
        self.count = 0
        self.val = 0.

    def add(self, val):
        self.val += float(val)
        self.count += 1

    def reset(self):
        self.val = 0.
        self.count = 0

    def get(self):
        return self.val / self.count

