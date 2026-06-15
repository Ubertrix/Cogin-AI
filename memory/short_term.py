class ShortTermMemory:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def get_context(self):
        return " ".join(self.buffer)
