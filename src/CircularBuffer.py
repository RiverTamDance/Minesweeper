class CircularBuffer:

#TODO: add a "random" method to get a random sample.

   def __init__(self, max_len):
      self.max_len = max_len
      self.buffer = [None]*max_len
      self.oldest_index = 0

   def __getitem__(self, index):
      return(self.buffer[index])

   def append(self, element):
      self.buffer[self.oldest_index] = element
      self.oldest_index += 1
      self.oldest_index %= self.max_len

   def __repr__(self):
      return(f"CircularBuffer({self.max_len})")

   def __str__(self):
      if len(self.buffer) < 100:
         return(self.buffer.__str__())
      else:
         return(f"{self.buffer[:10]}")